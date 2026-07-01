"""
Copyright (c) 2026 HIPP developers
Description: FiducialStrategy — highest-quality restitution strategy. Detects fiducial
    markers above and below the image using template matching, clusters detections into
    known pattern types via DBSCAN grid search, and fits a Thin Plate Spline transform
    to map detected fiducial centers to their known physical positions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from skimage.transform import ThinPlateSplineTransform
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler

from hipp.image import SubImage, match_multiple_templates, remap_tif_blockwise
from hipp.kh9pc.fiducial_patterns import (
    Patterns,
    DetectedPattern,
    centers_xy_from_boxes,
    compute_global_src_and_dst_points,
    evaluate_pattern,
)
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, DetectionError, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy


@dataclass
class FiducialResult:
    """Template matching results and pattern classifications for one film edge side.

    Attributes
    ----------
    boxes:
        (N, 4) detected bounding boxes in global raster coordinates ``[x, y, w, h]``.
    scores:
        (N,) template matching scores for each detection.
    patterns:
        Mapping from pattern name to its best ``DetectedPattern`` after DBSCAN grid search.
    features:
        (N, 2) feature matrix used for clustering: ``[matching_score, residual_to_edge_model]``.
    """

    boxes: NDArray[np.int_]
    scores: NDArray[np.floating]
    patterns: dict[str, DetectedPattern]
    features: NDArray[np.floating]  # (N, 2): (matching_score, residual_to_edge)

    @property
    def centers_xy(self) -> NDArray[np.floating]:
        """(N, 2) center coordinates of all detected bounding boxes."""
        return centers_xy_from_boxes(self.boxes)


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BLOCK_MARGIN = 0.1  # overlap fraction between adjacent blocks

_KIND_TEMPLATES: dict[str, list[Path]] = {
    "disk": sorted(_TEMPLATE_DIR.glob("disk*.png")),
    "wagon_wheel": sorted(_TEMPLATE_DIR.glob("wagon_wheel.png")),
}


def _load_kind(kind: str) -> list[cv2.typing.MatLike]:
    """Load all grayscale template images for *kind* (``"disk"`` or ``"wagon_wheel"``)."""
    paths = _KIND_TEMPLATES.get(kind, [])
    templates = [img for p in paths if (img := cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) is not None]
    if not templates:
        raise DetectionError(f"No templates loaded for kind {kind!r} — check {_TEMPLATE_DIR}")
    return templates


@dataclass
class FiducialStrategy(RestitutionStrategy):
    """Detect fiducial markers above and below the image area using template matching.

    The strategy relies on a fitted ``PolyStrategy`` to locate the top/bottom image
    edges, then slides overlapping blocks along the horizontal axis and runs
    multi-template matching on the strip above the top edge and below the bottom
    edge. A final global NMS pass deduplicates matches that span adjacent blocks.

    Parameters
    ----------
    poly_strategy:
        Fitted (or to-be-fitted) strategy that provides the horizontal edge models.
    kh9_image_spec:
        Image specification describing which fiducial template set to use
        (``fiducial_type`` field: ``"disk"`` for missions D3C1201–D3C1213,
        ``"wagon_wheel"`` for D3C1214–D3C1219). If ``None`` (default), the spec
        is inferred from the image filename at fit time via
    block_width:
        Width in pixels of each scanning block.
    threshold:
        Minimum template-matching score to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression within a block and globally.
    horizontal_margins:
        ``(left, right)`` fractional margins relative to the detected image width.
        The search window is inset by ``left * width`` on the left and
        ``right * width`` on the right of the vertical edges.
    """

    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    polynomial_degree: int = 7
    block_width: int = 512
    threshold: float = 0.5
    nms_threshold: float = 0.1
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT
    min_score_threshold: float = 0.8

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, FiducialResult] = {}
        self.__transformation_: Transformation | None = None
        self.kh9_image_spec_: KH9ImageSpec | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def top_(self) -> FiducialResult:
        """Detection results for the top film edge. Raises if ``fit()`` has not been called."""
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> FiducialResult:
        """Detection results for the bottom film edge. Raises if ``fit()`` has not been called."""
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def is_failed(self) -> bool:
        """True if both primary patterns (top and bottom) score below ``min_score_threshold``."""
        if self.kh9_image_spec_ is None:
            raise RuntimeError("Call fit() before")
        primary_top_pattern = self.kh9_image_spec_.top_fiducial_patterns[0]
        primary_bottom_pattern = self.kh9_image_spec_.bottom_fiducial_patterns[0]

        return (
            self.top_.patterns[primary_top_pattern].score < self.min_score_threshold
            and self.bottom_.patterns[primary_bottom_pattern].score < self.min_score_threshold
        )

    @property
    def transformation_(self) -> Transformation:
        """TPS Transformation from detected fiducial positions to known physical positions (computed lazily)."""
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def transform(self, output_path: str | Path) -> None:
        """Write the restituted image using the fiducial-based TPS warp."""
        tf = self.transformation_
        remap_tif_blockwise(
            tf.raster_filepath,
            output_path,
            tf.inverse_remap,
            tf.output_size,
            block_size=2**13,
            lowres_step=100,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(self, raster_filepath: Path) -> Self:
        """Infer image spec, run PolyStrategy, then scan both sides for fiducial detections."""
        self.kh9_image_spec_ = KH9ImageSpec.from_raster_filepath(raster_filepath)
        self.templates_ = _load_kind(self.kh9_image_spec_.fiducial_type)

        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)

        _, col_end = self.poly_strategy.vertical_detector.edges_

        with rasterio.open(raster_filepath) as src:
            for side in ("top", "bottom"):
                # scan from column 0, not col_left: fiducials can appear in the dark film margin before the effective image
                boxes, scores, ids = self._scan_side(src, 0, col_end, side)

                # apply NMS to remove duplicate detection
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, self.nms_threshold)
                keep = np.array(indices).reshape(-1).astype(int)
                boxes, scores, ids = boxes[keep], scores[keep], ids[keep]

                if len(boxes) == 0:
                    raise DetectionError(f"no fiducials detected on {side} side of {raster_filepath.name}")

                patterns, features = self._search_patterns(boxes, scores, side)

                self._results[side] = FiducialResult(boxes, scores, patterns, features)

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_side(
        self, src: rasterio.DatasetReader, col_start: int, col_end: int, side: Literal["top", "bottom"]
    ) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]]:
        """Slide overlapping blocks across [col_start, col_end] for one side.

        Each block overlaps its neighbours by ``_BLOCK_MARGIN`` so that fiducials
        near block boundaries are not missed. Detections from all blocks are merged
        into a single array; global NMS is applied by the caller.

        Parameters
        ----------
        src:
            Open rasterio dataset to read image strips from.
        col_start, col_end:
            Column bounds of the active image area (from the vertical detector).
        side:
            ``"top"`` scans the strip above the top edge; ``"bottom"`` scans below.

        Returns
        -------
        tuple of (boxes, scores, template_ids) in global raster coordinates.
        """
        boxes: list[list[int]] = []
        scores: list[float] = []
        template_ids: list[int] = []

        for cursor in range(col_start, col_end, self.block_width):
            # left_boundary extends 2 % of the detected width to the left of col_start
            w_start = max(col_start, int(cursor - self.block_width * _BLOCK_MARGIN))
            w_end = min(col_end, int(cursor + self.block_width * (1 + _BLOCK_MARGIN)))
            w_width = w_end - w_start
            w_center = w_start + w_width // 2

            # strip above the top edge: rows 0 → predicted top row
            if side == "top":
                top_row = int(self.poly_strategy.top_.model.predict(np.array([[w_center]])).flat[0])
                window = Window(w_start, 0, w_width, top_row)
            # strip below the bottom edge: predicted bottom row → end of raster
            else:
                bot_row = int(self.poly_strategy.bottom_.model.predict(np.array([[w_center]])).flat[0])
                window = Window(w_start, bot_row, w_width, src.height - bot_row)

            if window.height <= 0 or window.width <= 0:
                continue

            sub_image = SubImage(src, window)
            local_boxes, block_scores, block_ids = match_multiple_templates(
                image=sub_image.band,
                templates=self.templates_,
                threshold=self.threshold,
                nms_threshold=self.nms_threshold,
            )

            # convert local block coordinates to global raster coordinates
            for x, y, w, h in local_boxes:
                gx, gy = sub_image.to_global(np.array([x, y], dtype=np.float64))
                boxes.append([int(gx), int(gy), w, h])
            scores.extend(block_scores)
            template_ids.extend(block_ids)

        np_boxes = np.array(boxes, dtype=np.int_).reshape(-1, 4) if boxes else np.empty((0, 4), dtype=np.int_)
        return np_boxes, np.array(scores, dtype=np.float64), np.array(template_ids, dtype=np.int_)

    @staticmethod
    def _compute_detection_features(
        boxes: NDArray[np.int_],
        scores: NDArray[np.float64],
        model: RANSACRegressor,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return (centers_xy, features) where features are [matching_score, residual_to_edge]."""
        centers_xy = centers_xy_from_boxes(boxes)
        residuals = np.abs(centers_xy[:, 1] - model.predict(centers_xy[:, 0].reshape(-1, 1)).ravel())
        features: NDArray[np.floating] = np.column_stack((scores, residuals))
        return centers_xy, features

    def _score_clusters(
        self,
        labels: NDArray[np.int_],
        centers_xy: NDArray[np.floating],
        fiducial_pattern: Patterns,
    ) -> DetectedPattern:
        """Evaluate all DBSCAN clusters for one pattern type and return the highest-scoring one."""
        assert self.kh9_image_spec_ is not None
        expected_width = self.kh9_image_spec_.expected_size[0]
        result = evaluate_pattern(fiducial_pattern, np.empty((0, 2), dtype=np.float64), expected_width)

        for label in np.unique(labels):
            if label == -1:
                continue
            mask = labels == label
            if mask.sum() < 5:
                continue

            detected_pattern = evaluate_pattern(fiducial_pattern, centers_xy[mask], expected_width)
            if result.score < detected_pattern.score:
                result = detected_pattern

        return result

    def _grid_search_clustering(
        self,
        features: NDArray[np.floating],
        centers_xy: NDArray[np.floating],
        fiducial_patterns: tuple[Patterns, Patterns],
    ) -> dict[str, DetectedPattern]:
        """Grid search over DBSCAN (eps, residual weight) to maximise the score of each pattern."""
        assert self.kh9_image_spec_ is not None
        X_scaled: NDArray[np.floating] = StandardScaler().fit_transform(features)
        patterns: dict[str, DetectedPattern] = {
            pt: evaluate_pattern(pt, np.empty((0, 2), dtype=np.float64), self.kh9_image_spec_.expected_size[0])
            for pt in fiducial_patterns
        }

        for rw in np.linspace(0.5, 5, 20):
            X_weighted = (X_scaled * np.array([1.0, rw])).astype(np.float64)
            for eps in np.linspace(0.1, 5, 20):
                labels: NDArray[np.int_] = DBSCAN(eps, min_samples=5).fit(X_weighted).labels_

                for pt in fiducial_patterns:
                    pattern = self._score_clusters(labels, centers_xy, pt)
                    if patterns[pt].score < pattern.score:
                        patterns[pt] = pattern

        return patterns

    def _search_patterns(
        self,
        boxes: NDArray[np.int_],
        scores: NDArray[np.float64],
        side: Literal["top", "bottom"],
    ) -> tuple[dict[str, DetectedPattern], NDArray[np.floating]]:
        """Compute detection features for one side and run grid-search clustering to classify patterns."""
        assert self.kh9_image_spec_ is not None
        if side == "top":
            model = self.poly_strategy.top_.model
            fiducial_patterns = self.kh9_image_spec_.top_fiducial_patterns
        else:
            model = self.poly_strategy.bottom_.model
            fiducial_patterns = self.kh9_image_spec_.bottom_fiducial_patterns

        centers_xy, features = self._compute_detection_features(boxes, scores, model)
        patterns = self._grid_search_clustering(features, centers_xy, fiducial_patterns)

        return patterns, features

    def _compute_transformation(self) -> Transformation:
        """Build the TPS Transformation from matched fiducial src→dst control points.

        Only primary patterns (with known physical spacing) are used as control points.
        If one side failed, its pattern is synthesised from the other using the known
        physical row spacing. A forward TPS is used to map vertical edges into destination
        space so the crop is centred correctly.
        """
        assert self.kh9_image_spec_ is not None
        if self.is_failed:
            raise DetectionError("Can't compute the transformation with a failed estimation")

        # use only primary patterns (sparse & mid) cause we know the theorical spacing
        primary_top_pattern = self.kh9_image_spec_.top_fiducial_patterns[0]
        primary_bottom_pattern = self.kh9_image_spec_.bottom_fiducial_patterns[0]

        top = self.top_.patterns[primary_top_pattern]
        bottom = self.bottom_.patterns[primary_bottom_pattern]

        # provid only valid pattern else None
        src_pts, dst_pts = compute_global_src_and_dst_points(
            top if top.score >= self.min_score_threshold else None,
            bottom if bottom.score >= self.min_score_threshold else None,
        )

        y_center = (dst_pts[:, 1].min() + dst_pts[:, 1].max()) / 2

        # map vertical edges from src space to dst space to get a correct x_center
        forward_tps = ThinPlateSplineTransform().from_estimate(src_pts, dst_pts)
        col_left, col_right = self.poly_strategy.vertical_detector.edges_
        edges_dst = forward_tps(np.array([[col_left, y_center], [col_right, y_center]], dtype=np.float32))
        x_center = float((edges_dst[0, 0] + edges_dst[1, 0]) / 2)

        final_width, final_height = self.kh9_image_spec_.expected_size
        crop_offset = (int(x_center - final_width / 2), int(y_center - final_height / 2))

        # inverse source destination (important)
        deformation = ThinPlateSplineTransform().from_estimate(dst_pts, src_pts)

        # test for the moment without any crop to detect an other time for quality control and qc
        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(final_width, final_height),
        )
