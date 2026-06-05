import re
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
from sklearn.preprocessing import StandardScaler

from hipp.image import match_multiple_templates, remap_tif_blockwise
from hipp.kh9pc.restitution_strategy.poly_strategy import PolyStrategy
from hipp.kh9pc.types import (
    DEFAULT_OUTPUT_HEIGHT,
    DetectionError,
    FiducialFilteringResult,
    FiducialResult,
    RestitutionStrategy,
    Transformation,
)
from hipp.kh9pc.utils import SubImage, compute_spatial_regularization_score

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BLOCK_MARGIN = 0.1  # overlap fraction between adjacent blocks

# D3C12(01-13) → disk fiducials, D3C12(14-19) → wagon-wheel fiducials
_MISSION_RE = re.compile(r"D3C12(\d{2})")
_KIND_TEMPLATES: dict[str, list[Path]] = {
    "disk": sorted(_TEMPLATE_DIR.glob("disk*.png")),
    "wagon_wheel": sorted(_TEMPLATE_DIR.glob("wagon_wheel.png")),
}


def _infer_kind(stem: str) -> Literal["disk", "wagon_wheel"]:
    m = _MISSION_RE.search(stem)
    if m is None:
        raise DetectionError(f"Cannot infer template kind from {stem!r}: filename must match D3C12XX")
    n = int(m.group(1))
    if 1 <= n <= 13:
        return "disk"
    if 14 <= n <= 19:
        return "wagon_wheel"
    raise DetectionError(f"Unknown KH-9 mission D3C12{n:02d} in {stem!r} — expected 01–19")


def _load_kind(kind: str) -> list[cv2.typing.MatLike]:
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
    template_kind:
        Which fiducial template set to use: ``"disk"`` (missions D3C1201–D3C1213),
        ``"wagon_wheel"`` (missions D3C1214–D3C1219), or ``"auto"`` to infer from
        the image filename (default).  Raises ``DetectionError`` if auto-detection
        fails or the resolved template set is empty.
    block_width:
        Width in pixels of each scanning block.
    threshold:
        Minimum template-matching score to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression within a block and globally.
    """

    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    template_kind: Literal["auto", "disk", "wagon_wheel"] = "auto"
    polynomial_degree: int = 7
    block_width: int = 512
    threshold: float = 0.5
    nms_threshold: float = 0.1
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT
    min_fiducials: int = 10
    min_width_coverage: float = 0.7

    def __post_init__(self) -> None:
        super().__init__()
        self._templates: list[cv2.typing.MatLike] = (
            [] if self.template_kind == "auto" else _load_kind(self.template_kind)
        )
        self._results: dict[str, FiducialResult] = {}
        self.__transformation_: Transformation | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def top_(self) -> FiducialResult:
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> FiducialResult:
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def is_failed(self) -> bool:
        return (
            len(self.top_.centers) < self.min_fiducials
            or len(self.bottom_.centers) < self.min_fiducials
            or self.top_.width_coverage < self.min_width_coverage
            or self.bottom_.width_coverage < self.min_width_coverage
        )

    @property
    def transformation_(self) -> Transformation:
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def transform(self, output_path: str | Path) -> None:
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
        if self.template_kind == "auto":
            self._templates = _load_kind(_infer_kind(raster_filepath.stem))

        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)

        col_start, col_end = self.poly_strategy.vertical_detector.edges_

        with rasterio.open(raster_filepath) as src:
            for side in ("top", "bottom"):
                boxes, scores, ids = self._scan_side(src, 0, col_end, side)

                # apply NMS to remove duplicate detection
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, self.nms_threshold)
                keep = np.array(indices).reshape(-1).astype(int)
                boxes, scores, ids = boxes[keep], scores[keep], ids[keep]

                if len(boxes) == 0:
                    raise DetectionError(f"no fiducials detected on {side} side of {raster_filepath.name}")

                # filter outliers by applying clustering to find the better cluster
                filtering = self._filter_outliers(boxes, scores, ids, side)
                keep = np.where(filtering.labels == filtering.best_cluster_label)[0]
                boxes, scores, ids = boxes[keep], scores[keep], ids[keep]

                if len(boxes) == 0:
                    raise DetectionError(f"no inlier cluster found on {side} side of {raster_filepath.name}")

                cx = (boxes[:, 0] + 0.5 * boxes[:, 2]).astype(np.intp)
                cy = (boxes[:, 1] + 0.5 * boxes[:, 3]).astype(np.intp)
                centers = np.column_stack([cx, cy])

                poly = np.polynomial.Polynomial.fit(cx, cy, self.polynomial_degree)
                x = np.linspace(cx.min(), cx.max(), 100)
                y = poly(x)
                distortion = np.column_stack([x, y - y.mean()])
                width_coverage = float((cx.max() - cx.min()) / (col_end - col_start))

                self._results[side] = FiducialResult(
                    centers=centers,
                    poly=poly,
                    boxes=boxes,
                    distortion=distortion,
                    scores=scores,
                    template_ids=ids,
                    width_coverage=width_coverage,
                    filtering=filtering,
                )

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
                templates=self._templates,
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

    def _filter_outliers(
        self,
        boxes: NDArray[np.int_],
        scores: NDArray[np.float64],
        template_ids: NDArray[np.int_],
        side: Literal["top", "bottom"],
    ) -> FiducialFilteringResult:
        """Identify the inlier cluster among raw detections using DBSCAN with a grid search.

        After NMS, spurious detections can still remain (e.g. template matches on image
        artifacts far from the actual fiducial strip). This method clusters detections in a
        2-D feature space of (matching score, residual to the polynomial edge model) and
        selects the cluster whose spatial distribution best covers the full image width.

        The grid search explores 20×20 combinations of ``eps`` (DBSCAN neighbourhood radius)
        and ``residual_weight`` (relative importance of the residual feature vs. the score).
        For each combination every non-noise cluster is evaluated with
        ``compute_spatial_regularization_score * width_fraction``; the globally best
        (parameters, cluster) pair is retained.

        Parameters
        ----------
        boxes:
            Detection boxes ``[x, y, w, h]`` in global raster coordinates, after NMS.
        scores:
            Corresponding template-matching scores.
        template_ids:
            Index into ``self._templates`` for each detection.
        side:
            Which edge model to use for residual computation.

        Returns
        -------
        FiducialFilteringResult
            Full clustering state (all boxes, labels, residuals, best parameters) needed
            to reconstruct both inliers and outliers for plotting or debugging.
            The caller extracts inliers with ``labels == best_cluster_label``.
        """
        model = self.poly_strategy.top_.model if side == "top" else self.poly_strategy.bottom_.model

        # box centre coordinates in global raster space
        cx = boxes[:, 0] + 0.5 * boxes[:, 2]
        cy = boxes[:, 1] + 0.5 * boxes[:, 3]

        # vertical distance from each detection centre to the fitted polynomial edge
        residuals = np.abs(cy - model.predict(cx.reshape(-1, 1)).ravel())

        # reference width used to normalise the spatial coverage score
        edges = self.poly_strategy.vertical_detector.edges_
        detected_width = edges[1] - edges[0]

        # standardise so that score and residual are on the same scale before weighting
        X_scaled = StandardScaler().fit_transform(np.column_stack((scores, residuals)))

        best_score = -np.inf
        best_secondary_score = -np.inf
        best_eps = 0.0
        best_weight = 0.0
        best_labels = np.full(len(boxes), -1, dtype=np.intp)  # default: all noise
        best_cluster_label = -1

        for rw in np.linspace(0.5, 5, 20):
            # amplify the residual dimension relative to the score dimension
            features = X_scaled * np.array([1.0, rw])
            for eps in np.linspace(0.1, 5, 20):
                labels = DBSCAN(eps, min_samples=5).fit(features).labels_

                cluster_scores_current: dict[int, float] = {}
                for label in np.unique(labels):
                    if label == -1:  # DBSCAN noise points
                        continue
                    mask = labels == label
                    if mask.sum() < 5:
                        continue

                    # fraction of the total image width covered by this cluster
                    width_fraction = (np.max(cx[mask]) - np.min(cx[mask])) / detected_width

                    # prefer clusters that are both spatially regular and horizontally spread
                    cluster_scores_current[int(label)] = (
                        compute_spatial_regularization_score(cx[mask], cy[mask]) * width_fraction
                    )

                if not cluster_scores_current:
                    continue

                sorted_clusters = sorted(cluster_scores_current.items(), key=lambda kv: kv[1], reverse=True)
                primary_label, primary_score = sorted_clusters[0]
                secondary_score = sorted_clusters[1][1] if len(sorted_clusters) > 1 else 0.0

                # lexicographic comparison: best cluster is primary, second best is tiebreaker
                if (primary_score, secondary_score) > (best_score, best_secondary_score):
                    best_score = primary_score
                    best_secondary_score = secondary_score
                    best_eps = eps
                    best_weight = rw
                    best_labels = labels.copy()
                    best_cluster_label = primary_label

        # compute the spatial score for every cluster at the best (eps, weight) params
        cluster_scores: dict[int, float] = {}
        for label in np.unique(best_labels):
            if label == -1:
                continue
            mask = best_labels == label
            if mask.sum() < 5:
                continue
            width_fraction = float((np.max(cx[mask]) - np.min(cx[mask])) / detected_width)
            cluster_scores[int(label)] = compute_spatial_regularization_score(cx[mask], cy[mask]) * width_fraction

        return FiducialFilteringResult(
            boxes_all=boxes,
            scores_all=scores,
            template_ids_all=template_ids,
            cx=cx,
            cy=cy,
            residuals=residuals,
            labels=best_labels,
            best_cluster_label=best_cluster_label,
            best_eps=best_eps,
            best_weight=best_weight,
            cluster_scores=cluster_scores,
        )

    def _compute_transformation(self) -> Transformation:
        left, right = self.poly_strategy.vertical_detector.edges_
        detected_width = right - left
        output_width = self.output_width or detected_width

        n_points = self.poly_strategy.grid_shape[0]

        # Restrict control points to the range actually covered by detected fiducials on each
        # side — extrapolating the high-degree polynomial beyond that range causes instability.
        cx_top = self.top_.centers[:, 0]
        cx_bot = self.bottom_.centers[:, 0]
        x_top = np.linspace(cx_top.min(), cx_top.max(), n_points)
        x_bot = np.linspace(cx_bot.min(), cx_bot.max(), n_points)

        # High-degree fiducial polynomials as control points — more precise than edge model
        y_top_src = self.top_.poly(x_top)
        y_bot_src = self.bottom_.poly(x_bot)

        y_top_dst = np.full_like(x_top, y_top_src.mean())
        y_bot_dst = np.full_like(x_bot, y_bot_src.mean())

        src = np.column_stack((np.concatenate((x_top, x_bot)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x_top, x_bot)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important)
        deformation = ThinPlateSplineTransform().from_estimate(dst, src)

        # Image boundaries
        top, bot = int(np.mean(y_top_src)), int(np.mean(y_bot_src))
        detected_height = bot - top
        output_height = self.output_height or detected_height

        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2
        crop_offset = (int(left - pad_x), int(top - pad_y))

        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )
