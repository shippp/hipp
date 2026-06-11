from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler

from hipp.image import SubImage, match_multiple_templates, remap_tif_blockwise
from hipp.kh9pc.fiducials import KH9ImageSpec, centers_xy_from_boxes, compute_all_fiducial_pattern_scores
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, DetectionError, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.poly import PolyStrategy


@dataclass
class Clustering:
    features: NDArray[np.floating]
    labels: NDArray[np.integer]
    eps: float
    weight: float
    cluster_df: pd.DataFrame


@dataclass
class FiducialResult:
    boxes: NDArray[np.int_]
    scores: NDArray[np.float64]
    clustering: Clustering

    @property
    def centers_xy(self) -> NDArray[np.floating]:
        return centers_xy_from_boxes(self.boxes)


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BLOCK_MARGIN = 0.1  # overlap fraction between adjacent blocks

_KIND_TEMPLATES: dict[str, list[Path]] = {
    "disk": sorted(_TEMPLATE_DIR.glob("disk*.png")),
    "wagon_wheel": sorted(_TEMPLATE_DIR.glob("wagon_wheel.png")),
}


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
    kh9_image_spec:
        Image specification describing which fiducial template set to use
        (``fiducial_type`` field: ``"disk"`` for missions D3C1201–D3C1213,
        ``"wagon_wheel"`` for D3C1214–D3C1219). If ``None`` (default), the spec
        is inferred from the image filename at fit time via
        ``KH9ImageSpec.from_filename()`` and stored in ``self.kh9_image_spec``.
    block_width:
        Width in pixels of each scanning block.
    threshold:
        Minimum template-matching score to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression within a block and globally.
    """

    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    kh9_image_spec: KH9ImageSpec | None = None
    polynomial_degree: int = 7
    block_width: int = 512
    threshold: float = 0.5
    nms_threshold: float = 0.1
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT
    min_score_threshold: float = 0.9

    def __post_init__(self) -> None:
        super().__init__()
        self._templates: list[cv2.typing.MatLike] = (
            _load_kind(self.kh9_image_spec.fiducial_type) if self.kh9_image_spec is not None else []
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
        for result in (self.top_, self.bottom_):
            if not result.clustering.cluster_df["is_good"].any():
                return True
        return False

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
        if self.kh9_image_spec is None:
            self.kh9_image_spec = KH9ImageSpec.from_filename(raster_filepath)
            self._templates = _load_kind(self.kh9_image_spec.fiducial_type)

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

                result = self._filter_outliers(boxes, scores, side)

                if len(result.boxes) == 0:
                    raise DetectionError(f"no inlier cluster found on {side} side of {raster_filepath.name}")

                self._results[side] = result

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

    @staticmethod
    def _compute_detection_features(
        boxes: NDArray[np.int_],
        scores: NDArray[np.float64],
        model: RANSACRegressor,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        centers_xy = centers_xy_from_boxes(boxes)
        residuals = np.abs(centers_xy[:, 1] - model.predict(centers_xy[:, 0].reshape(-1, 1)).ravel())
        features: NDArray[np.floating] = np.column_stack((scores, residuals))
        return centers_xy, features

    @staticmethod
    def _score_clusters(
        labels: NDArray[np.int_],
        centers_xy: NDArray[np.floating],
        detected_width: int,
    ) -> pd.DataFrame:
        results = []
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = labels == label
            if mask.sum() < 5:
                continue
            pattern_scores = compute_all_fiducial_pattern_scores(centers_xy[mask], detected_width)
            pattern, score = max(pattern_scores.items(), key=lambda item: item[1])
            if score > 0.0:
                results.append({"label": label, "pattern": pattern, "score": score})
            else:
                results.append({"label": label, "pattern": None, "score": 0.0})

        return pd.DataFrame(results)

    def _grid_search_clustering(
        self,
        features: NDArray[np.floating],
        centers_xy: NDArray[np.floating],
        detected_width: int,
    ) -> Clustering:
        X_scaled: NDArray[np.floating] = StandardScaler().fit_transform(features)
        best_key: tuple[float, float] = (-1.0, -1.0)
        best: Clustering | None = None

        for rw in np.linspace(0.5, 5, 20):
            X_weighted = (X_scaled * np.array([1.0, rw])).astype(np.float64)
            for eps in np.linspace(0.1, 5, 20):
                labels: NDArray[np.int_] = DBSCAN(eps, min_samples=5).fit(X_weighted).labels_
                df = self._score_clusters(labels, centers_xy, detected_width)
                if df.empty:
                    continue
                top2 = df["score"].nlargest(2).tolist()
                while len(top2) < 2:
                    top2.append(0.0)
                key = (top2[0], top2[1])
                if key > best_key:
                    best_key = key
                    df["is_good"] = df["score"] >= self.min_score_threshold
                    best = Clustering(features=features, labels=labels, eps=eps, weight=rw, cluster_df=df)

        if best is None:
            raise DetectionError("No valid cluster found during grid search")

        return best

    def _filter_outliers(
        self,
        boxes: NDArray[np.int_],
        scores: NDArray[np.float64],
        side: Literal["top", "bottom"],
    ) -> FiducialResult:
        """Identify the two fiducial pattern clusters among raw detections using DBSCAN with a grid search.

        For each side (top/bottom) two distinct fiducial patterns coexist. This method clusters
        detections in a 2-D feature space of (matching score, residual to the polynomial edge model),
        then scores each cluster with ``compute_all_fiducial_pattern_scores`` and selects the
        configuration that maximises the best cluster score, with the second-best as tiebreaker
        (lexicographic comparison). The two top-scoring clusters are returned as inliers, each
        labelled with its best matching fiducial pattern.

        Parameters
        ----------
        boxes:
            Detection boxes ``[x, y, w, h]`` in global raster coordinates, after NMS.
        scores:
            Corresponding template-matching scores.
        side:
            Which edge model to use for residual computation.

        Returns
        -------
        FiducialResult
            Inlier boxes and scores from the two selected clusters, pattern label per cluster,
            and the full DBSCAN state at the best (eps, weight) parameters.
        """
        model = self.poly_strategy.top_.model if side == "top" else self.poly_strategy.bottom_.model
        detected_width = int(self.poly_strategy.vertical_detector.detected_width_)

        centers_xy, features = self._compute_detection_features(boxes, scores, model)
        clustering = self._grid_search_clustering(features, centers_xy, detected_width)

        return FiducialResult(boxes=boxes, scores=scores, clustering=clustering)

    def _compute_transformation(self) -> Transformation:
        raise NotImplementedError
