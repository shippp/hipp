from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self, Sequence

import cv2
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from hipp.image import match_multiple_templates
from hipp.kh9pc.restitution_strategy.poly_strategy import PolyStrategy
from hipp.kh9pc.types import FiducialFilteringResult, FiducialResult, RestitutionStrategy, Transformation
from hipp.kh9pc.utils import SubImage, compute_spatial_regularization_score

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_Side = Literal["top", "bottom"]
_BLOCK_MARGIN = 0.1  # overlap fraction between adjacent blocks


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
    template_paths:
        Paths to PNG template images. Defaults to all PNGs in the ``templates/``
        directory next to this file.
    block_width:
        Width in pixels of each scanning block.
    threshold:
        Minimum template-matching score to keep a detection.
    nms_threshold:
        IoU threshold for non-maximum suppression within a block and globally.
    """

    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    template_paths: Sequence[str | Path] = field(default_factory=lambda: list(_TEMPLATE_DIR.glob("*.png")))
    block_width: int = 512
    threshold: float = 0.7
    nms_threshold: float = 0.1

    def __post_init__(self) -> None:
        super().__init__()
        self._templates = [
            img for p in self.template_paths if (img := cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) is not None
        ]
        self._results: dict[str, FiducialResult] = {}

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
        return False

    @property
    def transformation_(self) -> Transformation:
        raise NotImplementedError

    def transform(self, output_path: str | Path) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)

        col_start, col_end = self.poly_strategy.vertical_detector.edges_

        with rasterio.open(raster_filepath) as src:
            for side in ("top", "bottom"):
                boxes, scores, ids = self._scan_side(src, col_start, col_end, side)

                # apply NMS to remove duplicate detection
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, self.nms_threshold)
                keep = np.array(indices).reshape(-1)
                boxes, scores, ids = boxes[keep], scores[keep], ids[keep]

                filtering = self._filter_outliers(boxes, scores, ids, side)
                keep = np.where(filtering.labels == filtering.best_cluster_label)[0]
                self._results[side] = FiducialResult(
                    boxes=boxes[keep],
                    scores=scores[keep],
                    template_ids=ids[keep],
                    filtering=filtering,
                )

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_side(
        self, src: rasterio.DatasetReader, col_start: int, col_end: int, side: _Side
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
            # expand block by _BLOCK_MARGIN on each side to avoid missing fiducials at boundaries
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
        side: _Side,
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
        best_eps = 0.0
        best_weight = 0.0
        best_labels = np.full(len(boxes), -1, dtype=np.intp)  # default: all noise
        best_cluster_label = -1

        for rw in np.linspace(0.5, 5, 20):
            # amplify the residual dimension relative to the score dimension
            features = X_scaled * np.array([1.0, rw])
            for eps in np.linspace(0.1, 5, 20):
                labels = DBSCAN(eps, min_samples=5).fit(features).labels_

                for label in np.unique(labels):
                    if label == -1:  # DBSCAN noise points
                        continue
                    mask = labels == label

                    # fraction of the total image width covered by this cluster
                    width_fraction = (np.max(cx[mask]) - np.min(cx[mask])) / detected_width

                    # prefer clusters that are both spatially regular and horizontally spread
                    score = compute_spatial_regularization_score(cx[mask], cy[mask]) * width_fraction

                    if score >= best_score:
                        best_score = score
                        best_eps = eps
                        best_weight = rw
                        best_labels = labels.copy()
                        best_cluster_label = label

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
        )
