import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import BaseModel
from rasterio.windows import Window
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from hipp.image import SubImage, match_multiple_templates
from hipp.kh9pc.detection.base import DetectionError, DetectionResult
from hipp.kh9pc.detection.detect_poly_horizontal_edges import PolyDetectionResult
from hipp.kh9pc.fiducial_patterns import Patterns, centers_xy_from_boxes, evaluate_pattern
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.tools import skip_existing

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BLOCK_MARGIN = 0.1  # overlap fraction between adjacent blocks


class FiducialOptions(BaseModel):
    """Parameters of the fiducial detection."""

    block_width: int = 512
    threshold: float = 0.5
    nms_threshold: float = 0.1
    min_score_threshold: float = 0.8


class FiducialPattern(BaseModel):
    """Best cluster of fiducial centers found for one pattern type."""

    points: list[tuple[float, float]]  # global coordinates
    score: float

    @property
    def count(self) -> int:
        """Number of fiducial centers in this pattern."""
        return len(self.points)


class FiducialSide(BaseModel):
    """Template matches above (top) or below (bottom) the film edge and the patterns found among them."""

    edge_coefficients: list[float]  # np.polyval order, film edge the strip was searched against
    boxes: list[tuple[int, int, int, int]]  # global [x, y, w, h], after NMS
    scores: list[float]
    patterns: dict[Patterns, FiducialPattern]

    @property
    def centers_xy(self) -> NDArray[np.floating]:
        """(N, 2) center coordinates of all detected boxes."""
        return centers_xy_from_boxes(np.array(self.boxes))

    @property
    def features(self) -> NDArray[np.floating]:
        """(N, 2) clustering features: matching score and distance to the film edge."""
        centers = self.centers_xy
        residuals = np.abs(centers[:, 1] - np.polyval(self.edge_coefficients, centers[:, 0]))
        return np.column_stack((self.scores, residuals))


class FiducialDetectionResult(DetectionResult):
    """Top and bottom fiducial detections of a raster whose vertical film edges are left and right."""

    raster_filepath: Path
    options: FiducialOptions
    left: int
    right: int
    top: FiducialSide
    bottom: FiducialSide

    @property
    def primary_pattern_names(self) -> tuple[Patterns, Patterns]:
        """Names of the top and bottom primary patterns, the only ones with a known theoretical spacing."""
        mission = KH9ImageSpec.mission_from_filepath(self.raster_filepath)
        return (
            KH9ImageSpec.top_fiducial_patterns_from_mission(mission)[0],
            KH9ImageSpec.bottom_fiducial_patterns_from_mission(mission)[0],
        )

    @property
    def failure_reason(self) -> str | None:
        """Why the detection is not trustworthy, or None if it is."""
        top_name, bottom_name = self.primary_pattern_names
        top_score, bottom_score = self.top.patterns[top_name].score, self.bottom.patterns[bottom_name].score
        if max(top_score, bottom_score) < self.options.min_score_threshold:
            return (
                f"both primary fiducial patterns score below {self.options.min_score_threshold} "
                f"(top={top_score:.3f}, bottom={bottom_score:.3f})"
            )
        return None


def detect_fiducials(
    raster_filepath: str | Path,
    output_file: str | Path,
    poly: str | Path | PolyDetectionResult,
    overwrite: bool = False,
    options: FiducialOptions | None = None,
) -> None:
    """Template-match fiducials outside the film edges, classify them into patterns, and write the result as JSON.

    The top strip spans from row 0 to the top film edge fitted in *poly*, the bottom strip from its bottom edge to the
    last row. Columns [0, right) are searched, *right* being the right film edge of *poly*, since fiducials can sit in
    the dark margin left of the image.
    Raises DetectionError if a side has no match. A detection whose primary patterns both score below
    ``min_score_threshold`` is still written, with ``is_failed`` set.
    Skipped if *output_file* already exists and *overwrite* is False.
    """
    raster_filepath = Path(raster_filepath)
    output_file = Path(output_file)

    if skip_existing(output_file, overwrite):
        return

    poly = PolyDetectionResult.load(poly)
    left, right = poly.left, poly.right

    options = options or FiducialOptions()
    spec = KH9ImageSpec.from_raster_filepath(raster_filepath)
    templates = load_templates(spec.fiducial_type)

    with rasterio.open(raster_filepath) as src:
        top = _detect_side(src, 0, right, poly.top.coefficients, "top", templates, spec, options)
        bottom = _detect_side(src, 0, right, poly.bottom.coefficients, "bottom", templates, spec, options)

    result = FiducialDetectionResult(
        raster_filepath=raster_filepath, options=options, left=left, right=right, top=top, bottom=bottom
    )

    top_name, bottom_name = result.primary_pattern_names
    logger.info(
        "%s - fiducials detected: top %s score=%.3f n=%d, bottom %s score=%.3f n=%d",
        raster_filepath.name,
        top_name,
        top.patterns[top_name].score,
        top.patterns[top_name].count,
        bottom_name,
        bottom.patterns[bottom_name].score,
        bottom.patterns[bottom_name].count,
    )
    if result.failure_reason:
        logger.warning("%s - fiducial detection failed: %s", raster_filepath.name, result.failure_reason)

    result.save(output_file)


def load_templates(kind: Literal["disk", "wagon_wheel"]) -> list[cv2.typing.MatLike]:
    """Load all grayscale template images for one fiducial *kind*."""
    paths = sorted(_TEMPLATE_DIR.glob("disk*.png" if kind == "disk" else "wagon_wheel.png"))
    templates = [img for p in paths if (img := cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) is not None]
    if not templates:
        raise DetectionError(f"No templates loaded for kind {kind!r} — check {_TEMPLATE_DIR}")
    return templates


def _detect_side(
    src: rasterio.DatasetReader,
    left: int,
    right: int,
    edge_coefficients: list[float],
    side: Literal["top", "bottom"],
    templates: list[cv2.typing.MatLike],
    spec: KH9ImageSpec,
    options: FiducialOptions,
) -> FiducialSide:
    """Match templates along one side, deduplicate with a global NMS, then classify the matches into patterns."""
    boxes, scores = _scan_side(src, left, right, edge_coefficients, side, templates, options)

    # blocks overlap, so the same fiducial can be matched twice
    keep = np.array(
        cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), options.threshold, options.nms_threshold)
    ).reshape(-1)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        raise DetectionError(f"no fiducials detected on {side} side")

    fiducial_side = FiducialSide(
        edge_coefficients=edge_coefficients,
        boxes=[(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes],
        scores=scores.tolist(),
        patterns={},
    )
    pattern_names = spec.top_fiducial_patterns if side == "top" else spec.bottom_fiducial_patterns
    fiducial_side.patterns = _grid_search_patterns(
        fiducial_side.features, fiducial_side.centers_xy, pattern_names, spec.expected_size[0]
    )
    return fiducial_side


def _scan_side(
    src: rasterio.DatasetReader,
    left: int,
    right: int,
    edge_coefficients: list[float],
    side: Literal["top", "bottom"],
    templates: list[cv2.typing.MatLike],
    options: FiducialOptions,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Slide overlapping blocks across [left, right) and return all template matches in global coordinates."""
    boxes: list[list[int]] = []
    scores: list[float] = []

    for cursor in range(left, right, options.block_width):
        # each block overlaps its neighbours so fiducials near block boundaries are not missed
        w_start = max(left, int(cursor - options.block_width * _BLOCK_MARGIN))
        w_end = min(right, int(cursor + options.block_width * (1 + _BLOCK_MARGIN)))
        edge_row = int(np.polyval(edge_coefficients, w_start + (w_end - w_start) / 2))

        if side == "top":
            window = Window(w_start, 0, w_end - w_start, edge_row)
        else:
            window = Window(w_start, edge_row, w_end - w_start, src.height - edge_row)
        if window.height <= 0 or window.width <= 0:
            continue

        sub_image = SubImage(src, window)
        local_boxes, block_scores, _ = match_multiple_templates(
            image=sub_image.band, templates=templates, threshold=options.threshold, nms_threshold=options.nms_threshold
        )
        for x, y, w, h in local_boxes:
            gx, gy = sub_image.to_global(np.array([x, y], dtype=np.float64))
            boxes.append([int(gx), int(gy), w, h])
        scores.extend(block_scores)

    return np.array(boxes, dtype=np.int_).reshape(-1, 4), np.array(scores, dtype=np.float64)


def _grid_search_patterns(
    features: NDArray[np.floating],
    centers_xy: NDArray[np.floating],
    pattern_names: tuple[Patterns, Patterns],
    expected_width: int,
) -> dict[Patterns, FiducialPattern]:
    """Grid search over DBSCAN (eps, residual weight), keeping the best-scoring cluster for each pattern."""
    empty = np.empty((0, 2), dtype=np.float64)
    best = {name: evaluate_pattern(name, empty, expected_width) for name in pattern_names}
    X_scaled: NDArray[np.floating] = StandardScaler().fit_transform(features)

    for rw in np.linspace(0.5, 5, 20):
        X_weighted = X_scaled * np.array([1.0, rw])
        for eps in np.linspace(0.1, 5, 20):
            labels = DBSCAN(eps=eps, min_samples=5).fit(X_weighted).labels_
            for label in np.unique(labels):
                mask = labels == label
                if label == -1 or mask.sum() < 5:
                    continue
                for name in pattern_names:
                    candidate = evaluate_pattern(name, centers_xy[mask], expected_width)
                    if candidate.score > best[name].score:
                        best[name] = candidate

    return {
        name: FiducialPattern(points=[(float(x), float(y)) for x, y in p.points], score=p.score)
        for name, p in best.items()
    }
