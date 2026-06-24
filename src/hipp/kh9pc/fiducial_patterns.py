from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from typing import Literal


PATTERNS = Literal[
    "regulare_sparse", "regulare_mid", "regular_dense", "segmented_mid", "segmented_dense", "serialized_time_word"
]


@dataclass
class DetectedPattern:
    pattern: PATTERNS
    points: NDArray[np.floating]
    expected_width: int
    score: float

    @property
    def count(self) -> int:
        return len(self.points)


def coverage_score(points: NDArray[np.floating], expected_width: int) -> float:
    if len(points) == 0:
        return 0.0
    result = float((np.max(points[:, 0]) - np.min(points[:, 0])) / expected_width)
    return min(result, 1.0)


def compute_spacings(points: NDArray[np.floating]) -> NDArray[np.floating]:
    sorted_points = points[np.argsort(points[:, 0])]
    return np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))


def compute_intra_segment_spacings(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return only intra-segment spacings, filtering out inter-segment gaps."""
    spacings = compute_spacings(points)
    if len(spacings) == 0:
        return spacings
    return spacings[spacings < np.median(spacings) * 1.5]  # type: ignore[no-any-return]


def theorical_spacing_from_pattern(pattern: PATTERNS) -> int:
    SPARSE_SPACING: int = 19014
    MID_SPACING: int = round(SPARSE_SPACING / 5)
    if "sparse" in pattern:
        return SPARSE_SPACING
    elif "mid" in pattern:
        return MID_SPACING
    else:
        raise ValueError(f"No theorical spacing exist for the pattern {pattern}")


def spacing_lo_hi_from_pattern(pattern: PATTERNS) -> tuple[int, int]:
    DENSE_MAX_SPACING: int = 1480
    if pattern == "serialized_time_word":
        raise ValueError(f"No spacing lo hi existe for the pattern : {pattern}")
    if "dense" in pattern:
        return (DENSE_MAX_SPACING - 600, DENSE_MAX_SPACING)
    else:
        max_delta = 200 if "sparse" in pattern else 100
        spacing = theorical_spacing_from_pattern(pattern)
        return (spacing - max_delta, spacing + max_delta)


def spacing_score_from_pattern(pattern: PATTERNS, points: NDArray[np.floating]) -> float:
    if pattern == "serialized_time_word":
        return 0.0

    if len(points) == 0:
        return 0.0

    spacings = compute_spacings(points)
    median_spacing = np.median(spacings)

    # test if the distribution is in bounds of the pattern else return 0
    lo, hi = spacing_lo_hi_from_pattern(pattern)
    if not lo <= median_spacing <= hi:
        return 0.0

    # 1.5× sits between 1× and 2× spacing, so it cleanly separates regular from gap spacings
    regular_spacings = spacings[spacings < median_spacing * 1.5]
    expected_count = int(sum(round(s / median_spacing) for s in spacings))
    detection_rate = len(spacings) / expected_count
    return float(coefficient_of_variation_score(regular_spacings) * detection_rate)


def evaluate_pattern(pattern: PATTERNS, points: NDArray[np.floating], expected_width: int) -> DetectedPattern:
    return DetectedPattern(
        pattern=pattern,
        points=points,
        expected_width=expected_width,
        score=float(spacing_score_from_pattern(pattern, points) * coverage_score(points, expected_width)),
    )


def compute_global_src_and_dst_points(
    top_pattern: DetectedPattern, bottom_pattern: DetectedPattern
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # top and bottom fiducials distances
    # computed with the median take on multiple images
    Y_DIST: int = 23242

    spacing = theorical_spacing_from_pattern(top_pattern.pattern)
    if theorical_spacing_from_pattern(bottom_pattern.pattern) != spacing:
        raise ValueError("Both pattern should have the same distribution (mid or sparse).")

    mid_actual = float((np.median(top_pattern.points[:, 1]) + np.median(bottom_pattern.points[:, 1])) / 2)
    top_y_dst = mid_actual - Y_DIST / 2
    bottom_y_dst = mid_actual + Y_DIST / 2

    top_src, top_dst = compute_src_and_dst_points(top_pattern.points, spacing, top_y_dst)
    bot_src, bot_dst = compute_src_and_dst_points(bottom_pattern.points, spacing, bottom_y_dst)

    return np.vstack((top_src, bot_src)), np.vstack((top_dst, bot_dst))


def compute_expected_fiducial_count(pattern: PATTERNS, expected_width: int) -> int:
    """Return expected number of fiducials across an image of expected_width pixels."""
    spacing = theorical_spacing_from_pattern(pattern)
    return round(expected_width / spacing) + 1


########################################################################################
#                                   UTILS FUNCTIONS
########################################################################################


def centers_xy_from_boxes(boxes: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.floating]:
    """Return (N, 2) array of box centers from (N, 4) ``[x, y, w, h]`` boxes."""
    return boxes[:, :2] + boxes[:, 2:] * 0.5


def coefficient_of_variation_score(x: NDArray[np.floating]) -> float:
    mean = np.mean(x)

    if mean == 0:
        return 0.0

    coefficient_of_variation = np.std(x) / mean
    return float(1.0 / (1.0 + coefficient_of_variation))


def compute_src_and_dst_points(
    points: NDArray[np.floating], true_distance: float, y_dst: float | None = None
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # compute y dst with median if not provideed
    y_dst = y_dst or float(np.median(points[:, 1]))

    # compute sorted spacing
    sorted_points = points[np.argsort(points[:, 0])]
    spacing = np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))

    # compute the median spacing with a filtering to remove gap between segement
    median_spacing = np.median(spacing[spacing < 1.5 * true_distance])

    idx = np.concatenate(([0], np.round(spacing / median_spacing)))
    idx = np.cumulative_sum(idx)
    dst_x = sorted_points[0, 0] + idx * true_distance

    dst_points = np.column_stack([dst_x, np.full_like(dst_x, y_dst)])

    return sorted_points, dst_points
