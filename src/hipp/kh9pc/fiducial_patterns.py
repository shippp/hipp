"""
Copyright (c) 2026 HIPP developers
Description: Fiducial pattern definitions, spacing constants, and scoring utilities for
    KH-9 Hexagon panoramic camera images. Provides the building blocks for classifying
    detected fiducial blobs into known pattern types (sparse, mid, dense, segmented,
    time-word) and computing the control-point pairs used for geometric restitution.
"""

from dataclasses import dataclass, replace

import numpy as np

from numpy.typing import NDArray
from typing import Literal


Patterns = Literal[
    "regular_sparse", "regular_mid", "regular_dense", "segmented_mid", "segmented_dense", "serialized_time_word"
]

# Physical spacings derived from KH-9 Hexagon panoramic camera documentation.
# Original values are in inches; converted to pixels with a scan resolution of 0.007 mm/px:
#   px = inches × 25.4 / 0.007
SPARSE_SPACING: int = 19014  # 5.24"
MID_SPACING: int = round(SPARSE_SPACING / 5)
DENSE_MAX_SPACING: int = 1480  # 0.41"
FIDUCIAL_ROW_SPACING: int = 23222  # 6.40"


@dataclass
class DetectedPattern:
    """A scored set of fiducial center points belonging to a single pattern type.

    Attributes
    ----------
    pattern:
        Pattern identifier (one of the ``Patterns`` literals).
    points:
        (N, 2) array of fiducial center coordinates in global raster pixels.
    expected_width:
        Full image width used to compute the coverage score.
    score:
        Combined quality metric in [0, 1]: product of spacing regularity and
        spatial coverage. Higher is better; 0 means the pattern was not detected.
    """

    pattern: Patterns
    points: NDArray[np.floating]
    expected_width: int
    score: float

    @property
    def count(self) -> int:
        """Number of detected fiducial centers in this pattern."""
        return len(self.points)


def coverage_score(points: NDArray[np.floating], expected_width: int) -> float:
    """Fraction of the expected image width covered by the x-span of detected points, clamped to [0, 1]."""
    if len(points) == 0:
        return 0.0
    result = float((np.max(points[:, 0]) - np.min(points[:, 0])) / expected_width)
    return min(result, 1.0)


def compute_spacings(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Euclidean distances between consecutive points sorted by x-coordinate."""
    sorted_points = points[np.argsort(points[:, 0])]
    return np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))


def compute_intra_segment_spacings(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return only intra-segment spacings, filtering out inter-segment gaps."""
    spacings = compute_spacings(points)
    if len(spacings) == 0:
        return spacings
    return spacings[spacings < np.median(spacings) * 1.5]  # type: ignore[no-any-return]


def theorical_spacing_from_pattern(pattern: Patterns) -> int:
    """Return the nominal inter-fiducial spacing in pixels for sparse or mid patterns."""
    if "sparse" in pattern:
        return SPARSE_SPACING
    elif "mid" in pattern:
        return MID_SPACING
    else:
        raise ValueError(f"No theorical spacing exist for the pattern {pattern}")


def spacing_lo_hi_from_pattern(pattern: Patterns) -> tuple[int, int]:
    """Return the (low, high) pixel spacing bounds used to validate whether detected points match a pattern."""
    if pattern == "serialized_time_word":
        raise ValueError(f"No spacing lo hi existe for the pattern : {pattern}")
    if "dense" in pattern:
        return (DENSE_MAX_SPACING - 600, DENSE_MAX_SPACING)
    else:
        max_delta = 200 if "sparse" in pattern else 100
        spacing = theorical_spacing_from_pattern(pattern)
        return (spacing - max_delta, spacing + max_delta)


def spacing_score_from_pattern(pattern: Patterns, points: NDArray[np.floating]) -> float:
    """Score the regularity of detected point spacings against the expected pattern spacing.

    Returns 0 if the median spacing falls outside the pattern's expected range.
    Otherwise returns the product of the coefficient-of-variation score and the
    detection rate (fraction of expected fiducials that were found).
    """
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
    detection_rate = min(len(spacings), expected_count) / max(len(spacings), expected_count)
    return float(coefficient_of_variation_score(regular_spacings) * detection_rate)


def evaluate_pattern(pattern: Patterns, points: NDArray[np.floating], expected_width: int) -> DetectedPattern:
    """Build a ``DetectedPattern`` with a combined score (spacing regularity × coverage)."""
    return DetectedPattern(
        pattern=pattern,
        points=points,
        expected_width=expected_width,
        score=float(spacing_score_from_pattern(pattern, points) * coverage_score(points, expected_width)),
    )


def _actual_y_dist(pattern: DetectedPattern) -> float:
    """Estimate vertical row distance from the actual median fiducial spacing."""
    spacings = compute_intra_segment_spacings(pattern.points)
    if len(spacings) == 0:
        return float(FIDUCIAL_ROW_SPACING)
    return float(np.median(spacings)) * (FIDUCIAL_ROW_SPACING / theorical_spacing_from_pattern(pattern.pattern))


def compute_global_src_and_dst_points(
    top_pattern: DetectedPattern | None, bottom_pattern: DetectedPattern | None
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute source and destination control points from top and bottom fiducial patterns.

    If one pattern is None, it is synthesized from the other by offsetting points by a y_dist
    derived from the actual median fiducial spacing of the detected pattern.
    """
    if top_pattern is None and bottom_pattern is None:
        raise ValueError("At least one pattern is needed to compute points.")

    if top_pattern is None:
        assert bottom_pattern is not None  # juste for mypy
        y_dist = _actual_y_dist(bottom_pattern)
        top_pattern = replace(bottom_pattern, points=bottom_pattern.points - np.array([0, y_dist]))
    elif bottom_pattern is None:
        y_dist = _actual_y_dist(top_pattern)
        bottom_pattern = replace(top_pattern, points=top_pattern.points + np.array([0, y_dist]))

    spacing = theorical_spacing_from_pattern(top_pattern.pattern)
    if theorical_spacing_from_pattern(bottom_pattern.pattern) != spacing:
        raise ValueError("Both pattern should have the same distribution (mid or sparse).")

    mid_actual = float((np.median(top_pattern.points[:, 1]) + np.median(bottom_pattern.points[:, 1])) / 2)
    top_y_dst = mid_actual - FIDUCIAL_ROW_SPACING / 2
    bottom_y_dst = mid_actual + FIDUCIAL_ROW_SPACING / 2

    top_src, top_dst = compute_src_and_dst_points(top_pattern.points, spacing, top_y_dst)
    bot_src, bot_dst = compute_src_and_dst_points(bottom_pattern.points, spacing, bottom_y_dst)

    return np.vstack((top_src, bot_src)), np.vstack((top_dst, bot_dst))


def compute_expected_fiducial_count(pattern: Patterns, expected_width: int) -> int:
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
    """Return 1 / (1 + CV) where CV = std/mean — close to 1 means very regular spacings."""
    mean = np.mean(x)

    if mean == 0:
        return 0.0

    coefficient_of_variation = np.std(x) / mean
    return float(1.0 / (1.0 + coefficient_of_variation))


def compute_src_and_dst_points(
    points: NDArray[np.floating], true_distance: float, y_dst: float | None = None
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Map detected fiducial points to regularised destination points on a uniform grid.

    Points are sorted by x, then each spacing is snapped to the nearest integer
    multiple of the median spacing to handle gaps between segments. The destination
    x positions follow a perfectly uniform grid starting at the first detected point.
    Returns (src_points, dst_points) both as (N, 2) arrays.
    """
    # use median y of detected points when no target row is specified
    y_dst = y_dst if y_dst is not None else float(np.median(points[:, 1]))

    sorted_points = points[np.argsort(points[:, 0])]
    spacing = np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))

    # filter out inter-segment gaps before computing the median (gaps are > 1.5× intra-segment spacing)
    median_spacing = np.median(spacing[spacing < 1.5 * true_distance])

    idx = np.concatenate(([0], np.round(spacing / median_spacing)))
    idx = np.cumsum(idx)
    dst_x = sorted_points[0, 0] + idx * true_distance

    dst_points = np.column_stack([dst_x, np.full_like(dst_x, y_dst)])

    return sorted_points, dst_points
