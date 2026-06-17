from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from typing import ClassVar


@dataclass
class FiducialPattern(ABC):
    points: NDArray[np.floating]
    expected_width: int

    @property
    def final_score(self) -> float:
        if self.count == 0:
            return 0.0
        return float((self.spacing_score * self.coverage_score))

    @property
    @abstractmethod
    def spacing_score(self) -> float: ...

    @property
    def coverage_score(self) -> float:
        result = float((np.max(self.points[:, 0]) - np.min(self.points[:, 0])) / self.expected_width)
        return min(result, 1.0)

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def spacing(self) -> NDArray[np.floating]:
        sorted_points = self.points[np.argsort(self.points[:, 0])]
        return np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))


class RegularSparse(FiducialPattern):
    SPACING: ClassVar[int] = 19014
    MAX_DELTA: ClassVar[int] = 200

    @property
    def spacing_score(self) -> float:
        lo, hi = RegularSparse.SPACING - RegularSparse.MAX_DELTA, RegularSparse.SPACING + RegularSparse.MAX_DELTA
        if not lo <= np.median(self.spacing) <= hi:
            return 0.0
        return float(coefficient_of_variation_score(self.spacing))


class RegularMid(FiducialPattern):
    SPACING: ClassVar[int] = round(RegularSparse.SPACING / 5)
    MAX_DELTA: ClassVar[int] = 100

    @property
    def spacing_score(self) -> float:
        lo, hi = RegularMid.SPACING - RegularMid.MAX_DELTA, RegularMid.SPACING + RegularMid.MAX_DELTA
        if not lo <= np.median(self.spacing) <= hi:
            return 0.0
        return float(coefficient_of_variation_score(self.spacing))


class RegularDense(FiducialPattern):
    MAX_SPACING: ClassVar[int] = 1480
    MIN_SPACING: ClassVar[int] = MAX_SPACING - 600

    @property
    def spacing_score(self) -> float:
        lo, hi = RegularDense.MIN_SPACING, RegularDense.MAX_SPACING
        if not lo <= np.median(self.spacing) <= hi:
            return 0.0
        return float(coefficient_of_variation_score(self.spacing))


class SegmentedMid(FiducialPattern):
    VALID_GAPS: ClassVar[tuple[int, ...]] = (2, 3)  # 1 or 2 missing markers between segments

    @property
    def spacing_score(self) -> float:
        lo, hi = RegularMid.SPACING - RegularMid.MAX_DELTA, RegularMid.SPACING + RegularMid.MAX_DELTA
        spacing = self.spacing
        if not lo <= np.median(spacing) <= hi:
            return 0.0

        # 1.5× sits between 1× and 2× spacing, so it cleanly separates regular from gap spacings
        split_threshold = np.median(spacing) * 1.5

        regular_spacing = spacing[spacing < split_threshold]
        gap_spacing = spacing[spacing > split_threshold]

        if len(regular_spacing) == 0 or len(gap_spacing) == 0:
            return 0.0

        expected_count = int(sum(round(s / np.median(regular_spacing)) for s in spacing))
        detection_rate = len(spacing) / expected_count

        return float(coefficient_of_variation_score(regular_spacing) * detection_rate)


class SegmentedDense(FiducialPattern):
    GAP: ClassVar[int] = 9

    @property
    def spacing_score(self) -> float:
        lo, hi = RegularDense.MIN_SPACING, RegularDense.MAX_SPACING
        spacing = self.spacing
        if not lo <= np.median(spacing) <= hi:
            return 0.0

        split_threshold = np.median(spacing) * SegmentedDense.GAP / 2

        regular_spacing = spacing[spacing < split_threshold]
        gap_spacing = spacing[spacing > split_threshold]

        if len(regular_spacing) == 0 or len(gap_spacing) == 0:
            return 0.0

        expected_count = int(sum(round(s / np.median(regular_spacing)) for s in spacing))
        detection_rate = len(spacing) / expected_count

        return float(coefficient_of_variation_score(regular_spacing) * detection_rate)


class SerializedTimeWord(FiducialPattern):
    @property
    def spacing_score(self) -> float:
        # TODO
        return 0.0


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


def compute_dst_points(
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
