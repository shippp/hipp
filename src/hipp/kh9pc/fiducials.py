from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from numpy.typing import NDArray
from typing import ClassVar, Literal, get_args


_PATERN_STR = Literal[
    "regular_sparse", "regular_dense", "regular_mid", "segmented_mid", "segmented_dense", "serialized_time_word"
]

SCAN_PIXEL_SIZE_MM: float = 0.007
"""Film distance in mm represented by one image pixel.

film_distance_mm = pixel_distance * SCAN_PIXEL_SIZE_MM
"""


@dataclass
class KH9ImageSpec:
    collimation_line: bool
    fiducial_type: Literal["disk", "wagon_wheel"]
    top_fiducial_patterns: tuple[_PATERN_STR, _PATERN_STR]
    bottom_fiducial_patterns: tuple[_PATERN_STR, _PATERN_STR]

    @classmethod
    def from_mission(cls, mission: str | int) -> "KH9ImageSpec":
        mission = int(mission)

        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")

        collimation_line = mission >= 1206
        fiducial_type: Literal["disk", "wagon_wheel"] = "disk" if mission <= 1213 else "wagon_wheel"

        # top profiles
        top_fiducial_patterns: tuple[_PATERN_STR, _PATERN_STR]
        if mission <= 1213:
            top_fiducial_patterns = ("regular_sparse", "serialized_time_word")
        elif mission <= 1217:
            top_fiducial_patterns = ("segmented_mid", "serialized_time_word")
        else:
            top_fiducial_patterns = ("segmented_mid", "segmented_dense")

        # bottom profiles
        bottom_fiducial_patterns: tuple[_PATERN_STR, _PATERN_STR]
        if mission <= 1213:
            bottom_fiducial_patterns = ("regular_dense", "regular_sparse")
        else:
            bottom_fiducial_patterns = ("regular_dense", "regular_mid")

        return cls(collimation_line, fiducial_type, top_fiducial_patterns, bottom_fiducial_patterns)

    @classmethod
    def from_filename(cls, name: str | Path) -> "KH9ImageSpec":
        pattern = re.compile(r"^(D3C)(\d{4})-(\d)(\d{5})([FA])(\d{3})$")
        stem = Path(name).stem
        m = pattern.match(stem)
        if m is None:
            raise ValueError(
                f"Cannot parse KH-9 image ID from {name!r}. Expected D3C{{mission}}-{{n}}{{roll}}{{F|A}}{{frame}}."
            )
        mission: str = m.group(2)
        return cls.from_mission(mission)


class FiducialConstants:
    SPARSE_DISTANCE_PX: ClassVar[int] = 19014
    MID_DISTANCE_PX: ClassVar[int] = round(SPARSE_DISTANCE_PX / 5)
    DENSE_MAX_DISTANCE_PX: ClassVar[int] = 1480

    SEGMENTED_MID_GAP: ClassVar[int] = 3
    SEGMENTED_DENSE_GAP: ClassVar[int] = 9

    KEYWORD_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "sparse": (SPARSE_DISTANCE_PX - 200, SPARSE_DISTANCE_PX + 200),
        "mid": (MID_DISTANCE_PX - 100, MID_DISTANCE_PX + 100),
        "dense": (DENSE_MAX_DISTANCE_PX - 600, DENSE_MAX_DISTANCE_PX),
    }

    SEGMENTED_GAP_RATIO_TOLERANCE: ClassVar[float] = 1.5
    """Allowed deviation from the expected gap-to-regular-spacing ratio for segmented patterns."""

    @staticmethod
    def is_valid_spacing(keyword: str, median_dist: float) -> bool:
        lo, hi = FiducialConstants.KEYWORD_RANGES[keyword]
        return lo <= median_dist <= hi

    @staticmethod
    def is_valid_gap_ratio(pattern: _PATERN_STR, regular_median: float, gap_median: float) -> bool:
        """Check that the gap median is within tolerance of gap_factor × regular_median."""
        if regular_median == 0:
            return False
        expected_ratio = FiducialConstants.segmented_gap(pattern)
        return abs(gap_median / regular_median - expected_ratio) <= FiducialConstants.SEGMENTED_GAP_RATIO_TOLERANCE

    @staticmethod
    def segmented_gap(pattern: _PATERN_STR) -> int:
        mapping = {
            "segmented_dense": FiducialConstants.SEGMENTED_DENSE_GAP,
            "segmented_mid": FiducialConstants.SEGMENTED_MID_GAP,
        }

        gap = mapping.get(pattern)
        if gap is None:
            raise ValueError(f"Unrecognized pattern : {pattern}")
        return gap


def centers_xy_from_boxes(boxes: NDArray[np.int_]) -> NDArray[np.floating]:
    """Return (N, 2) array of box centers from (N, 4) ``[x, y, w, h]`` boxes."""
    return boxes[:, :2] + boxes[:, 2:] * 0.5


def compute_fiducial_pattern_score(pattern: _PATERN_STR, points: NDArray[np.floating], image_width: int) -> float:
    if len(points) == 0 or pattern == "serialized_time_word":
        return 0.0

    sorted_points = points[np.argsort(points[:, 0])]
    coverage_score = float((sorted_points[-1, 0] - sorted_points[0, 0]) / image_width)
    distances = np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))
    median_dist = float(np.median(distances))

    for keyword in FiducialConstants.KEYWORD_RANGES:
        if keyword in pattern and not FiducialConstants.is_valid_spacing(keyword, median_dist):
            return 0.0

    if pattern.startswith("regular"):
        return float((coefficient_of_variation_score(distances) * coverage_score) ** (1 / 2))

    if pattern.startswith("segmented"):
        split_threshold = median_dist * FiducialConstants.segmented_gap(pattern) / 2
        regular_dist = distances[distances < split_threshold]
        gap_dist = distances[distances > split_threshold]
        if len(regular_dist) == 0 or len(gap_dist) == 0:
            return 0.0
        if not FiducialConstants.is_valid_gap_ratio(
            pattern, float(np.median(regular_dist)), float(np.median(gap_dist))
        ):
            return 0.0
        return float(
            (coefficient_of_variation_score(regular_dist) * coefficient_of_variation_score(gap_dist) * coverage_score)
            ** (1 / 3)
        )

    return 0.0


def compute_all_fiducial_pattern_scores(points: NDArray[np.floating], image_width: int) -> dict[_PATERN_STR, float]:
    return {p: compute_fiducial_pattern_score(p, points, image_width) for p in get_args(_PATERN_STR)}


def coefficient_of_variation_score(x: NDArray[np.floating]) -> float:
    mean = np.mean(x)

    if mean == 0:
        return 0.0

    coefficient_of_variation = np.std(x) / mean
    return float(1.0 / (1.0 + coefficient_of_variation))
