from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from numpy.typing import NDArray
from typing import Literal, get_args


_PATERN_STR = Literal[
    "regular_sparse", "regular_dense", "regular_mid", "segmented_mid", "segmented_dense", "serialized_time_word"
]

SCAN_PIXEL_SIZE_MM: float = 0.007
"""Film distance in mm represented by one image pixel.

film_distance_mm = pixel_distance * SCAN_PIXEL_SIZE_MM
"""

SPARSE_SCAN_MARKS_DISTANCE_MM: float = 133.096
MID_SCAN_MARKS_DISTANCE_MM: float = SPARSE_SCAN_MARKS_DISTANCE_MM / 5
TIME_RECORDING_MAX_DISTANCE_MM: float = 10.36

SEGMENTED_MID_GAP: int = 3
SEGMENTED_DENSE_GAP: int = 9

FIDUCIAL_MAX_SPARSE_DELTA: float = 200
FIDUCIAL_MAX_MID_DELTA: float = 100


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


def compute_fiducial_pattern_score(pattern: _PATERN_STR, points: NDArray[np.floating], image_width: int) -> float:
    if len(points) == 0 or pattern == "serialized_time_word":
        return 0.0

    sorted_points = points[np.argsort(points[:, 0])]
    coverage_score = float((sorted_points[-1, 0] - sorted_points[0, 0]) / image_width)
    distances = np.hypot(np.diff(sorted_points[:, 0]), np.diff(sorted_points[:, 1]))

    if "sparse" in pattern:
        expected_distance_px = SPARSE_SCAN_MARKS_DISTANCE_MM / SCAN_PIXEL_SIZE_MM
        if np.abs(np.median(distances) - expected_distance_px) > FIDUCIAL_MAX_SPARSE_DELTA:
            return 0.0
    if "mid" in pattern:
        expected_distance_px = MID_SCAN_MARKS_DISTANCE_MM / SCAN_PIXEL_SIZE_MM
        if np.abs(np.median(distances) - expected_distance_px) > FIDUCIAL_MAX_MID_DELTA:
            return 0.0
    if "dense" in pattern:
        max_distance_px = TIME_RECORDING_MAX_DISTANCE_MM / SCAN_PIXEL_SIZE_MM
        if np.median(distances) > max_distance_px:
            return 0.0

    if pattern.startswith("regular"):
        return coefficient_of_variation_score(distances) * coverage_score

    if pattern.startswith("segmented"):
        mult = SEGMENTED_MID_GAP if "mid" in pattern else SEGMENTED_DENSE_GAP
        split_threshold = np.median(distances) * mult / 2
        regular_dist = distances[distances < split_threshold]
        gap_dist = distances[distances > split_threshold]
        if len(regular_dist) == 0 or len(gap_dist) == 0:
            return 0.0
        return coefficient_of_variation_score(regular_dist) * coefficient_of_variation_score(gap_dist) * coverage_score

    return 0.0


def compute_all_fiducial_pattern_scores(points: NDArray[np.floating], image_width: int) -> dict[_PATERN_STR, float]:
    return {p: compute_fiducial_pattern_score(p, points, image_width) for p in get_args(_PATERN_STR)}


def coefficient_of_variation_score(x: NDArray[np.floating]) -> float:
    mean = np.mean(x)

    if mean == 0:
        return 0.0

    coefficient_of_variation = np.std(x) / mean
    return float(1.0 / (1.0 + coefficient_of_variation))
