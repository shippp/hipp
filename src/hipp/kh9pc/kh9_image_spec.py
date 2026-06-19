from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import re
import rasterio


from hipp.kh9pc.fiducial_patterns import PATTERNS

IMAGE_WIDTHS_PX: list[int] = [114082, 228165, 342247, 456329]
IMAGE_HEIGHT_PX: int = 21771


@dataclass
class KH9ImageSpec:
    expected_size: tuple[int, int]
    collimation_line: bool
    fiducial_type: Literal["disk", "wagon_wheel"]
    top_fiducial_patterns: tuple[PATTERNS, PATTERNS]
    bottom_fiducial_patterns: tuple[PATTERNS, PATTERNS]

    @classmethod
    def from_raster_filepath(cls, filepath: str | Path) -> "KH9ImageSpec":
        mission = KH9ImageSpec.mission_from_filepath(filepath)
        expected_size = KH9ImageSpec.expected_size_from_file(filepath)
        collimation_line = KH9ImageSpec.collimation_from_mission(mission)
        fiducial_type = KH9ImageSpec.fiducial_type_from_mission(mission)
        top_fiducial_patterns = KH9ImageSpec.top_fiducial_patterns_from_mission(mission)
        bottom_fiducial_patterns = KH9ImageSpec.bottom_fiducial_patterns_from_mission(mission)

        return cls(expected_size, collimation_line, fiducial_type, top_fiducial_patterns, bottom_fiducial_patterns)

    @staticmethod
    def mission_from_filepath(filepath: str | Path) -> int:
        pattern = re.compile(r"^(D3C)(\d{4})-(\d)(\d{5})([FA])(\d{3})$")
        stem = Path(filepath).stem
        m = pattern.match(stem)
        if m is None:
            raise ValueError(
                f"Cannot parse KH-9 image ID from {filepath!r}. Expected D3C{{mission}}-{{n}}{{roll}}{{F|A}}{{frame}}."
            )
        mission = int(m.group(2))
        return mission

    @staticmethod
    def collimation_from_mission(mission: int) -> bool:
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")
        collimation_line = mission >= 1206
        return collimation_line

    @staticmethod
    def fiducial_type_from_mission(mission: int) -> Literal["disk", "wagon_wheel"]:
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")
        fiducial_type: Literal["disk", "wagon_wheel"] = "disk" if mission <= 1213 else "wagon_wheel"
        return fiducial_type

    @staticmethod
    def top_fiducial_patterns_from_mission(mission: int) -> tuple[PATTERNS, PATTERNS]:
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")
        top_fiducial_patterns: tuple[PATTERNS, PATTERNS]
        if mission <= 1213:
            top_fiducial_patterns = ("regulare_sparse", "serialized_time_word")
        elif mission <= 1217:
            top_fiducial_patterns = ("segmented_mid", "serialized_time_word")
        else:
            top_fiducial_patterns = ("segmented_mid", "segmented_dense")
        return top_fiducial_patterns

    @staticmethod
    def bottom_fiducial_patterns_from_mission(mission: int) -> tuple[PATTERNS, PATTERNS]:
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")

        bottom_fiducial_patterns: tuple[PATTERNS, PATTERNS]
        if mission <= 1213:
            bottom_fiducial_patterns = ("regulare_sparse", "regular_dense")
        else:
            bottom_fiducial_patterns = ("regulare_mid", "regular_dense")

        return bottom_fiducial_patterns

    @staticmethod
    def expected_size_from_file(filepath: str | Path) -> tuple[int, int]:
        with rasterio.open(filepath) as src:
            width = src.width

        expected_widths_px = sorted(IMAGE_WIDTHS_PX)
        candidates = [w for w in expected_widths_px if w <= width]
        if not candidates:
            raise ValueError(f"Image width {width} is smaller than all known expected widths.")
        return (candidates[-1], IMAGE_HEIGHT_PX)
