"""
Copyright (c) 2025 HIPP developers
Description: VerticalDetector — detects left/right film frame edges.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from hipp.kh9pc.types import FittingClass, VerticalEdgeResult
from hipp.kh9pc.utils import SubImage, detect_ruptures


@dataclass
class VerticalDetector(FittingClass):
    background_threshold: int = 20
    width_fraction: float = 0.15
    stride: int = 10

    def __post_init__(self) -> None:
        super().__init__()
        self.__left_: VerticalEdgeResult | None = None
        self.__right_: VerticalEdgeResult | None = None

    @property
    def is_failed(self):
        return False

    @property
    def left_(self) -> VerticalEdgeResult:
        if self.__left_ is None:
            raise RuntimeError("left edge not available — call fit() first")
        return self.__left_

    @property
    def right_(self) -> VerticalEdgeResult:
        if self.__right_ is None:
            raise RuntimeError("right edge not available — call fit() first")
        return self.__right_

    @property
    def edges_(self) -> tuple[int, int]:
        return self.left_.position, self.right_.position

    def _fit(self, raster_filepath: Path) -> "VerticalDetector":
        with rasterio.open(raster_filepath) as src:
            window_width = int(src.width * self.width_fraction)
            out_shape = (1, 1, window_width // self.stride)

            for side, window in {
                "left": Window(0, 0, window_width, src.height),
                "right": Window(src.width - window_width, 0, window_width, src.height),
            }.items():
                sub_image = SubImage(src, window, out_shape)

                profile = sub_image.band.flatten()
                ruptures = detect_ruptures(profile, self.background_threshold, reverse_scan=(side == "left"))
                if len(ruptures) == 0:
                    raise RuntimeError(f"No rupture detected on the {side} edge.")
                rupture_local = int(ruptures[0])
                position = int(sub_image.to_global(np.array([rupture_local, 0.0]))[0])
                result = VerticalEdgeResult(
                    position=position, rupture_local=rupture_local, sub_image=sub_image, profile=profile
                )
                setattr(self, f"_VerticalDetector__{side}_", result)

        return self
