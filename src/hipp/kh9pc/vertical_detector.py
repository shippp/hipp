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
from hipp.kh9pc.utils import SubImage, compute_gradient_pcts, detect_ruptures


@dataclass
class VerticalDetector(FittingClass):
    background_threshold: int = 20
    width_fraction: float = 0.15
    stride: int = 10
    paddings_pct: tuple[float, float, float, float] = (0.0, 0.10, 0.0, 0.10)
    window_size: int = 30
    min_delta_pct: float = 0.1

    def __post_init__(self) -> None:
        super().__init__()
        self.__left_: VerticalEdgeResult | None = None
        self.__right_: VerticalEdgeResult | None = None

    @property
    def is_failed(self) -> bool:
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
            pad_left = int(src.width * self.paddings_pct[0])
            pad_top = int(src.height * self.paddings_pct[1])
            pad_right = int(src.width * self.paddings_pct[2])
            pad_bottom = int(src.height * self.paddings_pct[3])

            window_width = int(src.width * self.width_fraction)
            row_off = pad_top
            row_height = src.height - pad_top - pad_bottom
            out_shape = (1, 1, window_width // self.stride)

            for side, window in {
                "left": Window(pad_left, row_off, window_width, row_height),
                "right": Window(src.width - window_width - pad_right, row_off, window_width, row_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)

                profile = sub_image.band.flatten()

                # detect from profile all ruptures
                ruptures = detect_ruptures(profile, self.background_threshold, reverse_scan=(side == "left"))
                if len(ruptures) == 0:
                    raise RuntimeError(f"No rupture detected on the {side} edge.")

                gradients_pct = compute_gradient_pcts(profile, ruptures, self.window_size, use_max=(side == "left"))

                # first rupture above min_delta_pct threshold (fallback to first one)
                idx = next((i for i, x in enumerate(gradients_pct) if x > self.min_delta_pct), 0)
                rupture_local = int(ruptures[idx])
                gradient_pct = gradients_pct[idx]

                position = int(sub_image.to_global(np.array([rupture_local, 0.0]))[0])
                result = VerticalEdgeResult(
                    position=position,
                    rupture_local=rupture_local,
                    sub_image=sub_image,
                    profile=profile,
                    gradient_pct=gradient_pct,
                )
                setattr(self, f"_VerticalDetector__{side}_", result)

        return self
