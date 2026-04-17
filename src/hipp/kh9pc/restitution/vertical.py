"""
Copyright (c) 2025 HIPP developers
Description: VerticalDetector — detects left/right film frame edges.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window

from hipp.kh9pc.utils import SubImage, detect_ruptures


@dataclass
class VerticalEdgeResult:
    position: int
    rupture_local: int
    sub_image: SubImage
    profile: NDArray[np.integer]


@dataclass
class VerticalDetector:
    background_threshold: int = 20
    width_fraction: float = 0.15
    stride: int = 10

    def __post_init__(self) -> None:
        self.raster_filepath_: Path | None = None
        self.left_: VerticalEdgeResult | None = None
        self.right_: VerticalEdgeResult | None = None

    def fit(self, raster_filepath: str | Path) -> Self:
        with rasterio.open(raster_filepath) as src:
            window_width = int(src.width * self.width_fraction)
            out_shape = (1, 1, window_width // self.stride)

            for side, window in {
                "left": Window(0, 0, window_width, src.height),
                "right": Window(src.width - window_width, 0, window_width, src.height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                setattr(self, side + "_", self._process_side(sub_image, side))

        self.raster_filepath_ = Path(raster_filepath)
        return self

    def _process_side(self, sub_image: SubImage, side: str) -> VerticalEdgeResult:
        profile = sub_image.band.flatten()
        ruptures = detect_ruptures(profile, self.background_threshold, reverse_scan=(side == "left"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")
        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global(np.array([rupture_local, 0.0]))[0])
        return VerticalEdgeResult(position=position, rupture_local=rupture_local, sub_image=sub_image, profile=profile)

    @property
    def raster_filepath(self) -> Path:
        if self.raster_filepath_ is None:
            raise RuntimeError("need to call fit() first")
        return self.raster_filepath_

    @property
    def left(self) -> VerticalEdgeResult:
        if self.left_ is None:
            raise RuntimeError("left edge not available — call fit() first")
        return self.left_

    @property
    def right(self) -> VerticalEdgeResult:
        if self.right_ is None:
            raise RuntimeError("right edge not available — call fit() first")
        return self.right_

    @property
    def edges(self) -> tuple[int, int]:
        return self.left.position, self.right.position
