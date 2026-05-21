from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from rasterio.windows import Window

from hipp.image import remap_tif_blockwise
from hipp.kh9pc.types import FlatResult, RestitutionStrategy, Transformation
from hipp.kh9pc.utils import SubImage, detect_ruptures
from hipp.kh9pc.vertical_detector import VerticalDetector


@dataclass
class FlatStrategy(RestitutionStrategy):
    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10
    output_width: int | None = None
    output_height: int | None = 22064

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, FlatResult] = {}
        self.__transform_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        return False

    @property
    def top_(self) -> FlatResult:
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> FlatResult:
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        if self.__transform_ is None:
            self.__transform_ = self._compute_transformation()
        return self.__transform_

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_off, col_end = self.vertical_detector.edges_
        window_width = col_end - col_off

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, 1)

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                self._results[side] = self._process_side(sub_image, side)

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> FlatResult:
        ruptures = detect_ruptures(sub_image.band.flatten(), self.background_threshold, reverse_scan=(side == "top"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")
        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global(np.array([0.0, rupture_local]))[1])
        return FlatResult(position=position, rupture_local=rupture_local, sub_image=sub_image)

    def _compute_transformation(self) -> Transformation:
        left, right = self.vertical_detector.edges_
        detected_width = right - left
        output_width = self.output_width or detected_width

        top = self.top_.position
        bot = self.bottom_.position
        detected_height = bot - top
        output_height = self.output_height or detected_height

        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2

        crop_offset = (int(left - pad_x), int(top - pad_y))

        return Transformation(
            self.raster_filepath_,
            lambda coords: coords,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )

    def transform(self, output_path: str | Path) -> None:
        tf = self.transformation_
        remap_tif_blockwise(tf.raster_filepath, output_path, tf.inverse_remap, tf.output_size)
