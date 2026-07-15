"""
Copyright (c) 2026 HIPP developers
Description: FlatStrategy — simplest restitution approach. Assumes the top and bottom
    film edges are perfectly horizontal; detects each as a single rupture point in a
    column-collapsed intensity profile, then crops and centers the image without
    applying any distortion correction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import rasterio
from rasterio.windows import Window

from hipp.image import SubImage, remap_tif_blockwise
from hipp.kh9pc.restitution.base import detect_ruptures
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector


@dataclass
class FlatResult:
    """Raw output of a single-side flat edge detection.

    Attributes
    ----------
    position:
        Detected edge row in full-raster pixel coordinates.
    rupture_local:
        Row index of the rupture within the downsampled sub-image.
    sub_image:
        Downsampled strip used for detection (kept for QC plotting).
    """

    position: int
    rupture_local: int
    sub_image: SubImage


@dataclass
class FlatStrategy(RestitutionStrategy):
    """Restitution strategy that assumes perfectly horizontal top and bottom edges.

    The simplest and most robust strategy — no distortion correction is applied.
    Each edge is located by collapsing the image intensity horizontally (1-pixel
    wide column average) and finding the first intensity rupture. The output is
    then produced by a simple crop-and-center operation.

    Falls back gracefully when collimation lines or fiducial markers are absent
    or undetectable.
    """

    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, FlatResult] = {}
        self.__transformation_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        """Always False — a flat crop is always producible once vertical edges are found."""
        return False

    @property
    def top_(self) -> FlatResult:
        """Detected top edge result. Raises if ``fit()`` has not been called."""
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> FlatResult:
        """Detected bottom edge result. Raises if ``fit()`` has not been called."""
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        """Crop-and-center Transformation (computed lazily on first access)."""
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def _fit(self, raster_filepath: Path) -> Self:
        """Delegate vertical detection, then detect top and bottom horizontal edges."""
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_off, _ = self.vertical_detector.edges_
        window_width = self.vertical_detector.detected_width_

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
        """Find the first intensity rupture in the collapsed horizontal profile for one edge."""
        ruptures = detect_ruptures(sub_image.band.flatten(), self.background_threshold, reverse_scan=(side == "top"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")
        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global_y(rupture_local))
        return FlatResult(position=position, rupture_local=rupture_local, sub_image=sub_image)

    def _compute_transformation(self) -> Transformation:
        """Build a crop-and-center Transformation with identity deformation (no warp)."""
        left, right = self.vertical_detector.edges_
        detected_width = self.vertical_detector.detected_width_
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
        """Write the restituted image using the flat crop-and-center transformation."""
        tf = self.transformation_
        remap_tif_blockwise(tf.raster_filepath, output_path, tf.inverse_remap, tf.output_size)
