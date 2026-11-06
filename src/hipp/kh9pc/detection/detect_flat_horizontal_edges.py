import logging
from pathlib import Path
from typing import Literal

import rasterio
from pydantic import BaseModel
from rasterio.windows import Window

from hipp.image import SubImage, SubImageSpec
from hipp.kh9pc.detection.base import DetectionError, DetectionResult, detect_ruptures
from hipp.kh9pc.detection.detect_vertical_edges import VerticalDetectionResult
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.tools import skip_existing

logger = logging.getLogger(__name__)


class FlatOptions(BaseModel):
    """Parameters of the flat horizontal detection."""

    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10


class FlatEdge(BaseModel):
    """Detected horizontal film edge on one side, with the sub-image it was found on."""

    position: int
    sub_image: SubImageSpec


class FlatDetectionResult(DetectionResult):
    """Top and bottom flat edges of a raster, detected between columns left and right."""

    raster_filepath: Path
    options: FlatOptions
    left: int
    right: int
    top: FlatEdge
    bottom: FlatEdge

    @property
    def height(self) -> int:
        """Detected image height in pixels."""
        return self.bottom.position - self.top.position


def detect_flat_horizontal_edges(
    raster_filepath: str | Path,
    output_file: str | Path,
    vertical: str | Path | VerticalDetectionResult,
    overwrite: bool = False,
    options: FlatOptions | None = None,
) -> None:
    """Detect the top and bottom film edges as horizontal lines and write the result to *output_file* as JSON.

    The search is limited to the columns between the film edges of *vertical*.
    Skipped if *output_file* already exists and *overwrite* is False.
    """
    raster_filepath = Path(raster_filepath)
    output_file = Path(output_file)

    if skip_existing(output_file, overwrite):
        return

    vertical = VerticalDetectionResult.load(vertical)
    left, right = vertical.left.position, vertical.right.position

    options = options or FlatOptions()

    with rasterio.open(raster_filepath) as src:
        window_height = int(src.height * options.height_fraction)
        top = _detect_side(src, Window(left, 0, right - left, window_height), "top", options)
        bottom_window = Window(left, src.height - window_height, right - left, window_height)
        bottom = _detect_side(src, bottom_window, "bottom", options)

    result = FlatDetectionResult(
        raster_filepath=raster_filepath, options=options, left=left, right=right, top=top, bottom=bottom
    )

    expected_height = KH9ImageSpec.from_raster_filepath(raster_filepath).expected_size[1]
    logger.info(
        "%s - horizontal edges detected: top=%d, bottom=%d, height=%d (expected=%d, diff=%+d px)",
        raster_filepath.name,
        top.position,
        bottom.position,
        result.height,
        expected_height,
        result.height - expected_height,
    )

    result.save(output_file)


def _detect_side(
    src: rasterio.DatasetReader, window: Window, side: Literal["top", "bottom"], options: FlatOptions
) -> FlatEdge:
    """Find the first background rupture in the row profile, averaged over the window width."""
    sub_image = SubImage(src, window, out_shape=(1, int(window.height) // options.stride, 1))

    # scan from the outer film border inwards
    ruptures = detect_ruptures(sub_image.band.flatten(), options.background_threshold, reverse_scan=(side == "top"))
    if len(ruptures) == 0:
        raise DetectionError(f"No {side} edge detected")

    return FlatEdge(
        position=int(sub_image.to_global_y(int(ruptures[0]))),
        sub_image=SubImageSpec.from_sub_image(sub_image),
    )
