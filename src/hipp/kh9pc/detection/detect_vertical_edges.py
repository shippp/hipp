import logging
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import BaseModel
from rasterio.windows import Window

from hipp.image import SubImage, SubImageSpec
from hipp.kh9pc.detection.base import DetectionError, DetectionResult
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.tools import skip_existing

logger = logging.getLogger(__name__)


class VerticalDetectionOptions(BaseModel):
    """Parameters of the vertical edge detection."""

    vertical_padding: float = 0.25
    search_window_width: int = 10000
    downsample_scale: float = 0.01
    gradient_ratio_threshold: float = 0.3


class VerticalEdge(BaseModel):
    """Detected film edge on one side, with the profile it was found on."""

    position: int
    gradient_ratio: float
    sub_image: SubImageSpec


class VerticalDetectionResult(DetectionResult):
    """Left and right film edges of a raster."""

    raster_filepath: Path
    options: VerticalDetectionOptions
    left: VerticalEdge
    right: VerticalEdge

    @property
    def width(self) -> int:
        """Detected image width in pixels."""
        return self.right.position - self.left.position


def detect_vertical_edges(
    raster_filepath: str | Path,
    output_file: str | Path,
    overwrite: bool = False,
    options: VerticalDetectionOptions | None = None,
) -> None:
    """Detect the left and right film edges and write the result to *output_file* as JSON.

    Skipped if *output_file* already exists and *overwrite* is False.
    """
    raster_filepath = Path(raster_filepath)
    output_file = Path(output_file)
    options = options or VerticalDetectionOptions()

    if skip_existing(output_file, overwrite):
        return
    expected_width = KH9ImageSpec.from_raster_filepath(raster_filepath).expected_size[0]

    with rasterio.open(raster_filepath) as src:
        left = _detect_side(src, 0, src.width - expected_width, "left", options)
        right = _detect_side(
            src,
            left.position + expected_width - options.search_window_width // 2,
            options.search_window_width,
            "right",
            options,
        )

    result = VerticalDetectionResult(raster_filepath=raster_filepath, options=options, left=left, right=right)

    logger.info(
        "%s - vertical edges detected: left=%d, right=%d, width=%d (expected=%d, diff=%+d px)",
        raster_filepath.name,
        left.position,
        right.position,
        result.width,
        expected_width,
        result.width - expected_width,
    )

    result.save(output_file)


def _detect_side(
    src: rasterio.DatasetReader,
    col_off: int,
    width: int,
    side: Literal["right", "left"],
    options: VerticalDetectionOptions,
) -> VerticalEdge:
    """Detect the film edge in a full-width column window, averaged over the padded image height."""
    row_off = int(options.vertical_padding * src.height)
    window = Window(col_off, row_off, width, src.height - 2 * row_off)
    sub_image = SubImage(src, window=window, out_shape=(1, 1, int(width * options.downsample_scale)))
    profile = sub_image.band[0]

    # scan from the outer film border inwards
    scan = profile[::-1] if side == "right" else profile
    _, plateau_end_idx = find_longest_min_segment(scan)
    found = find_first_strong_gradient(scan, plateau_end_idx, options.gradient_ratio_threshold)
    if found is None:
        raise DetectionError(f"No {side} edge detected")

    edge_local, gradient_ratio = found
    if side == "right":
        edge_local = len(scan) - 1 - edge_local

    return VerticalEdge(
        position=int(sub_image.to_global_x(edge_local)),
        gradient_ratio=gradient_ratio,
        sub_image=SubImageSpec.from_sub_image(sub_image),
    )


def find_longest_min_segment(signal: NDArray[np.floating]) -> tuple[int, int]:
    """Find the longest contiguous segment where the signal equals its minimum value.

    Returns the (start, end) index pair of that segment.
    """
    mask = signal == signal.min()

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    longest = np.argmax(ends - starts)
    return int(starts[longest]), int(ends[longest])


def find_first_strong_gradient(
    signal: NDArray[np.floating], plateau_end_idx: int, ratio_threshold: float
) -> tuple[int, float] | None:
    """Scan forward from plateau_end_idx for the peak of the first gradient rise above the threshold.

    plateau_end_idx is expected to be the end of the background plateau, where the signal is
    still flat (zero gradient); the rising edge into image content follows right after it.
    threshold = ratio_threshold * gradient.max(). Once the threshold is crossed, the scan keeps
    climbing to the following index as long as its gradient is higher, so it lands on the peak
    of the rise rather than its first crossing.

    Returns (index, gradient_ratio) of the peak, or None if the threshold is never crossed.
    """
    gradient = np.gradient(signal)
    gradient_max = gradient.max()
    threshold = ratio_threshold * gradient_max

    for i in range(plateau_end_idx, len(signal)):
        if gradient[i] > threshold:
            while i + 1 < len(signal) and gradient[i + 1] >= gradient[i] and gradient[i] < gradient_max:
                i += 1
            return i, gradient[i] / gradient_max

    return None
