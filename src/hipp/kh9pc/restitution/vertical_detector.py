"""
Copyright (c) 2026 HIPP developers
Description: VerticalDetector — detects the left and right film frame edges of a KH-9 PC
    scan. For each side, a downsampled 1-row intensity profile is searched for the longest
    constant-minimum plateau (the film background/leader), then the strongest intensity
    gradient just after that plateau's end is taken as the edge. Used as the first step by
    all restitution strategies.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import FittingClass, DetectionError


logger = logging.getLogger(__name__)


@dataclass
class VerticalEdgeResult:
    """Detected edge: global position, local edge index, sub-image, and intensity profile."""

    position: int
    edge_local: int
    gradient_ratio: float
    sub_image: SubImage
    profile: NDArray[np.floating]


@dataclass
class VerticalDetector(FittingClass):
    """Detects the left and right film frame edges from a KH-9 PC raster."""

    vertical_padding: float = 0.25
    search_window_width: int = 10000
    downsample_scale: float = 0.01
    gradient_ratio_threshold: float = 0.3
    max_attempts: int = 5

    def __post_init__(self) -> None:
        """Initialise fitted-attribute slots."""
        super().__init__()
        self._results: dict[str, VerticalEdgeResult] = {}
        self._failed: bool = False

    @property
    def is_failed(self) -> bool:
        """True if the last fit() call failed to detect one or both edges."""
        return self._failed

    @property
    def left_(self) -> VerticalEdgeResult:
        """Detected left edge. Raises if fit() has not been called or failed."""
        if "left" not in self._results:
            raise RuntimeError("left edge not available — call fit() first")
        return self._results["left"]

    @property
    def right_(self) -> VerticalEdgeResult:
        """Detected right edge. Raises if fit() has not been called or failed."""
        if "right" not in self._results:
            raise RuntimeError("right edge not available — call fit() first")
        return self._results["right"]

    @property
    def edges_(self) -> tuple[int, int]:
        """(left_position, right_position) in full-raster pixel coordinates."""
        return self.left_.position, self.right_.position

    @property
    def detected_width_(self) -> int:
        """Width between detected edges in full-raster pixels."""
        return self.right_.position - self.left_.position

    def _fit(self, raster_filepath: Path) -> "VerticalDetector":
        """Detect the left edge, retrying with a shifted search window on failure, then the right edge
        in a single window centered on the left edge plus the expected width."""
        self._failed = False
        self._results = {}
        image_spec = KH9ImageSpec.from_raster_filepath(raster_filepath)
        expected_width = image_spec.expected_size[0]

        with rasterio.open(raster_filepath) as src:
            for attempt in range(self.max_attempts):
                col_off = attempt * self.search_window_width // 2
                try:
                    sub_image = self._sub_image(src, col_off)
                    self._results["left"] = self._detect_edge(sub_image, "left")
                    break
                except DetectionError as e:
                    logger.info(
                        "%s - left attempt %d/%d failed: %s", self.logging_prefix, attempt + 1, self.max_attempts, e
                    )
            else:
                self._failed = True
                logger.warning(
                    "%s - failed to detect left edge after %d attempts", self.logging_prefix, self.max_attempts
                )
                return self

            right_target = self.left_.position + expected_width
            try:
                sub_image = self._sub_image(src, right_target - self.search_window_width // 2)
                self._results["right"] = self._detect_edge(sub_image, "right")
            except DetectionError as e:
                self._failed = True
                logger.warning("%s - failed to detect right edge: %s", self.logging_prefix, e)
                return self

        logger.info(
            "%s - left=%d, right=%d, detected width=%d (expected=%d, diff=%+d px)",
            self.logging_prefix,
            self.left_.position,
            self.right_.position,
            self.detected_width_,
            expected_width,
            self.detected_width_ - expected_width,
        )
        return self

    def _detect_edge(self, sub_image: SubImage, side: str) -> VerticalEdgeResult:
        """Locate the edge within one sub-image, reversing the scan direction for the right side."""
        profile = sub_image.band.flatten()
        signal = profile[::-1] if side == "right" else profile

        _, plateau_end_idx = find_longest_min_segment(signal)
        result = find_first_strong_gradient(signal, plateau_end_idx, self.gradient_ratio_threshold)
        if result is None:
            raise DetectionError("No edge detected")
        edge_idx, gradient_ratio = result

        if side == "right":
            edge_idx = len(signal) - 1 - edge_idx

        position = int(sub_image.to_global(np.array([edge_idx, 0.0]))[0])
        return VerticalEdgeResult(
            position=position,
            edge_local=edge_idx,
            gradient_ratio=gradient_ratio,
            sub_image=sub_image,
            profile=profile,
        )

    def _sub_image(self, src: rasterio.DatasetReader, col_off: int) -> SubImage:
        window = Window(
            max(0, col_off),
            int(self.vertical_padding * src.height),
            self.search_window_width,
            int(src.height * (1 - 2 * self.vertical_padding)),
        )
        return SubImage(src, window=window, out_shape=(1, 1, int(window.width * self.downsample_scale)))


def find_longest_min_segment(signal: NDArray[np.floating]) -> tuple[int, int]:
    """Find the longest contiguous segment where the signal equals its minimum value.

    Returns the (start, end) index pair of that segment.
    """
    signal = np.asarray(signal)

    mask = signal == signal.min()

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    longest = np.argmax(ends - starts)
    return int(starts[longest]), int(ends[longest])


def find_first_strong_gradient(
    signal: NDArray[np.floating], plateau_end_idx: int, ratio_threshold: float = 0.3
) -> tuple[int, float] | None:
    """Scan forward from plateau_end_idx for the peak of the first gradient rise above the threshold.

    plateau_end_idx is expected to be the end of the background plateau, where the signal is
    still flat (zero gradient); the rising edge into image content follows right after it.
    threshold = ratio_threshold * gradient.max(). Once the threshold is crossed, the scan keeps
    climbing to the following index as long as its gradient is higher, so it lands on the peak
    of the rise rather than its first crossing.

    Returns (index, gradient_ratio) of the peak, or None if the threshold is never crossed.
    """
    signal = np.asarray(signal)

    gradient = np.gradient(signal)
    gradient_max = gradient.max()
    threshold = ratio_threshold * gradient_max

    for i in range(plateau_end_idx, len(signal)):
        if gradient[i] > threshold:
            while i + 1 < len(signal) and gradient[i + 1] >= gradient[i] and gradient[i] < gradient_max:
                i += 1
            return i, gradient[i] / gradient_max

    return None
