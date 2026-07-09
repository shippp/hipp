"""
Copyright (c) 2026 HIPP developers
Description: VerticalDetector — detects the left and right film frame edges of a KH-9 PC
    scan by thresholding a downsampled column-sum profile and finding the first intensity
    rupture on each side. Used as the first step by all restitution strategies.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import FittingClass, detect_ruptures


logger = logging.getLogger(__name__)


@dataclass
class VerticalEdgeResult:
    """Detected edge: global position, local rupture index, sub-image, and column-sum profile."""

    position: int
    rupture_local: int
    sub_image: SubImage
    profile: NDArray[np.floating]


@dataclass
class VerticalDetector(FittingClass):
    """Detects the left and right film frame edges from a KH-9 PC raster."""

    background_threshold: int = 20
    vertical_padding: float = 0.25
    search_half_width: int = 5000
    scale: float = 0.1
    left_search_max_attempts: int = 5

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
        """Detect left then right edge and populate results."""
        self._failed = False
        self._results = {}
        image_spec = KH9ImageSpec.from_raster_filepath(raster_filepath)
        expected_width = image_spec.expected_size[0]

        with rasterio.open(raster_filepath) as src:
            left = self._detect_left_edge(src)
            if left is None:
                self._failed = True
                return self
            self._results["left"] = left

            right_center = left.position + expected_width
            right = self._detect_edge(
                src, col_off=right_center - self.search_half_width, reverse_scan=False, side="right"
            )
            if right is None:
                self._failed = True
                return self
            self._results["right"] = right

        logger.info(
            "%s - left=%d, right=%d, detected width=%d (expected=%d, diff=%+d px)",
            self.logging_prefix,
            left.position,
            right.position,
            self.detected_width_,
            expected_width,
            self.detected_width_ - expected_width,
        )
        return self

    def _detect_left_edge(self, src: rasterio.DatasetReader) -> VerticalEdgeResult | None:
        """Detect the left edge, shifting the search window rightward by search_half_width on failure."""
        for attempt in range(self.left_search_max_attempts):
            col_off = attempt * self.search_half_width
            if attempt > 0:
                logger.info(
                    "%s - retrying left edge detection (attempt %d/%d) with search window shifted to col_off=%d",
                    self.logging_prefix,
                    attempt + 1,
                    self.left_search_max_attempts,
                    col_off,
                )
            edge = self._detect_edge(src, col_off=col_off, reverse_scan=True, side="left")
            if edge is not None:
                return edge
        return None

    def _detect_edge(
        self, src: rasterio.DatasetReader, col_off: int, reverse_scan: bool, side: str
    ) -> VerticalEdgeResult | None:
        """Detect a single edge in a window; return None and warn if no rupture found."""
        sub = self._sub_image(src, col_off=col_off)
        _, binary = cv2.threshold(sub.band, self.background_threshold, 1, cv2.THRESH_BINARY)
        profile = np.sum(binary, axis=0)
        ruptures = detect_ruptures(profile, 2, reverse_scan=reverse_scan)
        if ruptures.size == 0:
            logger.warning("%s - no %s edge found", self.logging_prefix, side)
            return None
        r_local = int(ruptures[0])
        position = int(sub.to_global(np.array([r_local, 0.0]))[0])
        return VerticalEdgeResult(position=position, rupture_local=r_local, sub_image=sub, profile=profile)

    def _sub_image(self, src: rasterio.DatasetReader, col_off: int) -> SubImage:
        """Read a downsampled window of width 2*search_half_width starting at col_off."""
        padding_px = int(self.vertical_padding * src.height)
        col_off = max(0, col_off)
        width = min(src.width - col_off, 2 * self.search_half_width)
        window = Window(col_off, padding_px, width, src.height - 2 * padding_px)
        out_shape = (1, int(window.height * self.scale), int(window.width * self.scale))
        return SubImage(src, window, out_shape)
