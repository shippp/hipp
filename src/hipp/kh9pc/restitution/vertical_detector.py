"""
Copyright (c) 2025 HIPP developers
Description: VerticalDetector — detects left/right film frame edges.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window

from hipp.kh9pc.restitution.base import detect_ruptures
from hipp.kh9pc.restitution.base import FittingClass
from hipp.image import SubImage


@dataclass
class VerticalEdgeResult:
    position: int
    rupture_local: int
    sub_image: SubImage
    profile: NDArray[np.integer]
    gradient_pct: float


logger = logging.getLogger(__name__)

IMAGE_WIDTHS_MM: list[float] = [798.576, 1597.152, 2395.728, 3194.304]
IMAGE_WIDTHS_PX: list[int] = [114082, 228165, 342247, 456329]


@dataclass
class VerticalDetector(FittingClass):
    background_threshold: int = 20
    width_fraction: float = 0.15
    stride: int = 10
    # tuple order: (left%, top%, right%, bottom%) — fractions of image width/height to ignore at each side
    paddings_pct: tuple[float, float, float, float] = (0.0, 0.10, 0.0, 0.10)
    window_size: int = 30
    min_gradient_pct: float = 0.1

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, VerticalEdgeResult] = {}
        self._failed: bool = False

    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def left_(self) -> VerticalEdgeResult:
        if "left" not in self._results:
            raise RuntimeError("left edge not available — call fit() first")
        return self._results["left"]

    @property
    def right_(self) -> VerticalEdgeResult:
        if "right" not in self._results:
            raise RuntimeError("right edge not available — call fit() first")
        return self._results["right"]

    @property
    def edges_(self) -> tuple[int, int]:
        return self.left_.position, self.right_.position

    @property
    def detected_width_(self) -> int:
        return self.right_.position - self.left_.position

    def _fit(self, raster_filepath: Path) -> "VerticalDetector":
        self._failed = False
        self._results = {}

        with rasterio.open(raster_filepath) as src:
            exp_width = expected_width(src)
            logger.info("VerticalDetector: raster width=%d, effective image width=%d", src.width, exp_width)

            pad_left = int(src.width * self.paddings_pct[0])
            pad_top = int(src.height * self.paddings_pct[1])
            pad_right = int(src.width * self.paddings_pct[2])
            pad_bottom = int(src.height * self.paddings_pct[3])

            row_off = pad_top
            row_height = src.height - pad_top - pad_bottom

            for side in ("left", "right"):
                result = None
                width_frac = self.width_fraction
                while width_frac <= 0.5:
                    window_width = int(src.width * width_frac)
                    out_shape = (1, 1, window_width // self.stride)
                    window = (
                        Window(pad_left, row_off, window_width, row_height)
                        if side == "left"
                        else Window(src.width - window_width - pad_right, row_off, window_width, row_height)
                    )
                    sub_image = SubImage(src, window, out_shape)
                    profile = sub_image.band.flatten()
                    ruptures = detect_ruptures(profile, self.background_threshold, reverse_scan=(side == "left"))
                    if width_frac > self.width_fraction:
                        logger.warning(
                            "VerticalDetector: no rupture found for %s edge at width_fraction=%.2f, retrying with %.2f",
                            side,
                            width_frac - 0.1,
                            width_frac,
                        )
                    if len(ruptures) > 0:
                        gradients_pct = compute_gradient_pcts(
                            profile, ruptures, self.window_size, use_max=(side == "left")
                        )
                        # first rupture above min_gradient_pct threshold (fallback to first one)
                        idx = next((i for i, x in enumerate(gradients_pct) if x > self.min_gradient_pct), 0)
                        rupture_local = int(ruptures[idx])
                        position = int(sub_image.to_global(np.array([rupture_local, 0.0]))[0])
                        result = VerticalEdgeResult(
                            position=position,
                            rupture_local=rupture_local,
                            sub_image=sub_image,
                            profile=profile,
                            gradient_pct=gradients_pct[idx],
                        )
                        break
                    width_frac += 0.1

                if result is None:
                    self._failed = True
                    return self
                self._results[side] = result

            logger.info(
                "VerticalDetector: left=%d, right=%d, detected width=%d (expected=%d, diff=%+d px)",
                self._results["left"].position,
                self._results["right"].position,
                self.detected_width_,
                exp_width,
                self.detected_width_ - exp_width,
            )

        return self


def expected_width(raster: str | Path | rasterio.DatasetReader) -> int:
    if not isinstance(raster, rasterio.DatasetReader):
        with rasterio.open(raster) as src:
            return expected_width(src)

    expected_widths_px = sorted(IMAGE_WIDTHS_PX)
    candidates = [w for w in expected_widths_px if w <= raster.width]
    if not candidates:
        raise ValueError(f"Image width {raster.width} is smaller than all known expected widths.")
    return candidates[-1]


def compute_gradient_pcts(
    profile: NDArray[np.number],
    ruptures: NDArray[np.integer],
    window_size: int,
    use_max: bool,
) -> list[float]:
    """Score each rupture by its local gradient relative to the global gradient extremum.

    For a rising edge (use_max=True)  : score = max(window_gradient) / max(profile_gradient)
    For a falling edge (use_max=False): score = min(window_gradient) / min(profile_gradient)
    """
    gradient = np.diff(profile.astype(np.float32))
    global_stat = float(np.max(gradient)) if use_max else float(np.min(gradient))
    if global_stat == 0:
        return [0.0] * len(ruptures)

    pcts: list[float] = []
    for r in ruptures:
        w = np.diff(profile[max(0, r - window_size) : r + window_size].astype(np.float32))
        local_stat = float(np.max(w)) if use_max else float(np.min(w))
        pcts.append(local_stat / global_stat)
    return pcts
