from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
import rasterio

from hipp.kh9pc.rectification_strategy.base import RectificationStrategy
from hipp.kh9pc.rectification_strategy.vertical_edges_estimator import VerticalEdgesEstimator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.windows import Window
from rasterio.warp import Resampling

from hipp.kh9pc.utils import SubImage, detect_ruptures


@dataclass
class FlatResult:
    position: int
    rupture_local: int
    sub_image: SubImage


class FlatRectificationStrategy(RectificationStrategy):
    """Simplest strategy: a single constant row for each of the top and bottom edges.

    Vertical boundaries are detected by :class:`~hipp.kh9pc.VerticalEdgesEstimator`.
    Horizontal boundaries are single-row ruptures (flat edges).

    Parameters
    ----------
    vertical_estimator:
        Estimator used to locate the left and right vertical boundaries.
        Defaults to a new :class:`VerticalEdgesEstimator` with default parameters.
    background_threshold:
        Intensity threshold below which a pixel is considered background.
    height_fraction:
        Fraction of image height to read on each side when probing.
    stride:
        Downsampling stride applied along the row axis when reading the band.
    img_height:
        Target height of the rectified image in pixels. If *None*, estimated
        as the distance between the detected top and bottom edges.
    """

    def __init__(
        self,
        vertical_estimator: VerticalEdgesEstimator | None = None,
        background_threshold: int = 20,
        height_fraction: float = 0.15,
        stride: int = 10,
        img_height: int | None = None,
    ):
        self.vertical_estimator = vertical_estimator if vertical_estimator is not None else VerticalEdgesEstimator()
        self.background_threshold = background_threshold
        self.height_fraction = height_fraction
        self.stride = stride
        self.img_height = img_height
        self.top: FlatResult | None = None
        self.bottom: FlatResult | None = None
        self.raster_filepath_: Path | None = None
        self.vertical_edges_: tuple[int, int] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.raster_filepath_ is not None

    def _fit(self, raster_filepath: str | Path) -> Self:
        self.raster_filepath_ = Path(raster_filepath)
        self.vertical_estimator.fit(raster_filepath)
        self.vertical_edges_ = self.vertical_estimator.edges

        with rasterio.open(raster_filepath) as src:
            window_width = self.vertical_edges_[1] - self.vertical_edges_[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, 1)

            for side, window in {
                "top": Window(self.vertical_edges_[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges_[0], src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                setattr(self, side, self._process_side(sub_image, side))

        return self

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None
        left, right = self.vertical_edges_
        output_width = right - left
        output_height = self.img_height if self.img_height is not None else self.bottom.position - self.top.position

        # 4-corner grid (2x2) — pure translation/crop, no distortion correction
        src_points = np.array(
            [
                [left, self.top.position],
                [left, self.bottom.position],
                [right, self.top.position],
                [right, self.bottom.position],
            ],
            dtype=float,
        )
        dst_points = np.array(
            [
                [0, 0],
                [0, output_height],
                [output_width, 0],
                [output_width, output_height],
            ],
            dtype=float,
        )
        return src_points, dst_points, (output_width, output_height)

    def __str__(self) -> str:
        params = [
            "Parameters",
            f"  background_threshold   : {self.background_threshold}",
            f"  height_fraction        : {self.height_fraction}",
            f"  stride                 : {self.stride}",
        ]

        if not self.is_fitted:
            return "\n".join(["FlatRectificationStrategy (not fitted)", ""] + params)

        assert self.top is not None
        assert self.bottom is not None
        assert self.raster_filepath_ is not None

        vertical_str = "\n".join(f"  {line}" for line in str(self.vertical_estimator).splitlines())

        fitted = [
            "FlatRectificationStrategy",
            "",
            f"Image                    : {self.raster_filepath_.name}",
            self._fitted_at_str(),
            self._fitting_time_str(),
            "",
            "Vertical edges estimator",
            vertical_str,
            "",
            "Detected edges",
            f"  top                    : row={self.top.position} px",
            f"  bottom                 : row={self.bottom.position} px",
            "",
        ]

        return "\n".join(fitted + params)

    def get_qc_figures(self) -> list[Figure]:
        return self.vertical_estimator.get_qc_figures() + [self._plot_horizontal_edges(), self._plot_ruptures()]

    def _plot_horizontal_edges(self) -> Figure:
        assert self.raster_filepath_ is not None and self.top is not None
        assert self.bottom is not None and self.vertical_edges_ is not None

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        with rasterio.open(self.raster_filepath_) as src:
            left, right = self.vertical_edges_
            roi_w = right - left
            margin = int(0.03 * src.height)

            for ax, side, result in zip(axes, ["top", "bottom"], [self.top, self.bottom]):
                row_off = max(0, result.position - margin)
                row_end = min(src.height, result.position + margin)
                win_h = row_end - row_off
                thumb = src.read(
                    1,
                    window=Window(left, row_off, roi_w, win_h),
                    out_shape=(512, 512),
                    resampling=Resampling.average,
                )
                line_row = (result.position - row_off) / win_h * 512
                ax.imshow(thumb, cmap="gray", aspect="auto")
                ax.axhline(line_row, color="yellow", linewidth=1.5)
                ax.set_title(f"{side} edge — position={result.position} px")
                ax.axis("off")

        return fig

    def _plot_ruptures(self) -> Figure:
        assert self.top is not None and self.bottom is not None

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        for ax, side, result in zip(axes, ["top", "bottom"], [self.top, self.bottom]):
            profile = result.sub_image.band.flatten()
            ax.plot(profile, color="steelblue", linewidth=1)
            ax.axvline(result.rupture_local, color="red", linewidth=1.5, label=f"rupture={result.rupture_local}")
            ax.set_title(f"{side} band profile")
            ax.set_xlabel("row index (downsampled)")
            ax.set_ylabel("intensity")
            ax.legend(fontsize=8)

        return fig

    def _process_side(self, sub_image: SubImage, side: str) -> FlatResult:
        ruptures = detect_ruptures(sub_image.band.flatten(), self.background_threshold, reverse_scan=(side == "top"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global(np.array([0.0, rupture_local]))[1])
        return FlatResult(position, rupture_local, sub_image)
