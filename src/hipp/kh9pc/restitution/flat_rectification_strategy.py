from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
import rasterio

from hipp.kh9pc.restitution.base import RectificationStrategy

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.windows import Window
from rasterio.warp import Resampling

from hipp.kh9pc.utils import SubImage, detect_ruptures, make_summary_figure


@dataclass
class FlatResult:
    position: int
    rupture_local: int
    sub_image: SubImage


@dataclass
class FlatRectificationStrategy(RectificationStrategy):
    vertical_edges: tuple[int, int]
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10

    def __post_init__(self) -> None:
        super().__init__()

        self.top_: FlatResult | None = None
        self.bottom_: FlatResult | None = None

    def __str__(self) -> str:
        base = super().__str__()
        if not self.is_fitted:
            return base
        return (
            base
            + "\n"
            + "\n".join(
                [
                    "Detected edges",
                    f"  top : row={self.top.position} px",
                    f"  bottom : row={self.bottom.position} px",
                ]
            )
        )

    @property
    def top(self) -> FlatResult:
        if self.top_ is None:
            raise RuntimeError("top edge not available — call fit() first")
        return self.top_

    @property
    def bottom(self) -> FlatResult:
        if self.bottom_ is None:
            raise RuntimeError("bottom edge not available — call fit() first")
        return self.bottom_

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        left, right = self.vertical_edges
        detected_width = right - left
        detected_height = self.bottom.position - self.top.position

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
                [0, detected_height],
                [detected_width, 0],
                [detected_width, detected_height],
            ],
            dtype=float,
        )
        return src_points, dst_points, (detected_width, detected_height)

    def get_qc_figures(self) -> list[Figure]:
        return [make_summary_figure(str(self).splitlines()), self._plot_horizontal_edges(), self._plot_ruptures()]

    def _fit(self, raster_filepath: str | Path) -> Self:
        with rasterio.open(raster_filepath) as src:
            window_width = self.vertical_edges[1] - self.vertical_edges[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, 1)

            for side, window in {
                "top": Window(self.vertical_edges[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges[0], src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                setattr(self, side + "_", self._process_side(sub_image, side))

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> FlatResult:
        ruptures = detect_ruptures(sub_image.band.flatten(), self.background_threshold, reverse_scan=(side == "top"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global(np.array([0.0, rupture_local]))[1])
        return FlatResult(position, rupture_local, sub_image)

    def _plot_horizontal_edges(self) -> Figure:
        left, right = self.vertical_edges
        roi_w = right - left

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        with rasterio.open(self.raster_filepath) as src:
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
