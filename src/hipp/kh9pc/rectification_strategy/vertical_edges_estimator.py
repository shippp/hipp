from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling
import numpy as np

import hipp.kh9pc.utils as utils


@dataclass
class VerticalEdgeResult:
    position: int
    rupture_local: int
    band: NDArray[np.integer]


class VerticalEdgesEstimator:
    def __init__(
        self,
        background_threshold: int = 20,
        width_fraction: float = 0.15,
        stride: int = 10,
    ):
        self.background_threshold = background_threshold
        self.width_fraction = width_fraction
        self.stride = stride
        self.left: VerticalEdgeResult | None = None
        self.right: VerticalEdgeResult | None = None

    def fit(self, raster_filepath: str | Path) -> "VerticalEdgesEstimator":
        with rasterio.open(raster_filepath) as src:
            window_width = int(src.width * self.width_fraction)
            out_shape = (1, 1, window_width // self.stride)

            for side, window in {
                "left": Window(0, 0, window_width, src.height),
                "right": Window(src.width - window_width, 0, window_width, src.height),
            }.items():
                band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)
                setattr(self, side, self._process_side(band, window, side))

        return self

    @property
    def edges(self) -> tuple[int, int]:
        assert self.left is not None
        assert self.right is not None

        return self.left.position, self.right.position

    def _process_side(self, band: NDArray[np.integer], window: Window, side: str) -> VerticalEdgeResult:
        ruptures = utils.detect_ruptures(band.flatten(), self.background_threshold, reverse_scan=(side == "left"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")
        rupture_local = int(ruptures[0])
        position = int(rupture_local * self.stride + window.col_off)
        return VerticalEdgeResult(position=position, rupture_local=rupture_local, band=band)

    def plot_edges(
        self,
        raster_filepath: str | Path,
        margin_fraction: float = 0.03,
        plot_res: float = 0.05,
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)

        with rasterio.open(raster_filepath) as src:
            margin = int(src.width * margin_fraction)

            for ax, (side, edge_col) in zip(axes, zip(["left", "right"], self.edges)):
                col_off = max(0, edge_col - margin)
                col_end = min(src.width, edge_col + margin)
                window = Window(col_off, 0, col_end - col_off, src.height)
                out_shape = (1, int(src.height * plot_res), int(window.width * plot_res))

                band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)
                ax.imshow(band, cmap="gray", aspect="auto")
                ax.axvline(x=(edge_col - col_off) * plot_res, color="red")
                ax.set_title(f"{side} edge (col={edge_col})")
                ax.axis("off")

        return fig

    def plot_ruptures(self) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        for ax, (side, result) in zip(axes, [("left", self.left), ("right", self.right)]):
            assert result is not None
            profile = result.band.flatten()
            ax.plot(profile, color="gray")
            ax.axvline(x=result.rupture_local, color="red", label=f"rupture (local={result.rupture_local})")
            ax.set_title(f"{side} band profile (global col={result.position})")
            ax.set_xlabel("local column index")
            ax.set_ylabel("intensity")
            ax.legend()

        return fig
