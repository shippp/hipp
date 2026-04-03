from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling
import numpy as np

import hipp.kh9pc.utils as utils


@dataclass
class VerticalEdgeResult:
    position: int
    rupture_local: int
    sub_image: utils.SubImage


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
        self.raster_filepath_: Path | None = None

    def fit(self, raster_filepath: str | Path) -> "VerticalEdgesEstimator":
        with rasterio.open(raster_filepath) as src:
            window_width = int(src.width * self.width_fraction)
            out_shape = (1, 1, window_width // self.stride)

            for side, window in {
                "left": Window(0, 0, window_width, src.height),
                "right": Window(src.width - window_width, 0, window_width, src.height),
            }.items():
                sub_image = utils.SubImage(src, window, out_shape)
                setattr(self, side, self._process_side(sub_image, side))

        self.raster_filepath_ = Path(raster_filepath)
        return self

    @property
    def edges(self) -> tuple[int, int]:
        assert self.left is not None
        assert self.right is not None

        return self.left.position, self.right.position

    def _process_side(self, sub_image: utils.SubImage, side: str) -> VerticalEdgeResult:
        ruptures = utils.detect_ruptures(
            sub_image.band.flatten(), self.background_threshold, reverse_scan=(side == "left")
        )
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")
        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global(np.array([rupture_local, 0.0]))[0])
        return VerticalEdgeResult(position=position, rupture_local=rupture_local, sub_image=sub_image)

    @property
    def is_fitted(self) -> bool:
        return self.raster_filepath_ is not None

    def __str__(self) -> str:
        params = [
            "Parameters",
            f"  background_threshold   : {self.background_threshold}",
            f"  width_fraction         : {self.width_fraction}",
            f"  stride                 : {self.stride}",
        ]

        if not self.is_fitted:
            return "\n".join(["VerticalEdgesEstimator (not fitted)", ""] + params)

        assert self.left is not None
        assert self.right is not None
        assert self.raster_filepath_ is not None

        fitted = [
            "VerticalEdgesEstimator",
            "",
            f"Image                    : {self.raster_filepath_.name}",
            "",
            "Detected edges",
            f"  left                   : col={self.left.position} px",
            f"  right                  : col={self.right.position} px",
            "",
        ]

        return "\n".join(fitted + params)

    def get_qc_figures(self) -> list[Figure]:
        return [self.plot_ruptures(), self.plot_edges()]

    def plot_edges(
        self,
        margin_fraction: float = 0.03,
        plot_res: float = 0.05,
    ) -> Figure:
        assert self.raster_filepath_ is not None

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        with rasterio.open(self.raster_filepath_) as src:
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
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        for ax, (side, result) in zip(axes, [("left", self.left), ("right", self.right)]):
            assert result is not None
            profile = result.sub_image.band.flatten()
            ax.plot(profile, color="gray")
            ax.axvline(x=result.rupture_local, color="red", label=f"rupture (local={result.rupture_local})")
            ax.set_title(f"{side} band profile (global col={result.position})")
            ax.set_xlabel("local column index")
            ax.set_ylabel("intensity")
            ax.legend()

        return fig
