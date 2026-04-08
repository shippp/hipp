from dataclasses import dataclass
from pathlib import Path
from typing import Self
import warnings

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
from rasterio.windows import Window
from rasterio.warp import Resampling
import rasterio
from sklearn.linear_model import RANSACRegressor

from hipp.kh9pc.restitution.base import RectificationStrategy
from hipp.kh9pc.utils import SubImage, detect_collimation_peak, fit_ransac_poly


@dataclass
class CollimationResult:
    peaks_local: NDArray[np.integer]
    peaks_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_img: SubImage


@dataclass
class CollimationRectificationStrategy(RectificationStrategy):
    vertical_edges: tuple[int, int]
    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    img_height: int | None = None
    collimation_line_dist: int = 21770
    grid_shape: tuple[int, int] = (100, 50)
    stride: int = 10
    height_fraction: float = 0.15
    max_width_peak: int = 200

    def __post_init__(self) -> None:
        super().__init__()

        self.top_: CollimationResult | None = None
        self.bottom_: CollimationResult | None = None

    def __str__(self) -> str:
        base = super().__str__()
        if not self.is_fitted:
            return base
        return (
            base
            + "\n"
            + "\n".join(
                [
                    "RANSAC fit",
                    f"  top collimation line   : {self.top.inlier_ratio:.1%}",
                    f"  bottom collimation line: {self.bottom.inlier_ratio:.1%}",
                ]
            )
        )

    @property
    def top(self) -> CollimationResult:
        if self.top_ is None:
            raise RuntimeError("top edge not available — call fit() first")
        return self.top_

    @property
    def bottom(self) -> CollimationResult:
        if self.bottom_ is None:
            raise RuntimeError("bottom edge not available — call fit() first")
        return self.bottom_

    def _fit(self, raster_filepath: str | Path) -> Self:
        with rasterio.open(raster_filepath) as src:
            window_width = self.vertical_edges[1] - self.vertical_edges[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(self.vertical_edges[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges[0], src.height - window_height, window_width, window_height),
            }.items():
                sub_img = SubImage(src, window, out_shape, resampling=Resampling.average)
                setattr(self, side + "_", self._process_side(side, sub_img))

        return self

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        left, right = self.vertical_edges
        output_width = right - left
        output_height = self.img_height if self.img_height is not None else self.collimation_line_dist

        # center the collimation lines vertically within the output
        y_offset = (output_height - self.collimation_line_dist) / 2

        x_src = np.linspace(left, right, self.grid_shape[0])
        y_top_src = self.top.model.predict(x_src.reshape(-1, 1)).ravel()
        y_bottom_src = self.bottom.model.predict(x_src.reshape(-1, 1)).ravel()

        x_dst = np.linspace(0, output_width, self.grid_shape[0])

        src_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        dst_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)

        for i, (xi_src, xi_dst, yt, yb) in enumerate(zip(x_src, x_dst, y_top_src, y_bottom_src)):
            src_points[i, :, 0] = xi_src
            src_points[i, :, 1] = np.linspace(yt, yb, self.grid_shape[1])
            dst_points[i, :, 0] = xi_dst
            dst_points[i, :, 1] = np.linspace(y_offset, y_offset + self.collimation_line_dist, self.grid_shape[1])

        return src_points, dst_points, (output_width, output_height)

    def get_qc_figures(self) -> list[Figure]:
        return [self._plot_horizontal_edges(), self._plot_distortions()]

    def _process_side(self, side: str, sub_img: SubImage) -> CollimationResult:
        h, w = sub_img.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)

        for col in range(w):
            vec = sub_img.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        # convert local coords to global
        peaks_global = sub_img.to_global(peaks_local).astype(int)

        model = fit_ransac_poly(
            peaks_global[:, 0],
            peaks_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())
        if inlier_ratio < 0.5:
            warnings.warn(
                f"{side} collimation line: low inlier ratio ({inlier_ratio:.1%}), RANSAC fit may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        y_global_pred = model.predict(peaks_global[:, 0].reshape(-1, 1))
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([peaks_global[:, 0], y_distortion])

        return CollimationResult(
            peaks_local=peaks_local,
            peaks_global=peaks_global,
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_img=sub_img,
        )

    def _plot_horizontal_edges(self) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        for ax, side, result in zip([axes[0], axes[1]], ["top", "bottom"], [self.top, self.bottom]):
            ax.imshow(result.sub_img.band, cmap="gray", aspect="auto")

            # add peaks with color depend on inliers
            peaks = result.peaks_local
            inliers = result.model.inlier_mask_
            ax.scatter(peaks[~inliers, 0], peaks[~inliers, 1], s=12, c="red", label="outliers")
            ax.scatter(peaks[inliers, 0], peaks[inliers, 1], s=12, c="green", label="inliers")

            y_global_pred = result.model.predict(result.peaks_global[:, 0].reshape(-1, 1))
            global_pred = np.column_stack([result.peaks_global[:, 0], y_global_pred])
            local_pred = result.sub_img.to_local(global_pred)
            ax.plot(local_pred[:, 0], local_pred[:, 1], color="blue", linewidth=1, label="model")

            ax.set_title(f"{side} collimation line")
            ax.legend(loc="best", fontsize=8)
            ax.axis("off")

        return fig

    def _plot_distortions(self) -> Figure:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

        for side, result in zip(["top", "bottom"], [self.top, self.bottom]):
            ax.plot(result.distortion[:, 0], result.distortion[:, 1], label=side)
        ax.legend()
        ax.set_title("global distortion (top & bottom)")
        ax.set_xlabel("column (px)")
        ax.set_ylabel("distortion (px)")
        return fig
