from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
from rasterio.windows import Window
from rasterio.warp import Resampling
import rasterio

from hipp.kh9pc.rectification_strategy.base import RectificationStrategy
from hipp.kh9pc.rectification_strategy.vertical_edges_estimator import VerticalEdgesEstimator
from hipp.kh9pc.utils import SubImage, detect_collimation_peak, fit_ransac_poly
from sklearn.linear_model import RANSACRegressor


@dataclass
class CollimationResult:
    peaks_local: NDArray[np.integer]
    peaks_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    model: RANSACRegressor

    sub_img: SubImage


class CollimationRectificationStrategy(RectificationStrategy):
    """Collimation-line strategy: uses collimation lines as top and bottom boundaries.

    Instead of detecting intensity ruptures, polynomial models are fitted
    directly to the collimation lines (the physical reference marks printed on
    KH-9 film) to derive the top and bottom boundaries.

    Parameters
    ----------
    polynomial_degree:
        Degree of the polynomial fitted to the collimation line points.
    ransac_residual_threshold:
        Maximum residual (in pixels) for a point to be considered an inlier.
    ransac_max_trials:
        Maximum number of RANSAC iterations.
    img_height:
        Target height of the rectified image in pixels. If *None*, estimated
        from the collimation line separation.
    grid_shape:
        Number of control points along ``(width, height)``.
    """

    def __init__(
        self,
        vertical_estimator: VerticalEdgesEstimator | None = None,
        polynomial_degree: int = 5,
        ransac_residual_threshold: float = 80.0,
        ransac_max_trials: int = 1000,
        img_height: int | None = None,
        grid_shape: tuple[int, int] = (100, 50),
        stride: int = 10,
        height_fraction: float = 0.15,
        max_width_peak: int = 200,
    ):
        self.vertical_estimator = vertical_estimator or VerticalEdgesEstimator()
        self.polynomial_degree = polynomial_degree
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials = ransac_max_trials
        self.img_height = img_height
        self.grid_shape = grid_shape
        self.stride = stride
        self.height_fraction = height_fraction
        self.max_width_peak = max_width_peak
        self.top: CollimationResult | None = None
        self.bottom: CollimationResult | None = None
        self.raster_filepath_: Path | None = None
        self.vertical_edges_: tuple[int, int] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.raster_filepath_ is not None

    def fit(self, raster_filepath: str | Path) -> "CollimationRectificationStrategy":
        # first step: detect vertical edges
        self.vertical_estimator.fit(raster_filepath)
        self.vertical_edges_ = self.vertical_estimator.edges

        with rasterio.open(raster_filepath) as src:
            # define windows and out_shape
            window_width = self.vertical_edges_[1] - self.vertical_edges_[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(self.vertical_edges_[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges_[0], src.height - window_height, window_width, window_height),
            }.items():
                sub_img = SubImage(src, window, out_shape, resampling=Resampling.average)
                setattr(self, side, self._process_side(side, sub_img))

        self.raster_filepath_ = Path(raster_filepath)
        return self

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        raise NotImplementedError

    def __str__(self) -> str:
        params = [
            "Parameters",
            f"  polynomial_degree      : {self.polynomial_degree}",
            f"  ransac_residual_thr    : {self.ransac_residual_threshold}",
            f"  ransac_max_trials      : {self.ransac_max_trials}",
            f"  height_fraction        : {self.height_fraction}",
            f"  max_width_peak         : {self.max_width_peak}",
            f"  stride                 : {self.stride}",
            f"  grid_shape             : {self.grid_shape}",
        ]

        if not self.is_fitted:
            return "\n".join(["CollimationRectificationStrategy (not fitted)", ""] + params)

        assert self.top is not None
        assert self.bottom is not None
        assert self.raster_filepath_ is not None

        n_top = int(self.top.model.inlier_mask_.sum())
        n_top_total = len(self.top.model.inlier_mask_)
        n_bot = int(self.bottom.model.inlier_mask_.sum())
        n_bot_total = len(self.bottom.model.inlier_mask_)

        vertical_str = "\n".join(f"  {line}" for line in str(self.vertical_estimator).splitlines())

        fitted = [
            "CollimationRectificationStrategy",
            "",
            f"Image                    : {self.raster_filepath_.name}",
            "",
            "Vertical edges estimator",
            vertical_str,
            "",
            "RANSAC fit",
            f"  top collimation line   : {n_top} inliers / {n_top_total} points",
            f"  bottom collimation line: {n_bot} inliers / {n_bot_total} points",
            "",
        ]

        return "\n".join(fitted + params)

    def get_qc_figures(self) -> list[Figure]:
        return self.vertical_estimator.get_qc_figures() + [self._plot_horizontal_edges(), self._plot_distortions()]

    def _process_side(self, side: str, sub_img: SubImage) -> CollimationResult:
        h, w = sub_img.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)

        for col in range(w):
            vec = sub_img.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        # convert local coords to global
        peaks_global = sub_img.to_global(peaks_local)

        model = fit_ransac_poly(
            peaks_global[:, 0],
            peaks_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        y_global_pred = model.predict(peaks_global[:, 0].reshape(-1, 1))
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([peaks_global[:, 0], y_distortion])

        return CollimationResult(
            peaks_local=peaks_local,
            peaks_global=sub_img.to_global(peaks_local).astype(int),
            distortion=distortion,
            model=model,
            sub_img=sub_img,
        )

    def _plot_horizontal_edges(self) -> Figure:
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None

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
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

        for side, result in zip(["top", "bottom"], [self.top, self.bottom]):
            ax.plot(result.distortion[:, 0], result.distortion[:, 1], label=side)
        ax.legend()
        ax.set_title("global distortion (top & bottom)")
        ax.set_xlabel("column (px)")
        ax.set_ylabel("distortion (px)")
        return fig
