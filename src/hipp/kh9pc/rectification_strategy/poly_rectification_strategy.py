from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import Pipeline

from hipp.kh9pc.rectification_strategy.base import RectificationStrategy
from hipp.kh9pc.utils import SubImage
from hipp.kh9pc.rectification_strategy.vertical_edges_estimator import VerticalEdgesEstimator


@dataclass
class EdgeResult:
    ruptures_local: NDArray[np.integer]
    ruptures_global: NDArray[np.integer]
    distortion_local: NDArray[np.floating]
    distortion_global: NDArray[np.floating]
    model: RANSACRegressor
    sub_image: SubImage

    @property
    def poly(self) -> Pipeline:
        return self.model.estimator_


class PolyRectificationStrategy(RectificationStrategy):
    """Polynomial interpolation strategy: fits a degree-N polynomial to detected edge points.

    Edge points are sampled across the image width, outliers are removed with
    RANSAC, and a polynomial of degree :attr:`polynomial_degree` is fitted to
    the remaining inliers.

    Vertical boundaries are detected by :class:`~hipp.kh9pc.VerticalEdgesEstimator`.
    After calling :meth:`fit`, per-side detection details are stored in
    :attr:`top` and :attr:`bottom`.

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
    polynomial_degree:
        Degree of the polynomial fitted to the edge points.
    ransac_residual_threshold:
        Maximum residual (in pixels) for a point to be considered an inlier.
    ransac_max_trials:
        Maximum number of RANSAC iterations.
    img_height:
        Target height of the rectified image in pixels. If *None*, estimated
        as the mean distance between the fitted top and bottom polynomials.
    grid_shape:
        Number of control points along ``(width, height)``.
    """

    def __init__(
        self,
        vertical_estimator: VerticalEdgesEstimator | None = None,
        background_threshold: int = 20,
        height_fraction: float = 0.15,
        stride: int = 10,
        polynomial_degree: int = 5,
        ransac_residual_threshold: float = 80.0,
        ransac_max_trials: int = 1000,
        img_height: int | None = None,
        grid_shape: tuple[int, int] = (100, 50),
    ):
        self.vertical_estimator = vertical_estimator if vertical_estimator is not None else VerticalEdgesEstimator()
        self.background_threshold = background_threshold
        self.height_fraction = height_fraction
        self.stride = stride
        self.polynomial_degree = polynomial_degree
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials = ransac_max_trials
        self.img_height = img_height
        self.grid_shape = grid_shape
        self.top: EdgeResult | None = None
        self.bottom: EdgeResult | None = None
        self.raster_filepath_: Path | None = None
        self.vertical_edges_: tuple[int, int] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.raster_filepath_ is not None

    def fit(self, raster_filepath: str | Path) -> "PolyRectificationStrategy":
        self.raster_filepath_ = Path(raster_filepath)
        self.vertical_estimator.fit(raster_filepath)
        self.vertical_edges_ = self.vertical_estimator.edges

        with rasterio.open(raster_filepath) as src:
            col_off, col_end = self.vertical_edges_
            window_width = col_end - col_off
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                setattr(self, side, self._process_side(sub_image, side))

        return self

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None
        left, right = self.vertical_edges_
        output_width = right - left

        x_src = np.linspace(left, right, self.grid_shape[0])
        y_top_src = self.top.poly.predict(x_src.reshape(-1, 1)).ravel()
        y_bottom_src = self.bottom.poly.predict(x_src.reshape(-1, 1)).ravel()

        img_height = self.img_height if self.img_height is not None else int(np.abs(np.mean(y_bottom_src - y_top_src)))

        x_dst = np.linspace(0, output_width, self.grid_shape[0])

        src_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        dst_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        for i, (xi_src, xi_dst, yt, yb) in enumerate(zip(x_src, x_dst, y_top_src, y_bottom_src)):
            src_points[i, :, 0] = xi_src
            src_points[i, :, 1] = np.linspace(yt, yb, self.grid_shape[1])
            dst_points[i, :, 0] = xi_dst
            dst_points[i, :, 1] = np.linspace(0, img_height, self.grid_shape[1])

        return src_points, dst_points, (output_width, img_height)

    def __str__(self) -> str:
        params = [
            "Parameters",
            f"  polynomial_degree      : {self.polynomial_degree}",
            f"  ransac_residual_thr    : {self.ransac_residual_threshold}",
            f"  ransac_max_trials      : {self.ransac_max_trials}",
            f"  background_threshold   : {self.background_threshold}",
            f"  height_fraction        : {self.height_fraction}",
            f"  stride                 : {self.stride}",
            f"  grid_shape             : {self.grid_shape}",
        ]

        if not self.is_fitted:
            return "\n".join(["PolyRectificationStrategy (not fitted)", ""] + params)

        assert self.top is not None
        assert self.bottom is not None
        assert self.raster_filepath_ is not None

        n_top = int(self.top.model.inlier_mask_.sum())
        n_top_total = len(self.top.model.inlier_mask_)
        n_bot = int(self.bottom.model.inlier_mask_.sum())
        n_bot_total = len(self.bottom.model.inlier_mask_)

        vertical_str = "\n".join(f"  {line}" for line in str(self.vertical_estimator).splitlines())

        fitted = [
            "PolyRectificationStrategy",
            "",
            f"Image                    : {self.raster_filepath_.name}",
            "",
            "Vertical edges estimator",
            vertical_str,
            "",
            "RANSAC fit",
            f"  top edge               : {n_top} inliers / {n_top_total} points",
            f"  bottom edge            : {n_bot} inliers / {n_bot_total} points",
            "",
        ]

        return "\n".join(fitted + params)

    def get_qc_figures(self) -> list[Figure]:
        return self.vertical_estimator.get_qc_figures() + [self._plot_horizontal_edges(), self._plot_distortions()]

    def _plot_horizontal_edges(self) -> Figure:
        assert self.top is not None and self.bottom is not None

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        for ax, side, result in zip(axes, ["top", "bottom"], [self.top, self.bottom]):
            ax.imshow(result.sub_image.band, cmap="gray", aspect="auto")

            inlier_mask = result.model.inlier_mask_
            pts = result.ruptures_local
            ax.scatter(pts[~inlier_mask, 0], pts[~inlier_mask, 1], s=12, c="red", label="outliers")
            ax.scatter(pts[inlier_mask, 0], pts[inlier_mask, 1], s=12, c="green", label="inliers")

            x_global_range = result.ruptures_global[:, 0].astype(float)
            y_global_pred = result.model.predict(x_global_range.reshape(-1, 1))
            global_pred = np.column_stack([x_global_range, y_global_pred.ravel()])
            local_pred = result.sub_image.to_local(global_pred)
            ax.plot(local_pred[:, 0], local_pred[:, 1], color="blue", linewidth=1, label="model")

            ax.set_title(f"{side} edge")
            ax.legend(loc="best", fontsize=8)
            ax.axis("off")

        return fig

    def _plot_distortions(self) -> Figure:
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

        left, right = self.vertical_edges_
        x_dist = np.linspace(left, right, self.grid_shape[0])
        ax.plot(x_dist, self.top.distortion_global, label="top")
        ax.plot(x_dist, self.bottom.distortion_global, label="bottom")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend()
        ax.set_title("global distortion (top & bottom)")
        ax.set_xlabel("column (px)")
        ax.set_ylabel("distortion (px)")

        return fig

    def _process_side(self, sub_image: SubImage, side: str) -> EdgeResult:
        from hipp.kh9pc import utils

        res = []
        for i in range(sub_image.band.shape[1]):
            ruptures = utils.detect_ruptures(
                sub_image.band[:, i], self.background_threshold, reverse_scan=(side == "top")
            )
            if len(ruptures > 0):
                res.append((i, ruptures[0]))

        if not res:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        ruptures_local = np.array(res)
        ruptures_global = sub_image.to_global(ruptures_local)

        model = utils.fit_ransac_poly(
            ruptures_global[:, 0],
            ruptures_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        x_sample = np.linspace(
            sub_image.window.col_off, sub_image.window.col_off + sub_image.window.width, self.grid_shape[0]
        )
        y_global_pred = model.predict(x_sample.reshape(-1, 1)).ravel()
        distortion_global = y_global_pred - y_global_pred.mean()
        distortion_local = distortion_global / self.stride

        return EdgeResult(
            ruptures_local=ruptures_local,
            ruptures_global=ruptures_global.astype(int),
            distortion_local=distortion_local,
            distortion_global=distortion_global,
            model=model,
            sub_image=sub_image,
        )
