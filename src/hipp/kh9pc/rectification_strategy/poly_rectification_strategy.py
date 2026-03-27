from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import Pipeline

from hipp.kh9pc.rectification_strategy.base import RectificationStrategy
from hipp.kh9pc.utils import make_summary_figure
from hipp.kh9pc.rectification_strategy.vertical_edges_estimator import VerticalEdgesEstimator


@dataclass
class EdgeResult:
    ruptures_local: NDArray[np.integer]
    ruptures_global: NDArray[np.integer]
    distortion_local: NDArray[np.floating]
    distortion_global: NDArray[np.floating]
    model: RANSACRegressor
    band: NDArray[np.integer]

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

    def fit(self, raster_filepath: str | Path) -> "PolyRectificationStrategy":
        self.raster_filepath_ = Path(raster_filepath)
        self.vertical_estimator.fit(raster_filepath)
        self.vertical_edges_ = self.vertical_estimator.edges

        with rasterio.open(raster_filepath) as src:
            col_off, col_end = self.vertical_edges_
            window_width = col_end - col_off
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])
            scale_x = window_width / self.grid_shape[0]

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)
                setattr(self, side, self._process_side(band, window, scale_x, side))

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

    def generate_qc_report(self, output_path: str | Path) -> None:
        if self.top is None or self.bottom is None or self.vertical_edges_ is None or self.raster_filepath_ is None:
            raise RuntimeError("Call fit() before generate_qc_report()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        left, right = self.vertical_edges_
        n_top = int(self.top.model.inlier_mask_.sum())
        n_top_total = len(self.top.model.inlier_mask_)
        n_bot = int(self.bottom.model.inlier_mask_.sum())
        n_bot_total = len(self.bottom.model.inlier_mask_)

        summary_lines = [
            "PolyRectificationStrategy — QC Report",
            "",
            f"Image                    : {self.raster_filepath_.name}",
            "",
            "Detected edges",
            f"  Vertical               : left={left} px,  right={right} px",
            "",
            "RANSAC fit",
            f"  top edge               : {n_top} inliers / {n_top_total} points",
            f"  bottom edge            : {n_bot} inliers / {n_bot_total} points",
            "",
            "Parameters",
            f"  polynomial_degree      : {self.polynomial_degree}",
            f"  ransac_residual_thr    : {self.ransac_residual_threshold}",
            f"  ransac_max_trials      : {self.ransac_max_trials}",
            f"  background_threshold   : {self.background_threshold}",
            f"  height_fraction        : {self.height_fraction}",
            f"  stride                 : {self.stride}",
            f"  grid_shape             : {self.grid_shape}",
        ]

        with PdfPages(output_path) as pdf:
            summary_fig = make_summary_figure(summary_lines)
            pdf.savefig(summary_fig)
            plt.close(summary_fig)

            vert_edges_fig = self.vertical_estimator.plot_edges(self.raster_filepath_)
            pdf.savefig(vert_edges_fig)
            plt.close(vert_edges_fig)

            vert_rupt_fig = self.vertical_estimator.plot_ruptures()
            pdf.savefig(vert_rupt_fig)
            plt.close(vert_rupt_fig)

            horiz_fig = self._plot_horizontal_edges()
            pdf.savefig(horiz_fig)
            plt.close(horiz_fig)

    def _plot_horizontal_edges(self) -> Figure:
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None

        fig, axes = plt.subplots(3, 1, figsize=(8, 20), constrained_layout=True)
        ax_top, ax_bot, ax_dist = axes

        for ax, side, result in zip([ax_top, ax_bot], ["top", "bottom"], [self.top, self.bottom]):
            ax.imshow(result.band, cmap="gray", aspect="auto")

            x_local = result.ruptures_local[:, 0].astype(float)
            y_local = result.ruptures_local[:, 1].astype(float)
            x_global = result.ruptures_global[:, 0]
            y_global = result.ruptures_global[:, 1]

            scale_x, col_off = np.polyfit(x_local, x_global, 1)
            row_off = float(np.mean(y_global - y_local * self.stride))

            inlier_mask = result.model.inlier_mask_
            ax.scatter(x_local[~inlier_mask], y_local[~inlier_mask], c="red", s=12, label="outliers", zorder=3)
            ax.scatter(x_local[inlier_mask], y_local[inlier_mask], c="lime", s=12, label="inliers", zorder=3)

            x_local_range = np.linspace(0, result.band.shape[1] - 1, 500)
            x_global_range = x_local_range * scale_x + col_off
            y_global_pred = result.model.predict(x_global_range.reshape(-1, 1))
            y_local_pred = (y_global_pred - row_off) / self.stride
            ax.plot(x_local_range, y_local_pred, color="yellow", linewidth=1.5, label="model")

            ax.set_title(f"{side} edge")
            ax.legend(loc="best", fontsize=8)
            ax.axis("off")

        left, right = self.vertical_edges_
        x_dist = np.linspace(left, right, self.grid_shape[0])
        ax_dist.plot(x_dist, self.top.distortion_global, color="steelblue", linewidth=1.5, label="top")
        ax_dist.plot(x_dist, self.bottom.distortion_global, color="tomato", linewidth=1.5, label="bottom")
        ax_dist.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_dist.set_title("global distortion (top & bottom)")
        ax_dist.set_xlabel("column (px)")
        ax_dist.set_ylabel("distortion (px)")
        ax_dist.legend(fontsize=8)

        return fig

    def _process_side(self, band: NDArray[np.integer], window: Window, scale_x: float, side: str) -> EdgeResult:
        from hipp.kh9pc import utils

        res = []
        for i in range(band.shape[1]):
            ruptures = utils.detect_ruptures(band[:, i], self.background_threshold, reverse_scan=(side == "top"))
            if len(ruptures > 0):
                res.append((i, ruptures[0]))

        if not res:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        np_res = np.array(res)
        x_global = np_res[:, 0] * scale_x + window.col_off
        y_global = np_res[:, 1] * self.stride + window.row_off

        model = utils.fit_ransac_poly(
            x_global,
            y_global,
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        x_sample = np.linspace(window.col_off, window.col_off + window.width, self.grid_shape[0])
        y_global_pred = model.predict(x_sample.reshape(-1, 1)).ravel()
        distortion_global = y_global_pred - y_global_pred.mean()
        distortion_local = distortion_global / self.stride

        return EdgeResult(
            ruptures_local=np_res,
            ruptures_global=np.column_stack((x_global, y_global)),
            distortion_local=distortion_local,
            distortion_global=distortion_global,
            model=model,
            band=band,
        )
