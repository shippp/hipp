"""
Copyright (c) 2025 HIPP developers
Description: Functions to process lines for KH-9 Panoramic camera images
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy.signal import find_peaks
from skimage.transform import AffineTransform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def detect_vertical_edges(
    raster_filepath: str,
    padding: tuple[int, int] = (0, 700),
    band_width: int = 10000,
    stride: int = 20,
    px_threshold: int = 20,
    ransac_residual_threshold: float = 100,
    ransac_min_samples: int = 1,
    ransac_max_trials: int = 1000,
    plot: bool = True,
    output_plot_path: str | None = None,
) -> dict[str, int]:
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for i, side in enumerate(["left", "right"]):
        band, offset = extract_raster_band_slice(raster_filepath, padding, band_width, stride, side)
        x_local, y_local = detect_vertical_edge(band, px_threshold, side)
        ransac_local, stats = vertical_ransac(
            x_local, y_local, ransac_residual_threshold, ransac_min_samples, ransac_max_trials
        )
        res[side] = int(ransac_local.estimator_.constant_ + offset[0])

        axes[i].imshow(band, cmap="gray")

        inlier_mask = ransac_local.inlier_mask_
        axes[i].scatter(x_local[inlier_mask], y_local[inlier_mask], s=5, color="green", label="inliers")
        axes[i].scatter(x_local[~inlier_mask], y_local[~inlier_mask], s=5, color="red", label="outliers")

        axes[i].axvline(x=ransac_local.estimator_.constant_, color="blue", label="RANSAC line")
        axes[i].set_title(f"{side} edge detection \n({_ransac_stats_to_str(stats)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    if output_plot_path:
        Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)
    if plot:
        plt.show()
    else:
        plt.close()

    return res


def detect_horizontal_collimation_lines(
    raster_filepath: str,
    padding: tuple[int, int] = (0, 700),
    band_height: int = 1500,
    stride: int = 256,
    peaks_distance: float = 200,
    peaks_prominence: float = 50,
    peaks_height: float = 150,
    peaks_width: float = 50,
    ransac_residual_threshold: float = 50,
    ransac_min_samples: int = 3,
    ransac_max_trials: int = 1000,
    plot: bool = True,
    output_plot_path: str | None = None,
) -> dict[str, RANSACRegressor]:
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)

    for i, side in enumerate(["top", "bottom"]):
        band, offset = extract_raster_band_slice(raster_filepath, padding, band_height, stride, side)
        x_local, y_local = detect_peaks_in_columns(band, peaks_distance, peaks_prominence, peaks_height, peaks_width)

        # convert local coordinates into global image coordinates
        x_global = x_local * stride + offset[0]
        y_global = y_local + offset[1]

        # fit a 2 degree polynomial ransac on global coordinates
        ransac, stats = polynomial_ransac(
            x_global, y_global, 2, ransac_residual_threshold, ransac_min_samples, ransac_max_trials
        )

        res[side] = ransac

        # manage the plot
        axes[i].imshow(band, cmap="gray")

        # add all of the peaks on the plot
        inlier_mask = ransac.inlier_mask_
        axes[i].scatter(x_local[inlier_mask], y_local[inlier_mask], s=5, color="green", label="inliers")
        axes[i].scatter(x_local[~inlier_mask], y_local[~inlier_mask], s=5, color="red", label="outliers")

        y_global_fit = ransac.predict(x_global.reshape(-1, 1))
        y_local_fit = y_global_fit - offset[1]
        axes[i].plot(x_local, y_local_fit, color="blue", label="RANSAC poly")

        # add the axes title
        axes[i].set_title(f"{side} edge detection \n({_ransac_stats_to_str(stats)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    if output_plot_path:
        Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)
    if plot:
        plt.show()
    else:
        plt.close()

    return res


def compute_transformation(
    detected_vertical_edges: dict[str, int],
    detected_horizontal_ransac: dict[str, RANSACRegressor],
    colimation_line_dist: int = 21771,
    stride: int = 256,
) -> tuple[cv2.typing.MatLike, tuple[int, int], dict[str, float]]:
    cropped_img_width = detected_vertical_edges["right"] - detected_vertical_edges["left"]
    x_dst = np.arange(0, cropped_img_width, stride)
    y_top_dst = np.repeat(0, len(x_dst))
    y_bottom_dst = np.repeat(colimation_line_dist, len(x_dst))
    dst_top_points = np.column_stack((x_dst, y_top_dst))
    dst_bottom_points = np.column_stack((x_dst, y_bottom_dst))
    dst_points = np.vstack((dst_top_points, dst_bottom_points))

    x_src = x_dst + detected_vertical_edges["left"]
    y_top_src = detected_horizontal_ransac["top"].predict(x_src.reshape(-1, 1))
    y_bottom_src = detected_horizontal_ransac["bottom"].predict(x_src.reshape(-1, 1))
    src_top_points = np.column_stack((x_src, y_top_src))
    src_bottom_points = np.column_stack((x_src, y_bottom_src))
    src_points = np.vstack((src_top_points, src_bottom_points))

    transform = AffineTransform()
    transform.estimate(src_points, dst_points)

    d_before_tf = np.linalg.norm(src_points - dst_points, axis=1)
    d_after_tf = np.linalg.norm(transform(src_points) - dst_points, axis=1)

    dists = {
        "mean_dist_before": np.mean(d_before_tf),
        "max_dist_before": np.max(d_before_tf),
        "mean_dist_after": np.mean(d_after_tf),
        "max_dsit_after": np.max(d_after_tf),
    }

    src = src_points[::10]
    dst = dst_points[::10]
    # scatter des points source (en bleu) et destination (en rouge)
    plt.scatter(src[:, 0], src[:, 1], c="blue", s=10, label="Source points")
    plt.scatter(dst[:, 0], dst[:, 1], c="red", s=10, label="Destination points")

    # tracer un trait entre chaque src et dst
    for (xs, ys), (xd, yd) in zip(src, dst):
        plt.plot([xs, xd], [ys, yd], color="gray", linewidth=0.5, alpha=0.5)

    # Rectangle de l’output (dans l’espace dst)
    rect_x = [0, cropped_img_width, cropped_img_width, 0, 0]
    rect_y = [0, 0, colimation_line_dist, colimation_line_dist, 0]
    plt.plot(rect_x, rect_y, color="green", linewidth=1, label="Output rectangle", linestyle="--")

    plt.gca().invert_yaxis()  # cohérent avec les coordonnées image
    plt.xlabel("x-coordinate [pixels]")
    plt.ylabel("y-coordinate [pixels]")
    plt.title("Source vs Destination points with correspondences")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return transform.params, (cropped_img_width, colimation_line_dist), dists


def plot_distance_between_collimation_lines(
    detected_vertical_edges: dict[str, int], detected_horizontal_ransac: dict[str, RANSACRegressor], stride: int = 256
) -> None:
    x = np.arange(detected_vertical_edges["left"], detected_vertical_edges["right"], stride)
    y_top = detected_horizontal_ransac["top"].predict(x.reshape(-1, 1))
    y_bottom = detected_horizontal_ransac["bottom"].predict(x.reshape(-1, 1))
    distances = np.abs(y_top - y_bottom)
    mean_distances = np.mean(distances)
    plt.plot(x, distances)
    plt.axhline(y=mean_distances, color="red", linestyle="--", label=f"mean : {mean_distances:.2f}")
    plt.legend()
    plt.title("Vertical distance between top and bottom collimation lines")
    plt.xlabel("Image x-coordinate [pixels]")
    plt.ylabel("Distance between lines [pixels]")
    plt.show()


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def extract_raster_band_slice(
    raster_filepath: str, padding: tuple[int, int] = (0, 0), band_size: int = 3000, stride: int = 10, side: str = "left"
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    with rasterio.open(raster_filepath) as src:
        if side == "left":
            row_off, height = padding[1], src.height - 2 * padding[1]
            col_off, width = padding[0], band_size
            axis = 0  # vertical average

        elif side == "right":
            row_off, height = padding[1], src.height - 2 * padding[1]
            col_off, width = src.width - band_size - padding[0], band_size
            axis = 0

        elif side == "top":
            col_off, width = padding[0], src.width - 2 * padding[0]
            row_off, height = padding[1], band_size
            axis = 1  # horizontal average

        elif side == "bottom":
            col_off, width = padding[0], src.width - 2 * padding[0]
            row_off, height = src.height - band_size - padding[1], band_size
            axis = 1

        else:
            raise ValueError("side must be 'left', 'right', 'top' or 'bottom'")

        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

        # Downsample already at read time (only along the axis of interest)
        if axis == 0:  # vertical
            out_shape = (1, height // stride, width)
        else:  # horizontal
            out_shape = (1, height, width // stride)

        subset = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)

    return subset, (col_off, row_off)


def detect_peaks_in_columns(
    median_image: cv2.typing.MatLike,
    distance: float = 200,
    prominence: float = 50,
    height: float = 150,
    width: float = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    peaks_x, peaks_y = [], []
    n_cols = median_image.shape[1]

    for col in range(n_cols):
        signal = median_image[:, col]

        pks, _ = find_peaks(signal, distance=distance, prominence=prominence, height=height, width=width)

        for y in pks:
            peaks_x.append(col)
            peaks_y.append(y)

    return np.array(peaks_x), np.array(peaks_y)


def detect_vertical_edge(
    image: cv2.typing.MatLike, px_threshold: int = 20, direction: str = "left"
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    mask = image > px_threshold

    if direction == "left":
        idx = np.argmax(mask, axis=1)
    elif direction == "right":
        idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    else:
        raise ValueError("direction must be 'left' or 'right'")
    return idx[mask.any(axis=1)], np.arange(len(idx))[mask.any(axis=1)]


def vertical_ransac(
    x: NDArray[np.generic],
    y: NDArray[np.generic],
    residual_threshold: float = 100,
    min_samples: int = 1,
    max_trials: int = 1000,
) -> tuple[RANSACRegressor, dict[str, float]]:
    Y = y.reshape(-1, 1)

    class ConstantRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
        """Regressor that predicts a constant value (mean of y)."""

        def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "ConstantRegressor":
            self.constant_ = np.median(y)  # ou moyenne
            return self

        def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.full(shape=(len(X),), fill_value=self.constant_, dtype=float)

    ransac = RANSACRegressor(
        estimator=ConstantRegressor(),
        max_trials=max_trials,
        residual_threshold=residual_threshold,
        min_samples=min_samples,
    )
    ransac.fit(Y, x)
    x_pred = ransac.predict(Y)
    stats = _compute_residuals_stats(x[ransac.inlier_mask_], x_pred[ransac.inlier_mask_])
    stats["inlier_percent"] = np.mean(ransac.inlier_mask_) * 100
    return ransac, stats


def polynomial_ransac(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int = 2,
    residual_threshold: float = 50.0,
    min_samples: int = 3,
    max_trials: int = 1000,
) -> tuple[RANSACRegressor, dict[str, float]]:
    X = x.reshape(-1, 1)

    base_model = make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    ransac = RANSACRegressor(
        base_model,
        residual_threshold=residual_threshold,
        min_samples=min_samples,
        random_state=0,
        max_trials=max_trials,
    )
    ransac.fit(X, y)
    y_pred = ransac.predict(X)
    stats = _compute_residuals_stats(y[ransac.inlier_mask_], y_pred[ransac.inlier_mask_])
    stats["inlier_percent"] = np.mean(ransac.inlier_mask_) * 100
    return ransac, stats


def _compute_residuals_stats(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> dict[str, float]:
    return {
        "residuals_mae": mean_absolute_error(y_true, y_pred),
        "residuals_rmse": root_mean_squared_error(y_true, y_pred),
        "residuals_r2": r2_score(y_true, y_pred),
    }


def _ransac_stats_to_str(stats: dict[str, float]) -> str:
    return "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
