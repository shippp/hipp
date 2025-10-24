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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

####################################################################################################################################
#                                                   PUBLIC FUNCTIONS
####################################################################################################################################


def detect_vertical_edges(
    raster_filepath: str | Path,
    px_threshold: int = 20,
    width_fraction: float = 0.05,
    stride: tuple[int, int] = (20, 20),
    ransac_residual_threshold: float = 100,
    ransac_max_trials: int = 100,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> dict[str, int]:
    """
    Detect the left and right vertical edges of a raster image using RANSAC regression.

    This function extracts two vertical bands (left and right) from a raster image,
    identifies strong vertical edge points based on pixel intensity changes, and fits
    a robust RANSAC regression line to estimate the most probable edge position.
    The detected vertical positions (in pixel coordinates) represent the image's
    lateral boundaries, which can be used for geometric calibration or alignment tasks.

    Args:
        raster_filepath (str | Path):
            Path to the raster image file.
        px_threshold (int, optional):
            Minimum pixel intensity difference used to identify edge points. Defaults to 20.
        width_fraction (float, optional):
            Fraction of the image width used to define the left and right edge bands. Defaults to 0.05.
        stride (tuple[int, int], optional):
            Downsampling step (width, height) applied when reading the image to reduce computation. Defaults to (20, 20).
        ransac_residual_threshold (float, optional):
            Maximum distance for a data point to be classified as an inlier by the RANSAC algorithm. Defaults to 100.
        ransac_max_trials (int, optional):
            Maximum number of iterations performed by the RANSAC algorithm. Defaults to 100.
        plot (bool, optional):
            Whether to display the visualization of detected edges and RANSAC fits. Defaults to True.
        output_plot_path (str | Path | None, optional):
            Path to save the resulting plot as an image file. If None, the plot is not saved. Defaults to None.

    Returns:
        dict[str, int]:
            A dictionary mapping "left" and "right" to the detected x-coordinate positions (in pixels)
            of the corresponding vertical edges.

    Notes:
        - This function relies on helper functions `extract_vertical_edge_points()` and `vertical_ransac()`.
        - The RANSAC method ensures robustness against noise and false edge detections.
        - The detected vertical positions can be used to correct lateral distortions in remote sensing imagery.
    """
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    with rasterio.open(raster_filepath) as src:
        window_width = int(src.width * width_fraction)
        window_left = Window(0, 0, window_width, src.height)
        window_right = Window(src.width - window_width, 0, window_width, src.height)

        windows = {"left": window_left, "right": window_right}
        for i, (side, window) in enumerate(windows.items()):
            out_shape = (1, window.height // stride[1], window.width // stride[0])
            band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)
            x_local, y_local = extract_vertical_edge_points(band, px_threshold, side)
            ransac_local, stats = vertical_ransac(x_local, y_local, ransac_residual_threshold, ransac_max_trials)
            res[side] = int(ransac_local.estimator_.constant_ * stride[0] + window.col_off)

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


def detect_collimation_lines(
    raster_filepath: str | Path,
    height_fraction: float = 0.15,
    stride: tuple[int, int] = (256, 10),
    polynomial_degree: int = 2,
    ransac_residual_threshold: float = 50.0,
    ransac_max_trial: int = 100,
    peaks_strategy: str = "distribution",
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> dict[str, RANSACRegressor]:
    """
    Detect the top and bottom collimation lines from a raster image using peak detection and RANSAC polynomial fitting.

    This function extracts two horizontal bands (top and bottom) from a raster image,
    detects prominent peaks corresponding to collimation features, and fits polynomial
    models to them using a RANSAC regression approach. The resulting models represent
    the collimation lines that can be used for geometric correction or alignment analysis.

    Args:
        raster_filepath (str | Path):
            Path to the raster image file.
        height_fraction (float, optional):
            Fraction of the image height to extract for the top and bottom regions. Defaults to 0.15.
        stride (tuple[int, int], optional):
            Step size (width, height) for downsampling the image during processing. Defaults to (256, 10).
        polynomial_degree (int, optional):
            Degree of the polynomial model fitted to the detected peaks. Defaults to 2.
        ransac_residual_threshold (float, optional):
            Maximum residual threshold for the RANSAC algorithm. Defaults to 50.0.
        ransac_max_trial (int, optional):
            Maximum number of iterations for the RANSAC algorithm. Defaults to 100.
        peaks_strategy (str, optional):
            Strategy used for peak detection within the collimation band.
            Possible options depend on `detect_peaks_in_collimation_line`. Defaults to "distribution".
        plot (bool, optional):
            Whether to display the resulting plots of detected peaks and polynomial fits. Defaults to True.
        output_plot_path (str | Path | None, optional):
            Path to save the plot as an image file. If None, the plot is not saved. Defaults to None.

    Returns:
        dict[str, RANSACRegressor]:
            A dictionary containing the fitted RANSAC models for the "top" and "bottom" collimation lines.
    Note:
        - This collimation line should be approximatly on the center of the extracting window.

    See Also:
        detect_peaks_in_collimation_line : Function used internally for peak detection.
    """
    res = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    with rasterio.open(raster_filepath) as src:
        window_height = int(src.height * height_fraction)
        window_top = Window(0, 0, src.width, window_height)
        window_bottom = Window(0, src.height - window_height, src.width, window_height)

        windows = {"top": window_top, "bottom": window_bottom}
        for i, (side, window) in enumerate(windows.items()):
            out_shape = (1, window.height // stride[1], window.width // stride[0])
            band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)

            # scaled by columns the band
            band_scaled = (band - band.mean(axis=0)) / band.std(axis=0)

            # detect peaks by the maximum of prominence
            x_local, y_local = detect_peaks_in_collimation_line(band_scaled, strategy=peaks_strategy)

            # convert local coordinates into global image coordinates
            x_global = x_local * stride[0]
            y_global = y_local * stride[1] + window.row_off

            poly_model = make_pipeline(
                StandardScaler(), PolynomialFeatures(degree=polynomial_degree), LinearRegression()
            )
            ransac = RANSACRegressor(
                poly_model, residual_threshold=ransac_residual_threshold, max_trials=ransac_max_trial, min_samples=3
            )
            ransac.fit(x_global.reshape(-1, 1), y_global)

            res[side] = ransac

            # manage the plot
            axes[i].imshow(band, cmap="gray")

            # add all of the peaks on the plot
            axes[i].scatter(x_local, y_local, s=5, color="blue", label="peaks")

            y_global_fit = ransac.predict(x_global.reshape(-1, 1))
            y_local_fit = (y_global_fit - window.row_off) / stride[1]
            axes[i].plot(x_local, y_local_fit, color="red", label="polynomial fit")

            # add the axes title
            axes[i].set_title(f"{side} edge estimation with polynom")

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


def compute_source_and_target_grid(
    detected_vertical_edges: dict[str, int],
    detected_horizontal_ransac: dict[str, RANSACRegressor],
    colimation_line_dist: int = 21770,
    margin: tuple[int, int] = (0, 147),
    grid_shape: tuple[int, int] = (100, 50),
) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
    """
    Generate source and destination control points for Thin Plate Spline (TPS) rectification
    as structured 2D grids.

    This function creates two corresponding 2D grids of control points:
    - `src_points`: distorted coordinates from the detected vertical edges and horizontal
      RANSAC lines (top and bottom). Shape `(grid_shape[0], grid_shape[1], 2)`.
    - `dst_points`: regular target coordinates forming a rectified rectangular grid.
      Shape `(grid_shape[0], grid_shape[1], 2)`.

    The grids are generated column-wise: each column of points spans from top to bottom
    between the detected/fitted top and bottom edges.

    Parameters
    ----------
    detected_vertical_edges : dict[str, int]
        Dictionary containing pixel positions of left and right vertical edges.
    detected_horizontal_ransac : dict[str, RANSACRegressor]
        Dictionary containing RANSAC models for the top and bottom horizontal edges.
    colimation_line_dist : int, optional
        Target distance (in pixels) between the top and bottom collimation lines in the
        rectified frame. Default is 21770.
    margin : tuple[int, int], optional
        (horizontal, vertical) pixel margins added to all points. Default is (0, 147).
    grid_shape : tuple[int, int], optional
        Number of control points along (width, height) axes of the grid. Default is (100, 50).

    Returns
    -------
    src_points : np.ndarray
        Array of distorted source coordinates with shape `(grid_shape[0], grid_shape[1], 2)`.
        Each entry contains `[x, y]` coordinates in the original image.
    dst_points : np.ndarray
        Array of regular destination coordinates with shape `(grid_shape[0], grid_shape[1], 2)`.
        Each entry contains `[x, y]` coordinates in the rectified frame.
    output_size : tuple[int, int]
        Expected size `(width, height)` of the rectified raster including margins.

    Notes
    -----
    - The source points are computed by evaluating the RANSAC fits for the top and bottom
      horizontal edges and interpolating linearly between them for each column.
    - The destination points form a uniform rectangular grid spanning from (0,0) to
      (cropped_img_width, colimation_line_dist), shifted by the specified margin.
    """
    cropped_img_width = detected_vertical_edges["right"] - detected_vertical_edges["left"]

    # --- Destination points ---
    x_dst = np.linspace(0, cropped_img_width, grid_shape[0])
    y_top_dst = np.zeros_like(x_dst)
    y_bottom_dst = np.full_like(x_dst, colimation_line_dist)

    dst_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)

    for i, (xi, yt, yb) in enumerate(zip(x_dst, y_top_dst, y_bottom_dst)):
        ys = np.linspace(yt, yb, grid_shape[1])
        xs = np.full_like(ys, xi)
        dst_points[i, :, 0] = xs  # x coordinates
        dst_points[i, :, 1] = ys  # y coordinates

    # Apply margin
    dst_points += np.array(margin)

    # --- Source points ---
    x_src = x_dst + detected_vertical_edges["left"]
    y_top_src = detected_horizontal_ransac["top"].predict(x_src.reshape(-1, 1))
    y_bottom_src = detected_horizontal_ransac["bottom"].predict(x_src.reshape(-1, 1))

    src_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)

    for i, (xi, yt, yb) in enumerate(zip(x_src, y_top_src, y_bottom_src)):
        ys = np.linspace(yt, yb, grid_shape[1])
        xs = np.full_like(ys, xi)
        src_points[i, :, 0] = xs  # x coordinates
        src_points[i, :, 1] = ys  # y coordinates

    # --- Output size ---
    output_size = (cropped_img_width + 2 * margin[0], colimation_line_dist + 2 * margin[1])

    return src_points, dst_points, output_size


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def detect_peaks_in_collimation_line(
    image: cv2.typing.MatLike,
    distance: float = 0.2,
    half_window: float = 0.2,
    strategy: str = "distribution",  # "distribution" ou "prominence"
) -> tuple[NDArray[np.uint], NDArray[np.uint]]:
    """
    Detect one peak per image column corresponding to the collimation line.

    Two detection strategies are available:
      - **"distribution"**: selects the peak that maximizes the intensity difference
        between the left and right sides of the column around the candidate peak.
        This approach leverages the fact that the collimation line separates the
        Region of Interest (ROI) from the background. By maximizing the contrast
        between both sides, it avoids detecting bright lines located within the background.
      - **"prominence"**: selects the most prominent peak in each column based on
        local prominence criteria. This method is mainly used after rectification
        and cropping, where the distribution-based approach may fail.
    Args:
        image (cv2.typing.MatLike):
            2D grayscale or intensity image.
        distance (float, optional):
            Minimum vertical distance between peaks, as a fraction of image height.
        half_window (float, optional):
            Fraction of image height used to evaluate intensity distribution
            around each peak (only used with "distribution" strategy).
        strategy (str, optional):
            Peak selection strategy. Either "distribution" or "prominence".

    Returns:
        tuple[np.ndarray, np.ndarray]:
            (peaks_x, peaks_y) coordinates of detected peaks.
    """
    if strategy not in {"distribution", "prominence"}:
        raise ValueError("strategy must be 'distribution' or 'prominence'")

    peaks_x, peaks_y = [], []
    n_rows, n_cols = image.shape[:2]
    half_window_px = int(half_window * n_rows)
    distance_px = int(distance * n_rows)

    for col in range(n_cols):
        signal = image[:, col].astype(int)

        # --- Detect peaks and compute their prominence ---
        peaks, properties = find_peaks(signal, prominence=0, distance=distance_px)
        if len(peaks) == 0:
            continue

        # --- Select up to 5 best candidates ---
        k = min(5, len(peaks))
        top_indices = np.argpartition(properties["prominences"], -k)[-k:]
        selected_peaks = peaks[top_indices]

        # --- Optional border filtering (only for 'distribution') ---
        if strategy == "distribution":
            selected_peaks = selected_peaks[
                (selected_peaks > half_window_px) & (selected_peaks < len(signal) - half_window_px)
            ]
            if len(selected_peaks) == 0:
                continue

        # --- Choose the best peak depending on strategy ---
        if strategy == "distribution":
            scores = []
            for peak in selected_peaks:
                left_part = signal[max(0, peak - half_window_px) : peak]
                right_part = signal[peak : min(len(signal), peak + half_window_px)]
                score = np.abs(np.median(left_part) - np.median(right_part))
                scores.append(score)
            best_peak = selected_peaks[np.argmax(scores)]
        else:
            best_peak = selected_peaks[np.argmax(properties["prominences"][top_indices])]

        peaks_x.append(col)
        peaks_y.append(best_peak)

    return np.array(peaks_x), np.array(peaks_y)


def detect_peaks_in_columns(image: cv2.typing.MatLike, n_peaks: int = 3, distance: float = 0.2) -> NDArray[np.generic]:
    peaks_x, peaks_y = [], []
    n_rows, n_cols = image.shape[:2]
    distance_px = int(distance * n_rows)

    for col in range(n_cols):
        signal = image[:, col].astype(int)

        # --- Detect peaks and compute their prominence ---
        peaks, properties = find_peaks(signal, prominence=0, distance=distance_px)

        k = min(n_peaks, len(peaks))
        top_indices = np.argpartition(properties["prominences"], -k)[-k:]
        selected_peaks = peaks[top_indices]

        for y in selected_peaks:
            peaks_x.append(col)
            peaks_y.append(y)
    return np.column_stack((peaks_x, peaks_y))


def fit_n_ransac_poly(
    peaks: NDArray[np.generic], residual_threshold: float = 20.0, n_ransac: int = 3, degree: int = 2
) -> list[Pipeline]:
    X = peaks[:, 0].reshape(-1, 1)
    y = peaks[:, 1]
    remaining_mask = np.ones_like(y, dtype=bool)
    res = []
    for i in range(n_ransac):
        if np.sum(remaining_mask) < 3:
            break
        # Modèle polynomial + RANSAC
        poly_model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree),
            LinearRegression(),
        )
        ransac = RANSACRegressor(poly_model, residual_threshold=residual_threshold, min_samples=3)
        ransac.fit(X[remaining_mask], y[remaining_mask])
        res.append(ransac.estimator_)

        # remove inliers for next iteration
        inliers_mask = np.zeros_like(y, dtype=bool)
        inliers_mask[remaining_mask] = ransac.inlier_mask_
        remaining_mask[inliers_mask] = False
    return res


def evaluate_poly(image: cv2.typing.MatLike, polys: list[Pipeline], band_thickness: float = 0.2) -> list[float]:
    band_thickness_px = int(band_thickness * image.shape[0])
    x = np.arange(image.shape[1])
    results = []

    rows = np.arange(image.shape[0])[:, None]  # pour vectorisation

    for poly in polys:
        y = poly.predict(x.reshape(-1, 1)).astype(np.int32)
        y_top = np.clip(y + band_thickness_px, 0, image.shape[0] - 1)
        y_bottom = np.clip(y - band_thickness_px, 0, image.shape[0] - 1)

        mask_top = (rows >= y) & (rows <= y_top)
        mask_bottom = (rows >= y_bottom) & (rows <= y)

        top_mean = np.mean(image[mask_top])
        bottom_mean = np.mean(image[mask_bottom])

        results.append(np.abs(top_mean - bottom_mean))
    return results


def extract_vertical_edge_points(
    image: cv2.typing.MatLike, px_threshold: int = 20, direction: str = "left"
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Extract candidate points corresponding to a vertical edge (left or right) in an image.

    For each image row, this function locates the first (or last) pixel
    exceeding a given intensity threshold. These edge points can be used
    later to fit a vertical boundary (e.g., using RANSAC).

    Args:
        image (cv2.typing.MatLike): Grayscale image array.
        px_threshold (int, optional): Pixel intensity threshold used to detect edge pixels.
            Default is 20.
        direction (str, optional): Edge direction to detect.
            Must be either "left" (first pixel above threshold in each row)
            or "right" (last pixel above threshold in each row). Default is "left".

    Returns:
        tuple[NDArray[np.int64], NDArray[np.int64]]:
            - x_coords: 1D array of detected x-coordinates (column indices).
            - y_coords: 1D array of corresponding y-coordinates (row indices).

    Raises:
        ValueError: If `direction` is not "left" or "right".

    Example:
        >>> x, y = extract_vertical_edge_points(image, px_threshold=30, direction="right")
        >>> plt.scatter(x, y, s=2, color='red')
    """
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
    max_trials: int = 1000,
) -> tuple[RANSACRegressor, dict[str, float]]:
    """
    Fit a vertical edge model (constant x-value) using RANSAC regression.

    This function estimates the most probable vertical boundary in an image
    given a set of (x, y) points that approximately follow a vertical line.
    The model fitted is a constant regressor (predicting a fixed x value)
    robust to outliers via the RANSAC algorithm.

    Args:
        x (NDArray[np.generic]): Array of x-coordinates (column indices).
        y (NDArray[np.generic]): Array of y-coordinates (row indices).
        residual_threshold (float, optional): Maximum residual allowed for a point
            to be classified as an inlier. Default is 100.
        max_trials (int, optional): Maximum number of RANSAC iterations. Default is 1000.

    Returns:
        tuple[RANSACRegressor, dict[str, float]]:
            - ransac: Fitted RANSACRegressor model.
            - stats: Dictionary containing:
                * "residuals_rmse": Root Mean Squared Error (RMSE) of inliers.
                * "inlier_percent": Percentage of inlier points.

    Example:
        >>> x_edge, y_edge = extract_vertical_edge_points(image, px_threshold=30)
        >>> model, stats = vertical_ransac(x_edge, y_edge, residual_threshold=50)
        >>> print(stats)
        {'residuals_rmse': 2.1, 'inlier_percent': 93.4}
    """
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
        min_samples=1,
    )
    ransac.fit(Y, x)
    x_pred = ransac.predict(Y)
    stats = {
        "residuals_rmse": root_mean_squared_error(x[ransac.inlier_mask_], x_pred[ransac.inlier_mask_]),
        "inlier_percent": np.mean(ransac.inlier_mask_) * 100,
    }
    return ransac, stats


def _ransac_stats_to_str(stats: dict[str, float]) -> str:
    return "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
