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
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

####################################################################################################################################
#                                                   PUBLIC FUNCTIONS
####################################################################################################################################


def detect_collimation_lines(
    raster_filepath: str | Path,
    height_fraction: float = 0.15,
    stride: tuple[int, int] = (256, 10),
    polynomial_degree: int = 2,
    ransac_residual_threshold: float = 80.0,
    collimation_line_dist: int = 21770,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> dict[str, Pipeline]:
    """
    Detects and fits collimation lines in the top and bottom portions of a raster image.

    The function reads the input raster, extracts two horizontal windows (top and bottom),
    detects peak positions in each, fits several polynomial RANSAC models, and selects
    the best matching pair of top/bottom lines based on their vertical distance consistency.

    Parameters
    ----------
    raster_filepath : str or Path
        Path to the input raster image.
    height_fraction : float, optional
        Fraction of the raster height to use for top and bottom windows.
    stride : tuple[int, int], optional
        Downsampling stride for (x, y) directions.
    polynomial_degree : int, optional
        Degree of the polynomial model used in RANSAC fitting.
    ransac_residual_threshold : float, optional
        Maximum residual allowed for inlier detection in RANSAC.
    collimation_line_dist : int, optional
        Expected distance between top and bottom collimation lines.
    plot : bool, optional
        If True, display the results interactively.
    output_plot_path : str or Path, optional
        If provided, save the plot to this path.

    Returns
    -------
    dict
        Dictionary containing the best polynomial models for the top and bottom lines.
    """

    # Create figure with two subplots (top and bottom)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    polys_dict, inliers_dict, peaks_dict = {}, {}, {}

    with rasterio.open(raster_filepath) as src:
        # Define top and bottom windows based on height fraction
        window_height = int(src.height * height_fraction)
        window_top = Window(0, 0, src.width, window_height)
        window_bottom = Window(0, src.height - window_height, src.width, window_height)
        windows = {"top": window_top, "bottom": window_bottom}

        # Process both top and bottom sections
        for i, (side, window) in enumerate(windows.items()):
            # Read and downsample raster band in the selected window
            out_shape = (1, window.height // stride[1], window.width // stride[0])
            band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)

            # Detect peaks (local maxima) in each column
            peaks_local = find_column_peaks(band)

            # Convert peak coordinates from local (window) to global raster coordinates
            peaks_global = peaks_local * np.array(stride) + np.array([0, window.row_off])

            # Fit several RANSAC polynomial models to the detected peaks
            polys, inlier_masks = fit_iterative_ransac_polynomials(
                peaks_global, residual_threshold=ransac_residual_threshold, degree=polynomial_degree
            )
            polys_dict[side] = polys
            inliers_dict[side] = inlier_masks
            peaks_dict[side] = peaks_global

            # Display the raster window with proper spatial extent
            extent = [
                window.col_off,
                window.col_off + window.width,
                window.row_off + window.height,
                window.row_off,
            ]
            axes[i].imshow(band, cmap="gray", extent=extent, aspect="auto")
            axes[i].set_title(side.upper())

        # ---- Select the best matching pair of top/bottom polynomials ----
        x = np.linspace(0, src.width, 100)
        best_score, best_pair = np.inf, (0, 0)

        # Compare all combinations of top/bottom models
        for i, poly_top in enumerate(polys_dict["top"]):
            for j, poly_bottom in enumerate(polys_dict["bottom"]):
                y_top = poly_top.predict(x.reshape(-1, 1))
                y_bottom = poly_bottom.predict(x.reshape(-1, 1))

                # Compute deviation from expected collimation distance
                dist = np.abs(collimation_line_dist - np.abs(y_top - y_bottom))
                score = np.mean(dist) + 10 * np.std(dist)

                if score < best_score:
                    best_score = score
                    best_pair = (i, j)

        # ---- Plot selected polynomial models and their inliers/outliers ----
        for idx, side in enumerate(["top", "bottom"]):
            poly = polys_dict[side][best_pair[idx]]
            peaks = peaks_dict[side]
            inliers_mask = inliers_dict[side][best_pair[idx]]

            y_pred = poly.predict(x.reshape(-1, 1))

            # Plot polynomial curve
            axes[idx].plot(x, y_pred, color="red", lw=2, label="Best polynomial")

            # Plot inliers and outliers
            axes[idx].scatter(peaks[inliers_mask, 0], peaks[inliers_mask, 1], s=8, color="lime", label="Inliers")
            axes[idx].scatter(peaks[~inliers_mask, 0], peaks[~inliers_mask, 1], s=8, color="gray", label="Outliers")

            axes[idx].legend(loc="upper right")

        # ---- Plot display and saving ----
        if output_plot_path:
            Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_plot_path)

        if plot:
            plt.show()
        else:
            plt.close()

    # Return the best pair of fitted models
    return {
        "top": polys_dict["top"][best_pair[0]],
        "bottom": polys_dict["bottom"][best_pair[1]],
    }


def compute_source_and_target_grid_v2(
    vertical_edges: tuple[int, int],
    horizontal_polys: tuple[Pipeline, Pipeline],
    img_height: int | None = None,
    grid_shape: tuple[int, int] = (100, 50),
) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
    """Generate source and destination control point grids for TPS rectification.

    Parameters
    ----------
    vertical_edges : tuple[int, int]
        ``(left, right)`` column indices as returned by :func:`detect_vertical_edges`.
    horizontal_polys : tuple[Pipeline, Pipeline]
        ``(top, bottom)`` polynomial pipelines as returned by
        :func:`estimate_horizontal_poly`.
    img_height : int or None, optional
        Target height of the rectified image in pixels. If None, estimated as the
        mean distance between the top and bottom polynomial models. Default is None.
    grid_shape : tuple[int, int], optional
        Number of control points along ``(width, height)``. Default is (100, 50).

    Returns
    -------
    src_points : np.ndarray
        Distorted source coordinates, shape ``(grid_shape[0], grid_shape[1], 2)``.
    dst_points : np.ndarray
        Regular destination coordinates, shape ``(grid_shape[0], grid_shape[1], 2)``.
    output_size : tuple[int, int]
        Expected ``(width, height)`` of the rectified raster.
    """
    cropped_img_width = vertical_edges[1] - vertical_edges[0]

    x_src = np.linspace(vertical_edges[0], vertical_edges[1], grid_shape[0])
    y_top_src = horizontal_polys[0].predict(x_src.reshape(-1, 1)).ravel()
    y_bottom_src = horizontal_polys[1].predict(x_src.reshape(-1, 1)).ravel()

    # compute the approximate img_height with the mean distance between
    # top and bottom poly
    if img_height is None:
        img_height = int(np.abs(np.mean(y_bottom_src - y_top_src)))

    x_dst = np.linspace(0, cropped_img_width, grid_shape[0])
    y_top_dst = np.zeros_like(x_dst)
    y_bottom_dst = np.full_like(x_dst, img_height)

    dst_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)
    for i, (xi, yt, yb) in enumerate(zip(x_dst, y_top_dst, y_bottom_dst)):
        ys = np.linspace(yt, yb, grid_shape[1])
        dst_points[i, :, 0] = np.full_like(ys, xi)
        dst_points[i, :, 1] = ys

    src_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)
    for i, (xi, yt, yb) in enumerate(zip(x_src, y_top_src, y_bottom_src)):
        ys = np.linspace(yt, yb, grid_shape[1])
        src_points[i, :, 0] = np.full_like(ys, xi)
        src_points[i, :, 1] = ys

    output_size = (cropped_img_width, img_height)
    return src_points, dst_points, output_size


def compute_source_and_target_grid(
    detected_vertical_edges: tuple[int, int],
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
    cropped_img_width = detected_vertical_edges[1] - detected_vertical_edges[0]

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
    x_src = x_dst + detected_vertical_edges[0]
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


def find_column_peaks(image: cv2.typing.MatLike, n_peaks: int = 3, distance: float = 0.2) -> NDArray[np.generic]:
    """
    Detects the most prominent peaks along each column of an image.

    This function scans each column of the input image as a 1D signal and identifies
    up to `n_peaks` local maxima based on their prominence. The detected peaks are
    returned as (x, y) coordinates in image space.

    Args:
        image (cv2.typing.MatLike): Input grayscale image or 2D array.
        n_peaks (int, optional): Maximum number of peaks to keep per column.
            Defaults to 3.
        distance (float, optional): Minimum vertical separation between peaks
            (as a fraction of image height). Defaults to 0.2.

    Returns:
        NDArray[np.generic]: Array of shape (N, 2) containing (x, y) coordinates
        of detected peaks across all columns.

    Example:
        >>> peaks = find_column_peaks(image, n_peaks=2, distance=0.1)
        >>> plt.scatter(peaks[:,0], peaks[:,1], s=2, color="red")
    """
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


def fit_iterative_ransac_polynomials(
    peaks: NDArray[np.generic], residual_threshold: float, n_ransac: int = 3, degree: int = 2
) -> tuple[list[Pipeline], list[NDArray[np.bool]]]:
    """
    Fit multiple polynomial models iteratively using RANSAC regression.

    Each iteration fits a polynomial model on the remaining (non-inlier) points,
    removing detected inliers after each successful fit. This allows extraction
    of several dominant polynomial trends from a set of (x, y) peak coordinates.

    Args:
        peaks (NDArray[np.floating]): Array of shape (n_samples, 2) containing (x, y) points.
        residual_threshold (float): Maximum residual for a data point to be classified as an inlier.
        n_ransac (int, optional): Maximum number of RANSAC iterations/models to fit. Default is 3.
        degree (int, optional): Degree of the polynomial features. Default is 2.

    Returns:
        Tuple[List[Pipeline], List[NDArray[np.bool_]]]:
            - models: List of fitted polynomial pipelines (StandardScaler + PolyFeatures + LinearRegression).
            - inlier_masks: List of boolean masks indicating inliers for each fitted model.
    """
    X = peaks[:, 0].reshape(-1, 1)
    y = peaks[:, 1]

    # Initialize tracking masks
    remaining_mask = np.ones_like(y, dtype=bool)
    models: list[Pipeline] = []
    inlier_masks: list[NDArray[np.bool]] = []

    for _ in range(n_ransac):
        # Stop if too few points remain
        if np.sum(remaining_mask) < 3:
            break

        # Create polynomial regression pipeline
        poly_model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree),
            LinearRegression(),
        )

        # Fit RANSAC on remaining points
        ransac = RANSACRegressor(poly_model, residual_threshold=residual_threshold, min_samples=3)
        ransac.fit(X[remaining_mask], y[remaining_mask])

        # Store fitted model
        models.append(ransac.estimator_)

        # Compute inlier mask in global coordinates
        inliers_mask = np.zeros_like(y, dtype=bool)
        inliers_mask[remaining_mask] = ransac.inlier_mask_
        inlier_masks.append(inliers_mask)

        # Exclude inliers for next iteration
        remaining_mask[inliers_mask] = False

    return models, inlier_masks
