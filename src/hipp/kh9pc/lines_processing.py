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
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

####################################################################################################################################
#                                                   PUBLIC FUNCTIONS
####################################################################################################################################


def detect_vertical_edges(
    raster_filepath: str | Path,
    padding: tuple[int, int] = (0, 700),
    band_width: int = 15000,
    stride: int = 20,
    px_threshold: int = 20,
    ransac_residual_threshold: float = 100,
    ransac_max_trials: int = 1000,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> dict[str, int]:
    """
    Detect the left and right vertical edges of a raster image using RANSAC-based line fitting.

    This function extracts narrow vertical bands from both sides of a raster image, detects
    potential vertical edge points, and fits vertical lines to those points using RANSAC.
    The resulting x-coordinates represent the estimated positions of the left and right
    image boundaries (collimation edges).

    Parameters
    ----------
    raster_filepath : str
        Path to the input raster image file.
    padding : tuple[int, int], optional
        Tuple specifying horizontal and vertical padding (in pixels) applied when extracting
        image bands. Default is (0, 700).
    band_width : int, optional
        Width (in pixels) of the vertical band extracted from each image side. Default is 15000.
    stride : int, optional
        Step size (in pixels) used when sampling image columns within each band. Default is 20.
    px_threshold : int, optional
        Intensity threshold (in pixel values) used to identify potential edge points. Default is 20.
    ransac_residual_threshold : float, optional
        Maximum residual error allowed for RANSAC inlier points. Default is 100.
    ransac_max_trials : int, optional
        Maximum number of iterations used during RANSAC fitting. Default is 1000.
    plot : bool, optional
        If True, displays plots showing detected inliers, outliers, and the fitted RANSAC line
        for both image sides. Default is True.
    output_plot_path : str | None, optional
        Optional path to save the generated plot. If None, no file is saved.

    Returns
    -------
    res : dict[str, int]
        Dictionary containing the detected vertical edge positions (x-coordinates) in pixels.
        Keys:
            - "left": x-coordinate of the left vertical edge.
            - "right": x-coordinate of the right vertical edge.

    Notes
    -----
    - This function relies on helper methods:
        * `extract_raster_band_slice()` – extracts image bands for analysis.
        * `extract_vertical_edge_points()` – detects potential edge points.
        * `vertical_ransac()` – performs robust line fitting.
    - The output x-positions are adjusted by the extraction offset to align with global
      image coordinates.
    - Inliers are plotted in green, outliers in red, and the RANSAC fitted line in blue.

    Example
    -------
    >>> edges = detect_vertical_edges(
    ...     raster_filepath="input/image.tif",
    ...     band_width=12000,
    ...     stride=25,
    ...     px_threshold=15,
    ...     output_plot_path="results/vertical_edges.png"
    ... )
    >>> print(edges)
    {'left': 153, 'right': 20482}
    """
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for i, side in enumerate(["left", "right"]):
        band, offset = extract_raster_band_slice(raster_filepath, padding, band_width, stride, side)
        x_local, y_local = extract_vertical_edge_points(band, px_threshold, side)
        ransac_local, stats = vertical_ransac(x_local, y_local, ransac_residual_threshold, ransac_max_trials)
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
    raster_filepath: str | Path,
    padding_dict: dict[str, tuple[int, int]] = {"top": (0, 700), "bottom": (0, 700)},
    band_height: int = 1700,
    stride: int = 256,
    polynomial_degree: int = 2,
    ransac_residual_threshold: float = 50.0,
    ransac_max_trial: int = 100,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> dict[str, RANSACRegressor]:
    """
    Detect the top and bottom horizontal collimation lines from a raster image using RANSAC polynomial fitting.

    This function extracts horizontal bands from the top and bottom regions of a raster image,
    detects prominent peaks within each band, and fits a polynomial model to estimate the
    horizontal collimation lines. RANSAC regression is used to improve robustness against noise
    and outliers. The resulting models represent the geometric position of each collimation line.

    Parameters
    ----------
    raster_filepath : str
        Path to the input raster image file.
    padding_dict : dict[str, tuple[int, int]], optional
        Dictionary specifying the vertical padding applied when extracting the top and bottom
        bands. Each key ("top" or "bottom") maps to a tuple (x_padding, y_padding).
        Default is {"top": (0, 700), "bottom": (0, 700)}.
    band_height : int, optional
        Height (in pixels) of the horizontal band extracted from the raster image.
        Default is 1700.
    stride : int, optional
        Step size in pixels used to sample vertical columns within the image.
        Default is 256.
    polynomial_degree : int, optional
        Degree of the polynomial used for modeling the collimation line.
        Default is 2.
    ransac_residual_threshold : float, optional
        Maximum residual allowed for inliers in the RANSAC algorithm.
        Default is 50.0.
    ransac_max_trial : int, optional
        Maximum number of iterations for RANSAC fitting.
        Default is 100.
    plot : bool, optional
        If True, displays the diagnostic plots showing detected peaks and polynomial fits.
        If False, no figure is shown. Default is True.
    output_plot_path : str | None, optional
        Optional path where the generated plot will be saved. If None, no file is saved.

    Returns
    -------
    res : dict[str, RANSACRegressor]
        Dictionary containing fitted RANSAC models for each detected collimation line.
        Keys:
            - "top": RANSAC model for the top line.
            - "bottom": RANSAC model for the bottom line.

    Notes
    -----
    - The function assumes that the raster file can be read by `extract_raster_band_slice()`.
    - Peak detection is performed column-wise using `detect_peaks_in_columns()`.
    - The fitted models can be used later to compute distances or transformations between lines.
    - Plots include detected peaks (in blue) and fitted polynomial curves (in red).

    Example
    -------
    >>> ransac_models = detect_horizontal_collimation_lines(
    ...     raster_filepath="input/image.tif",
    ...     band_height=1800,
    ...     stride=256,
    ...     polynomial_degree=3,
    ...     output_plot_path="results/collimation_detection.png"
    ... )
    >>> top_line_model = ransac_models["top"]
    >>> bottom_line_model = ransac_models["bottom"]
    """
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)

    for i, side in enumerate(["top", "bottom"]):
        # extract the raster band profile
        band, offset = extract_raster_band_slice(raster_filepath, padding_dict[side], band_height, stride, side)

        # scaled by columns the band
        band_scaled = (band - band.mean(axis=0)) / band.std(axis=0)

        # detect peaks by the maximum of prominence
        x_local, y_local = detect_peaks_in_columns(band_scaled)

        # convert local coordinates into global image coordinates
        x_global = x_local * stride + offset[0]
        y_global = y_local + offset[1]

        poly_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=polynomial_degree), LinearRegression())
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
        y_local_fit = y_global_fit - offset[1]
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


def compute_transformation(
    detected_vertical_edges: dict[str, int],
    detected_horizontal_ransac: dict[str, RANSACRegressor],
    colimation_line_dist: int = 21771,
    margin: tuple[int, int] = (0, 147),
    stride: int = 256,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> tuple[cv2.typing.MatLike, tuple[int, int], dict[str, float]]:
    """
    Compute the affine transformation that maps detected collimation lines to a target geometry.

    This function estimates an affine transformation aligning two detected horizontal
    collimation lines (top and bottom) to a predefined reference layout. The transformation
    ensures that the collimation lines are parallel and separated by a fixed physical
    distance in pixels. Optionally, it visualizes the correspondence between source and
    destination points.

    Parameters
    ----------
    detected_vertical_edges : dict[str, int]
        Dictionary containing the x-coordinates of the detected vertical boundaries.
        Expected keys: "left" and "right".
    detected_horizontal_ransac : dict[str, RANSACRegressor]
        Dictionary containing the RANSAC regression models for the top and bottom
        collimation lines. Expected keys: "top" and "bottom".
    colimation_line_dist : int, optional
        Expected vertical distance (in pixels) between the top and bottom collimation lines
        in the target geometry. Default is 21771.
    margin : tuple[int, int], optional
        Margin (in pixels) added to the destination coordinates as (x_margin, y_margin).
        Default is (0, 147).
    stride : int, optional
        Step size in pixels used to sample points along the x-axis. Default is 256.
    plot : bool, optional
        If True, displays the scatter plot showing source and destination points.
        If False, closes the figure after processing. Default is True.
    output_plot_path : str | None, optional
        Optional file path where the generated plot will be saved. If None, no file is saved.

    Returns
    -------
    transform.params : cv2.typing.MatLike
        3×3 affine transformation matrix mapping source points to destination points.
    output_size : tuple[int, int]
        Output image size (width, height) after transformation, including margins.
    dists : dict[str, float]
        Dictionary containing error metrics before and after transformation:
            - "mean_dist_before": Mean distance between source and destination before transformation.
            - "max_dist_before": Maximum distance before transformation.
            - "mean_dist_after": Mean distance after applying transformation.
            - "max_dsit_after": Maximum distance after applying transformation.

    Notes
    -----
    - The affine transformation is estimated using the `skimage.transform.AffineTransform` class.
    - The output height is adjusted to ensure an even multiple of 2 (by subtracting 1 pixel).
    - This function inverts the y-axis to match standard image coordinate conventions.

    Example
    -------
    >>> tf_matrix, output_size, errors = compute_transformation(
    ...     detected_vertical_edges={"left": 0, "right": 2048},
    ...     detected_horizontal_ransac={"top": top_ransac, "bottom": bottom_ransac},
    ...     colimation_line_dist=21771,
    ...     margin=(50, 150),
    ...     stride=128,
    ...     output_plot_path="results/transformation.png"
    ... )
    >>> print(errors["mean_dist_after"])
    2.35
    """
    cropped_img_width = detected_vertical_edges["right"] - detected_vertical_edges["left"]
    x_dst = np.arange(0, cropped_img_width, stride)
    y_top_dst = np.repeat(0, len(x_dst))
    y_bottom_dst = np.repeat(colimation_line_dist, len(x_dst))
    dst_top_points = np.column_stack((x_dst, y_top_dst))
    dst_bottom_points = np.column_stack((x_dst, y_bottom_dst))
    dst_points = np.vstack((dst_top_points, dst_bottom_points))

    # Translate all dst_points with margin
    dst_points = dst_points + np.array(margin)

    x_src = x_dst + detected_vertical_edges["left"]
    y_top_src = detected_horizontal_ransac["top"].predict(x_src.reshape(-1, 1))
    y_bottom_src = detected_horizontal_ransac["bottom"].predict(x_src.reshape(-1, 1))
    src_top_points = np.column_stack((x_src, y_top_src))
    src_bottom_points = np.column_stack((x_src, y_bottom_src))
    src_points = np.vstack((src_top_points, src_bottom_points))

    # compute the output size
    # the trick here is to minus 1 to the collimation dist to have a mutiple of 2
    output_size = (cropped_img_width + 2 * margin[0], colimation_line_dist - 1 + 2 * margin[1])

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
    rect_x = [0, output_size[0], output_size[0], 0, 0]
    rect_y = [0, 0, output_size[1], output_size[1], 0]
    plt.plot(rect_x, rect_y, color="green", linewidth=1, label="Output rectangle", linestyle="--")

    plt.gca().invert_yaxis()  # cohérent avec les coordonnées image
    plt.xlabel("x-coordinate [pixels]")
    plt.ylabel("y-coordinate [pixels]")
    plt.title("Source vs Destination points with correspondences")
    plt.legend()
    plt.tight_layout()

    if output_plot_path:
        Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)

    if plot:
        plt.show()
    else:
        plt.close()

    return transform.params, output_size, dists


def plot_distance_between_collimation_lines(
    detected_vertical_edges: dict[str, int],
    detected_horizontal_ransac: dict[str, RANSACRegressor],
    stride: int = 256,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> None:
    """
    Plot and optionally save the vertical distance between two detected horizontal collimation lines.

    This function computes the pixel-wise vertical distance between the top and bottom
    collimation lines estimated using RANSAC regression models. The distances are
    evaluated at regular intervals along the x-axis and visualized as a curve. The mean
    distance is also displayed as a red dashed horizontal line.

    Parameters
    ----------
    detected_vertical_edges : dict[str, int]
        Dictionary containing the x-coordinates of the detected vertical edges.
        Expected keys: "left" and "right".
    detected_horizontal_ransac : dict[str, RANSACRegressor]
        Dictionary containing the RANSAC regression models for the top and bottom
        horizontal collimation lines. Expected keys: "top" and "bottom".
    stride : int, optional
        Step size in pixels used to sample x-coordinates along the image width.
        Default is 256.
    plot : bool, optional
        If True, the plot will be displayed. If False, it will be closed after
        saving or computation. Default is True.
    output_plot_path : str | None, optional
        Optional path where the generated plot will be saved. If None, no file is saved.

    Returns
    -------
    None
        This function does not return any value. It produces a plot or saves it to disk.

    Notes
    -----
    - The function assumes that the RANSAC models have already been fitted.
    - The vertical distance is computed as the absolute difference between
      the predicted y-values of the top and bottom lines.

    Example
    -------
    >>> plot_distance_between_collimation_lines(
    ...     detected_vertical_edges={"left": 0, "right": 2048},
    ...     detected_horizontal_ransac={"top": top_ransac, "bottom": bottom_ransac},
    ...     stride=128,
    ...     output_plot_path="results/distances.png"
    ... )
    """
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

    if output_plot_path:
        Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)

    if plot:
        plt.show()
    else:
        plt.close()


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def extract_raster_band_slice(
    raster_filepath: str | Path,
    padding: tuple[int, int] = (0, 0),
    band_size: int = 3000,
    stride: int = 10,
    side: str = "left",
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    """
    Extract a thin slice (band) from one side of a raster image for edge analysis.

    This function reads a narrow strip from one side (left, right, top, or bottom)
    of a raster file using Rasterio, optionally applying padding and downsampling
    along the axis of interest using averaged resampling.

    Args:
        raster_filepath (str): Path to the raster file to read.
        padding (tuple[int, int], optional): Number of pixels to exclude from the
            (left/right, top/bottom) edges of the raster. Default is (0, 0).
        band_size (int, optional): Width (or height) of the extracted slice in pixels.
            Default is 3000.
        stride (int, optional): Downsampling factor along the axis orthogonal to the band.
            A higher stride reduces the number of pixels via averaging. Default is 10.
        side (str, optional): Which side of the raster to extract. Must be one of
            "left", "right", "top", or "bottom". Default is "left".

    Returns:
        tuple[cv2.typing.MatLike, tuple[int, int]]:
            - subset: The extracted raster slice (2D NumPy array).
            - origin: A tuple (col_off, row_off) giving the upper-left offset (in pixels)
              of the extracted region within the original raster.

    Raises:
        ValueError: If `side` is not one of "left", "right", "top", or "bottom".

    Example:
        >>> subset, origin = extract_raster_band_slice("dem.tif", band_size=2000, stride=5, side="right")
        >>> plt.imshow(subset, cmap="gray")
        >>> print("Top-left offset:", origin)
    """
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
    image: cv2.typing.MatLike, peak_per_col: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Detect the most prominent peaks in each column of an image.

    For every column in the input image, this function extracts the intensity profile,
    finds all local maxima using `scipy.signal.find_peaks`, and keeps the specified
    number of peaks with the highest prominence. This is typically used to detect
    strong vertical structures or features across an image.

    Args:
        image (cv2.typing.MatLike): 2D grayscale image or intensity map.
        peak_per_col (int, optional): Number of most prominent peaks to retain per column.
            Default is 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - peaks_x: Array of x (column) coordinates for detected peaks.
            - peaks_y: Array of y (row) coordinates for detected peaks.

    Example:
        >>> peaks_x, peaks_y = detect_peaks_in_columns_v2(image, peak_per_col=3)
        >>> plt.imshow(image, cmap="gray")
        >>> plt.scatter(peaks_x, peaks_y, color="red", s=5)
    """
    peaks_x, peaks_y = [], []
    n_cols = image.shape[1]

    for col in range(n_cols):
        signal = image[:, col]

        peaks, properties = find_peaks(signal, prominence=0)
        prominences = properties["prominences"]

        top_indices = peaks[np.argsort(prominences)[-peak_per_col:]]

        for x in top_indices:
            peaks_x.append(col)
            peaks_y.append(x)

    return np.array(peaks_x), np.array(peaks_y)


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
