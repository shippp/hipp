"""
Copyright (c) 2025 HIPP developers
Description: main function for aerial preprocessing
"""

import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from hipp.aerial.fiducials import (
    CORNER_FIDUCIAL_NAME,
    CORNER_KEYS,
    MIDSIDE_FIDUCIAL_NAME,
    MIDSIDE_KEYS,
    SUBPIXEL_CORNER_FIDUCIAL_NAME,
    SUBPIXEL_MIDSIDE_FIDUCIAL_NAME,
    compute_fiducial_transformation,
    compute_principal_point,
    get_fiducial_template_paths,
    get_pseudo_fiducial_paths,
    warp_fiducial_coordinates,
)
from hipp.image import apply_clahe, read_image_block_grayscale, resize_img
from hipp.math import nmad, transform_coord
from hipp.tools import pick_point_from_image


####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################
def create_fiducial_templates(
    input_image_path: str,
    output_directory: str,
    distance_around_fiducial: int = 100,
    subpixel_distance_around_fiducial: int = 100,
    corner: bool = False,
    midside: bool = False,
    fiducial_coordinate: tuple[int, int] | None = None,
    subpixel_center_coordinate: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Generates fiducial templates from an input image by cropping regions around specified points.

    This function allows creating either a corner or midside fiducial template at regular and subpixel resolutions. It prompts for user input to select fiducial coordinates if not provided, crops image patches around these points, and saves the resulting templates into the output directory. It returns the coordinates used for the fiducials and their subpixel centers.
    """

    # Ensure exactly one fiducial type is selected
    if (not corner and not midside) or (corner and midside):
        raise ValueError("Need either corner of midside")

    os.makedirs(output_directory, exist_ok=True)

    window_name = "fiducial template"

    # Create and save the regular-resolution fiducial template if needed
    full_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if full_image is None:
        raise FileNotFoundError(f"Error while opening the file {input_image_path}")

    if fiducial_coordinate is None:
        fiducial_coordinate = pick_point_from_image(full_image, window_name, destroy_window=True)
        assert fiducial_coordinate is not None
    fiducial_image = _crop_around_point(full_image, fiducial_coordinate, distance_around_fiducial)

    # save the first fiducial template image
    fiducial_name = MIDSIDE_FIDUCIAL_NAME if midside else CORNER_FIDUCIAL_NAME
    cv2.imwrite(os.path.join(output_directory, fiducial_name), fiducial_image)

    # SUBPIXEL
    enhanced_image = resize_img(fiducial_image)
    if subpixel_center_coordinate is None:
        subpixel_center_coordinate = pick_point_from_image(enhanced_image, window_name, destroy_window=True)
        assert subpixel_center_coordinate is not None
    subpixel_fiducial_image = _crop_around_point(
        enhanced_image, subpixel_center_coordinate, subpixel_distance_around_fiducial
    )
    # save the subpixel fiducial template image
    subpixel_fiducial_name = SUBPIXEL_MIDSIDE_FIDUCIAL_NAME if midside else SUBPIXEL_CORNER_FIDUCIAL_NAME
    cv2.imwrite(os.path.join(output_directory, subpixel_fiducial_name), subpixel_fiducial_image)

    return {"fiducial_coordinate": fiducial_coordinate, "subpixel_center_coordinate": subpixel_center_coordinate}


def create_pseudo_fiducial_templates(
    input_image_path: str,
    output_directory: str,
    side: str = "midside_left",
    distance_around_fiducial: int = 100,
    coordinate: tuple[int, int] | None = None,
    center_coordinate: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int]]:
    if side not in CORNER_KEYS + MIDSIDE_KEYS:
        raise ValueError(f"Side need to be in {CORNER_KEYS + MIDSIDE_KEYS}")

    window_name = "Pseudo Fiducial Template"

    full_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    assert full_image is not None

    if coordinate is None:
        window_title = f"Pick the {side} pseudo fiducial markers"
        coordinate = pick_point_from_image(full_image, window_name, window_title, True)
        assert coordinate is not None
    fiducial_image = _crop_around_point(full_image, coordinate, distance_around_fiducial)

    # save the pseudo fiducial image
    cv2.imwrite(os.path.join(output_directory, f"pseudo_fiducial_{side}.png"), fiducial_image)

    if center_coordinate is None:
        window_title = "Pick the center of the pseudo fiducial markers"
        center_coordinate = pick_point_from_image(fiducial_image, window_name, window_title, True)
        assert center_coordinate is not None

    # save the center of the pseudo fiducial in a csv
    center_csv_path = os.path.join(output_directory, f"pseudo_fiducial_{side}.csv")
    pd.DataFrame([center_coordinate]).to_csv(center_csv_path, index=False, header=False)

    return {"coordinate": coordinate, "center_coordinate": center_coordinate}


def iter_detect_fiducials(
    images_directory: str,
    fiducials_directory: str,
    subpixel_factor: float = 8,
    grid_size: int = 3,
    progress_bar: bool = True,
    max_workers: int = 5,
) -> pd.DataFrame:
    """
    Performs fiducial detection on a collection of images in parallel.

    This function processes all TIFF images found in the specified images directory, detecting fiducials using templates from the fiducials directory. It supports subpixel accuracy and grid-based searching, executes detections concurrently up to a maximum number of workers, and optionally displays a progress bar. The results are collected and returned as a DataFrame indexed by image ID.
    """

    image_paths = sorted(glob.glob(os.path.join(images_directory, "*.tif")))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_path in image_paths:
            futures.append(
                executor.submit(detect_fiducials, image_path, fiducials_directory, subpixel_factor, grid_size)
            )
        iterable = (
            tqdm(as_completed(futures), total=len(futures), desc="Fiducial detections", unit="Image")
            if progress_bar
            else as_completed(futures)
        )

        for future in iterable:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[!] Error: {e}")

    df = pd.DataFrame(results)
    df = df.set_index("image_id").sort_index()

    return df


def iter_detect_pseudo_fiducials(
    images_directory: str,
    fiducials_directory: str,
    grid_size: int = 3,
    progress_bar: bool = True,
    max_workers: int = 5,
) -> pd.DataFrame:
    """
    Perform pseudo-fiducial detection on multiple images in parallel.

    This function iterates over all `.tif` images within a specified directory and applies
    `detect_pseudo_fiducials()` to each one in parallel using a process pool. It aggregates
    all individual detection results into a single pandas DataFrame for further analysis
    or quality assessment.

    Parameters
    ----------
    images_directory : str
        Path to the directory containing the input `.tif` images to process.
    fiducials_directory : str
        Path to the directory containing fiducial templates and metadata, as required by
        `detect_pseudo_fiducials()`.
    grid_size : int, optional
        The grid subdivision size used to locate fiducials within each image.
        Must be an odd number. Default is 3 (for the standard 8-fiducial layout).
    progress_bar : bool, optional
        Whether to display a progress bar during processing. Default is True.
    max_workers : int, optional
        Maximum number of parallel worker processes. Default is 5.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by image ID, containing for each image:
        - Fiducial coordinates (`{key}_x`, `{key}_y`) for all detected points.
        - Corresponding template matching scores (`{key}_score`).
        The resulting DataFrame is sorted alphabetically by image ID.

    Raises
    ------
    Exception
        Any unhandled exception during detection is caught and printed, but the process continues
        for the remaining images. Images that fail detection are skipped in the output DataFrame.

    Notes
    -----
    - The function uses `concurrent.futures.ProcessPoolExecutor` for parallel execution,
      which can significantly speed up processing on multi-core systems.
    - A progress bar (via `tqdm`) provides feedback on detection progress when enabled.
    - Each detection task calls `detect_pseudo_fiducials()` independently and safely.

    See Also
    --------
    detect_pseudo_fiducials : Detects all pseudo-fiducials within a single image.
    detect_pseudo_fiducial : Detects one fiducial mark using template matching.
    get_pseudo_fiducial_paths : Loads fiducial templates and their anchor metadata.
    """
    image_paths = sorted(glob.glob(os.path.join(images_directory, "*.tif")))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_path in image_paths:
            futures.append(executor.submit(detect_pseudo_fiducials, image_path, fiducials_directory, grid_size))
        iterable = (
            tqdm(as_completed(futures), total=len(futures), desc="Fiducial detections", unit="Image")
            if progress_bar
            else as_completed(futures)
        )

        for future in iterable:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[!] Error: {e}")

    df = pd.DataFrame(results)
    df = df.set_index("image_id").sort_index()

    return df


def filter_detected_fiducials(
    detected_fiducials_df: pd.DataFrame, score_threshold: float = 0.1, sigma: float = 3
) -> pd.DataFrame:
    """
    Filters detected fiducials based on correlation scores and spatial consistency,
    and computes the principal point for each image.

    The function performs the following steps:
    1. Extracts `_score` columns from the input DataFrame and constructs a mask
       for fiducials with scores above the median minus `score_threshold`.
    2. Extracts `_x` and `_y` columns representing fiducial coordinates, and
       creates a mask for points that lie within `sigma` Normalized Median Absolute
       Deviations (NMAD) from the median coordinate of each fiducial.
    3. Combines the score mask and spatial mask to generate a final mask of valid
       fiducials. Invalid points are replaced with `NaN`.
    4. Computes the principal point `(principal_point_x, principal_point_y)` for
       each image based on the filtered fiducials.
    5. Issues a warning if the principal point could not be computed for any image.

    Parameters
    ----------
    detected_fiducials_df : pd.DataFrame
        DataFrame containing detected fiducials for each image. Expected to have:
            - Columns ending with `_score` for the detection scores.
            - Columns ending with `_x` or `_y` for fiducial coordinates.
            - Index representing image identifiers.
    score_threshold : float, optional
        Margin below the median score used to accept fiducials (default: 0.1).
    sigma : float, optional
        Number of NMADs used to define acceptable coordinate deviation
        from the median (default: 3).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame of fiducial coordinates with the following properties:
            - Coordinates not passing the masks are set to `NaN`.
            - Adds columns `principal_point_x` and `principal_point_y`.
            - Maintains the same index as `detected_fiducials_df`.

    Warnings
    --------
    A `UserWarning` is raised if the principal point cannot be computed for
    any images, listing the affected image identifiers.

    Notes
    -----
    - The NMAD (Normalized Median Absolute Deviation) is used instead of standard
      deviation to robustly handle outliers in fiducial coordinates.
    - The function assumes the presence of a `compute_principal_point(row)` function
      that returns the principal point as a tuple or list `(x, y)` for a row of coordinates.
    """
    # construct the score df with only _score columns
    df_score = detected_fiducials_df.filter(regex=r"(_score)$")
    df_score.columns = df_score.columns.str.replace(r"_score$", "", regex=True)

    # construct the x y df with only columns finishing by _x or _y
    df_xy = detected_fiducials_df.filter(regex=r"(_x|_y)$")

    # create the first score mask where we accept all score above the median - score_threshold
    mask_score = df_score >= df_score.median() - score_threshold

    # create the second mask where we accept all point if their are not too far from median point
    upper_threshold = df_xy.apply(lambda col: np.median(col) + sigma * nmad(col), axis=0)
    lower_threshold = df_xy.apply(lambda col: np.median(col) - sigma * nmad(col), axis=0)

    mask_coord_xy = (df_xy < upper_threshold) & (df_xy > lower_threshold)

    mask_coord = mask_coord_xy.T.groupby(mask_coord_xy.columns.str.rsplit("_", n=1).str[0]).agg(all).T

    final_mask = (mask_score | mask_coord) & (df_score > 0.5)

    final_mask_xy = pd.concat([final_mask.add_suffix("_x"), final_mask.add_suffix("_y")], axis=1).sort_index(axis=1)

    filtered_xy = df_xy.where(final_mask_xy, np.nan)

    # compute principal points and store them in principal_point_x, principal_point_y
    filtered_xy[["principal_point_x", "principal_point_y"]] = filtered_xy.apply(
        lambda row: pd.Series(compute_principal_point(row)), axis=1
    )
    # Check for missing principal points
    missing_mask = filtered_xy[["principal_point_x", "principal_point_y"]].isna().any(axis=1)
    if missing_mask.any():
        missing_ids = filtered_xy.index[missing_mask].tolist()
        warnings.warn(
            f"Principal point could not be computed for {len(missing_ids)} detection(s): {missing_ids}",
            UserWarning,
        )
    return filtered_xy


def compute_transformations(
    detected_fiducials_df: pd.DataFrame,
    true_fiducials_mm: pd.Series | None,
    image_square_dim: int | None = 10800,
    scanning_resolution_mm: float = 0.02,
) -> dict[str, cv2.typing.MatLike]:
    """
    Computes geometric transformation matrices for a set of images based on detected and true fiducial points.

    For each image, this function optionally calculates an affine transformation aligning detected fiducials with known true fiducials, applies scaling and flipping based on scanning resolution, and recenters the image around a specified square dimension. It returns a dictionary mapping image names to their corresponding 3x3 transformation matrices.
    """

    result: dict[str, cv2.typing.MatLike] = {}
    for image_name, detected_fiducials in detected_fiducials_df.iterrows():
        matrix = np.eye(3)
        principal_point = (detected_fiducials["principal_point_x"], detected_fiducials["principal_point_y"])

        if any(np.isnan(x) for x in principal_point):
            warnings.warn(
                f"Skip {image_name} : no principal point found.",
                UserWarning,
            )
            continue

        # if true_fiducials_mm is set we compute an affine transformation between
        # detected fiducials and true fiducials
        if true_fiducials_mm is not None:
            M = np.array(
                [
                    [1 / scanning_resolution_mm, 0, principal_point[0]],
                    [0, -1 / scanning_resolution_mm, principal_point[1]],
                    [0, 0, 1],
                ]
            )
            true_fiducials = warp_fiducial_coordinates(true_fiducials_mm, M)
            matrix = compute_fiducial_transformation(detected_fiducials, true_fiducials) @ matrix

        # if image_square_dim is set we translate the centered the image from the principal point in the center
        # of the new image_square_dim
        if image_square_dim is not None:
            pp_transformed = transform_coord(principal_point, matrix)
            top_left_x = int(pp_transformed[0] - image_square_dim // 2)
            top_left_y = int(pp_transformed[1] - image_square_dim // 2)

            M = np.array(
                [
                    [1, 0, -top_left_x],
                    [0, 1, -top_left_y],
                    [0, 0, 1],
                ]
            )
            matrix = M @ matrix

        result[image_name] = matrix
    return result


def iter_image_restitution(
    images_directory: str,
    output_directory: str,
    transformations: dict[str, cv2.typing.MatLike],
    image_square_dim: int | None = 10800,
    interpolation_flag: int = cv2.INTER_CUBIC,
    clahe_enhancement: bool = True,
    max_workers: int = 5,
    progress_bar: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Coordinates the parallel processing of image restitution tasks.

    This function distributes the workload of applying geometric transformations to images across multiple worker processes, optionally displaying a progress bar. It handles image loading, transformation, enhancement, and saving to an output directory.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_id, transformation_matrix in transformations.items():
            output_image_path = os.path.join(output_directory, image_id)
            if not os.path.exists(output_image_path) or overwrite:
                futures.append(
                    executor.submit(
                        restitute_image,
                        os.path.join(images_directory, image_id),
                        output_image_path,
                        transformation_matrix,
                        image_square_dim,
                        interpolation_flag,
                        clahe_enhancement,
                    )
                )
        iterable = tqdm(as_completed(futures), total=len(futures)) if progress_bar else as_completed(futures)

        for future in iterable:
            try:
                future.result()
            except Exception as e:
                print(f"[!] Error: {e}")


def warp_fiducials_df(fiducials_df: pd.DataFrame, transformations: dict[str, cv2.typing.MatLike]) -> pd.DataFrame:
    """
    Applies geometric transformations to each row of fiducial coordinates in a DataFrame.

    This function uses a dictionary of transformation matrices keyed by the DataFrame index to warp each set of fiducial points accordingly, producing a new DataFrame with transformed coordinates.
    """
    return fiducials_df.apply(lambda row: warp_fiducial_coordinates(row, transformations[row.name]), axis=1)


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################
def detect_fiducials(
    image_path: str,
    fiducials_directory: str,
    subpixel_factor: float = 8,
    grid_size: int = 3,
) -> pd.Series:
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")

    paths = get_fiducial_template_paths(fiducials_directory)

    corner_config = (
        {
            "corner_top_left": (0, 0),
            "corner_top_right": (0, grid_size - 1),
            "corner_bottom_left": (grid_size - 1, 0),
            "corner_bottom_right": (grid_size - 1, grid_size - 1),
        },
        "corner_fiducial_path",
        "subpixel_corner_fiducial_path",
    )
    midside_config = (
        {
            "midside_top": (0, grid_size // 2),
            "midside_bottom": (grid_size - 1, grid_size // 2),
            "midside_left": (grid_size // 2, 0),
            "midside_right": (grid_size // 2, grid_size - 1),
        },
        "midside_fiducial_path",
        "subpixel_midside_fiducial_path",
    )

    result = {"image_id": os.path.basename(image_path)}

    with rasterio.open(image_path) as src:
        for blocs, k1, k2 in (corner_config, midside_config):
            if k1 in paths and k2 in paths:
                fiducial = cv2.imread(paths[k1], cv2.IMREAD_GRAYSCALE)
                subpixel_fiducial = cv2.imread(paths[k2], cv2.IMREAD_GRAYSCALE)
                assert fiducial is not None and subpixel_fiducial is not None

                for bloc_name, (bloc_row, block_col) in blocs.items():
                    block, (offset_x, offset_y) = read_image_block_grayscale(
                        src, bloc_row, block_col, (grid_size, grid_size)
                    )

                    center, score = detect_fiducial(block, fiducial, subpixel_fiducial, subpixel_factor)

                    result[f"{bloc_name}_x"] = center[0] + offset_x  # type: ignore[assignment]
                    result[f"{bloc_name}_y"] = center[1] + offset_y  # type: ignore[assignment]
                    result[f"{bloc_name}_score"] = score  # type: ignore[assignment]

    return pd.Series(result)


def detect_fiducial(
    image: cv2.typing.MatLike,
    fiducial: cv2.typing.MatLike,
    subpixel_fiducial: cv2.typing.MatLike,
    subpixel_factor: float = 8,
) -> tuple[tuple[float, float], float]:
    """
    Detects a fiducial marker in an image using standard and subpixel template matching.

    This function first performs a coarse detection of the fiducial using OpenCV's normalized cross-correlation
    on the original template. It then refines this detection by extracting the matched region, upscaling it, and
    applying template matching again with a higher-resolution subpixel template to achieve subpixel accuracy.

    Args:
        image (cv2.typing.MatLike): Input grayscale or color image in which the fiducial marker is to be detected.
        fiducial (cv2.typing.MatLike): Template used for initial coarse template matching.
        subpixel_fiducial (cv2.typing.MatLike): High-resolution template used for subpixel refinement.
        subpixel_factor (float, optional): Upscaling factor applied to the cropped region for subpixel matching.
                                           Default is 8.

    Returns:
        FiducialDetection: A dictionary with the following fields:
            - "approx_center" (tuple[float, float]): Approximate (x, y) center of the fiducial in image coordinates.
            - "approx_score" (float): Correlation score from the initial coarse template matching.
            - "subpixel_center" (tuple[float, float]): Refined (x, y) center after subpixel matching.
            - "subpixel_score" (float): Correlation score from the subpixel-level template matching.

    Notes:
        - Uses OpenCV's `cv2.matchTemplate` with `cv2.TM_CCOEFF_NORMED` for both coarse and subpixel matching.
        - The subpixel matching is applied within a cropped region of the input image, centered at the coarse match location.
        - The cropped region is upscaled before matching with the subpixel template to improve localization precision.
        - The output coordinates are returned in the original image's coordinate space (not in the upscaled space).
    """
    h, w = fiducial.shape[:2]

    # Coarse template matching
    coarse_result = cv2.matchTemplate(image, fiducial, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(coarse_result)

    # Extract the matched region for subpixel refinement
    crop = image[max_loc[1] : max_loc[1] + h, max_loc[0] : max_loc[0] + w]
    resized_crop = resize_img(crop, factor=subpixel_factor)

    # Subpixel matching
    sub_result = cv2.matchTemplate(resized_crop, subpixel_fiducial, cv2.TM_CCOEFF_NORMED)
    _, sub_max_val, _, sub_max_loc = cv2.minMaxLoc(sub_result)

    sub_h, sub_w = subpixel_fiducial.shape[:2]

    # Compute subpixel center
    sub_center_x = (sub_max_loc[0] + sub_w / 2) / subpixel_factor
    sub_center_y = (sub_max_loc[1] + sub_h / 2) / subpixel_factor
    refined_center = (
        max_loc[0] + sub_center_x,
        max_loc[1] + sub_center_y,
    )

    return refined_center, max_val


def detect_pseudo_fiducials(
    image_path: str,
    fiducials_directory: str,
    grid_size: int = 3,
) -> pd.Series:
    """
    Detect pseudo-fiducial marks in a grid layout within a given image.

    This function locates all fiducial marks (corners and midsides) in an aerial or
    calibration image by performing template matching for each expected fiducial position.
    Each fiducial template is matched against a corresponding sub-block of the image,
    determined by a grid-based spatial layout.

    The function returns a pandas Series containing the detected fiducial coordinates
    (x, y) and their associated correlation scores for quality assessment.

    Parameters
    ----------
    image_path : str
        Path to the input grayscale image in which fiducial marks are to be detected.
    fiducials_directory : str
        Path to the directory containing fiducial templates and metadata. The directory
        must be structured as expected by `get_pseudo_fiducial_paths()`, which provides
        template file paths and their anchor points.
    grid_size : int, optional
        The number of divisions in both horizontal and vertical directions for splitting
        the image into regions where fiducials are expected. Must be an odd number.
        Default is 3 (corresponding to corners and midsides).

    Returns
    -------
    pandas.Series
        A Series containing:
        - `"image_id"` : the basename of the processed image file.
        - For each fiducial key (e.g. `"corner_top_left"`, `"midside_top"`, ...):
            - `"{key}_x"` : detected x-coordinate (in pixels).
            - `"{key}_y"` : detected y-coordinate (in pixels).
            - `"{key}_score"` : normalized cross-correlation score from template matching.

    Raises
    ------
    ValueError
        If `grid_size` is not an odd number.

    Notes
    -----
    - The image is divided into `grid_size × grid_size` blocks, and each fiducial
      is searched only within its corresponding block.
    - The helper function `detect_pseudo_fiducial()` is used internally to find the
      best match and compute its confidence score.
    - The function assumes the fiducial layout follows a symmetric grid pattern
      (e.g. 3×3 for 8 fiducials: 4 corners + 4 midsides).

    See Also
    --------
    detect_pseudo_fiducial : Detect a single fiducial mark using template matching.
    get_pseudo_fiducial_paths : Retrieve template paths and anchor positions for fiducial marks.
    """
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")

    paths = get_pseudo_fiducial_paths(fiducials_directory)

    mapping = {
        "corner_top_left": (0, 0),
        "corner_top_right": (0, grid_size - 1),
        "corner_bottom_left": (grid_size - 1, 0),
        "corner_bottom_right": (grid_size - 1, grid_size - 1),
        "midside_top": (0, grid_size // 2),
        "midside_bottom": (grid_size - 1, grid_size // 2),
        "midside_left": (grid_size // 2, 0),
        "midside_right": (grid_size // 2, grid_size - 1),
    }

    result = {"image_id": os.path.basename(image_path)}

    with rasterio.open(image_path) as src:
        for key, (f_path, template_anchor) in paths.items():
            fiducial = cv2.imread(str(f_path), cv2.IMREAD_GRAYSCALE)
            assert fiducial is not None

            bloc_row, block_col = mapping[key]
            block, (offset_x, offset_y) = read_image_block_grayscale(src, bloc_row, block_col, (grid_size, grid_size))

            center, score = detect_pseudo_fiducial(block, fiducial, template_anchor)

            result[f"{key}_x"] = center[0] + offset_x  # type: ignore[assignment]
            result[f"{key}_y"] = center[1] + offset_y  # type: ignore[assignment]
            result[f"{key}_score"] = score  # type: ignore[assignment]

    return pd.Series(result)


def detect_pseudo_fiducial(
    image: cv2.typing.MatLike, template: cv2.typing.MatLike, template_anchor: tuple[int, int]
) -> tuple[tuple[float, float], float]:
    """
    Detect a pseudo-fiducial mark in an image using multi-filter template matching.

    This function performs template matching between a grayscale image and a fiducial
    template, applying multiple image filters to improve robustness. It averages the
    normalized cross-correlation results obtained from each filter and selects the
    location with the highest matching score.

    The returned coordinates correspond to the fiducial center, computed by offsetting
    the best match position with the template anchor point (i.e., the point of interest
    within the template image).

    Parameters
    ----------
    image : cv2.typing.MatLike
        Input grayscale image in which to search for the fiducial mark.
    template : cv2.typing.MatLike
        Template image of the fiducial mark to be matched.
    template_anchor : tuple[int, int]
        Coordinates (x, y) of the reference point inside the template image,
        typically the geometric center of the fiducial mark.

    Returns
    -------
    tuple[tuple[float, float], float]
        A tuple containing:
        - The detected fiducial center coordinates (x, y) in pixel units.
        - The maximum normalized correlation score (float) indicating match confidence.

    Notes
    -----
    - Two filters are applied before matching: identity (raw image) and Laplacian
      edge enhancement. The correlation maps from both are averaged to improve
      detection robustness.
    - The matching method used is `cv2.TM_CCOEFF_NORMED`.
    """
    # Define filters
    filters = [
        lambda img: img,
        lambda img: cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F)),
    ]

    # Compute template matching for each filter
    tpl_results = [cv2.matchTemplate(f(image), f(template), cv2.TM_CCOEFF_NORMED) for f in filters]

    # Average correlation maps
    avg_result = np.mean(np.array(tpl_results), axis=0)

    _, max_val, _, max_loc = cv2.minMaxLoc(avg_result)

    # max_loc = top-left corner of the best match
    # template_anchor   = point of interest inside the template (in template coordinates)
    detected_center = (
        max_loc[0] + template_anchor[0],
        max_loc[1] + template_anchor[1],
    )

    return detected_center, max_val


def restitute_image(
    image_path: str,
    output_image_path: str,
    transformation_matrix: cv2.typing.MatLike,
    image_square_dim: int | None = 10800,
    interpolation_flag: int = cv2.INTER_CUBIC,
    clahe_enhancement: bool = True,
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert image is not None
    height, width = image.shape[:2]

    # if a crop need to be apply we update the width and the height
    if image_square_dim is not None:
        width, height = image_square_dim, image_square_dim

    image = cv2.warpAffine(image, transformation_matrix[:2], dsize=(width, height), flags=interpolation_flag)
    if clahe_enhancement:
        image = apply_clahe(image)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image)


def _crop_around_point(
    image: cv2.typing.MatLike, point: tuple[int, int], distance_around_point: int
) -> cv2.typing.MatLike:
    x, y = point
    x_L = max(0, x - distance_around_point)
    x_R = min(image.shape[1], x + distance_around_point)
    y_T = max(0, y - distance_around_point)
    y_B = min(image.shape[0], y + distance_around_point)

    return image[y_T:y_B, x_L:x_R]
