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
    _get_fiducial_template_paths,
    compute_fiducial_transformation,
    compute_principal_point,
    filter_by_angle,
    filter_scores_by_local_median,
    warp_fiducial_coordinates,
)
from hipp.image import apply_clahe, read_image_block_grayscale, resize_img
from hipp.math import transform_coord
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


def filter_detected_fiducials(
    detected_fiducials_df: pd.DataFrame,
    score_threshold: float = 0.1,
    angle_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Filters detected fiducials based on matching score and angle thresholds to improve data quality.

    This function applies local median filtering on scores and filters by angle deviation, then combines the filtered results. It also recalculates the principal points for each detection, warning if any principal points could not be computed. The output is a cleaned DataFrame with filtered fiducials.
    """

    # filtering with local median and remove score columns
    filtered_scores = filter_scores_by_local_median(detected_fiducials_df, score_threshold)

    filtered_angles = filter_by_angle(detected_fiducials_df, angle_threshold)

    # combine both filtering
    combined = filtered_scores.copy()
    for col in filtered_scores.columns:
        if col in filtered_angles.columns:
            combined[col] = combined[col].fillna(filtered_angles[col])

    # compute principal points and store them in principal_point_x, principal_point_y
    combined[["principal_point_x", "principal_point_y"]] = combined.apply(
        lambda row: pd.Series(compute_principal_point(row)), axis=1
    )
    # Check for missing principal points
    missing_mask = combined[["principal_point_x", "principal_point_y"]].isna().any(axis=1)
    if missing_mask.any():
        missing_ids = combined.index[missing_mask].tolist()
        warnings.warn(
            f"Principal point could not be computed for {len(missing_ids)} detection(s): {missing_ids}",
            UserWarning,
        )
    return combined


def open_camera_model_intrinsics(csv_file: str) -> tuple[float, pd.Series]:
    """
    Load scanning resolution and true fiducial positions from a camera model CSV file.

    :param csv_file: Path to the CSV file containing camera intrinsics.
    :return: A tuple containing the scanning resolution (in mm) and a Series of fiducial positions.
    :raises ValueError: If the file format is invalid or required keys are missing.
    """
    try:
        df_row = pd.read_csv(csv_file).iloc[0]
        df_row.index = df_row.index.str.replace("_mm", "", regex=False)
        scanning_resolution_mm = float(df_row["pixel_pitch"])

        fiducials_keys = [key + suffix for key in CORNER_KEYS + MIDSIDE_KEYS for suffix in ["_x", "_y"]]
        true_fiducials_mm = df_row[fiducials_keys]
        return scanning_resolution_mm, true_fiducials_mm
    except Exception as e:
        raise ValueError(
            "Invalid CSV format. Expected structure similar to:\n"
            "https://github.com/shippp/hipp/blob/main/notebooks/data/aerial/camera_model_intrinsics.csv"
        ) from e


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
) -> None:
    """
    Coordinates the parallel processing of image restitution tasks.

    This function distributes the workload of applying geometric transformations to images across multiple worker processes, optionally displaying a progress bar. It handles image loading, transformation, enhancement, and saving to an output directory.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_id, transformation_matrix in transformations.items():
            futures.append(
                executor.submit(
                    restitute_image,
                    os.path.join(images_directory, image_id),
                    os.path.join(output_directory, image_id),
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

    paths = _get_fiducial_template_paths(fiducials_directory)

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


def restitute_image(
    image_path: str,
    output_image_path: str,
    transformation_matrix: cv2.typing.MatLike,
    image_square_dim: int | None = 10800,
    interpolation_flag: int = cv2.INTER_CUBIC,
    clahe_enhancement: bool = True,
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
