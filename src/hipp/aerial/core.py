"""
Copyright (c) 2025 HIPP developers
Description: functions for aerial fiducials manipulation
"""

import copy
import os
import statistics
from collections import defaultdict
from typing import TypedDict

import cv2
import rasterio

import hipp.aerial.quality_control as qc
import hipp.image
from hipp.aerial.fiducials import Fiducials, FiducialsCoordinate
from hipp.math import affine_matrix
from hipp.tools import points_picker


class FiducialDetection(TypedDict):
    approx_center: tuple[float, float]
    approx_score: float
    subpixel_center: tuple[float, float]
    subpixel_score: float


def create_fiducial_template_from_image(
    image: cv2.typing.MatLike,
    fiducial_coordinate: tuple[int, int] | None = None,
    distance_around_fiducial: int = 100,
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    """
    Create a fiducial template by cropping a portion of an input image around a fiducial point.

    Args:
        image (np.ndarray): The input grayscale image.
        fiducial_coordinate (tuple[int, int] | None, optional): The coordinate (x, y) of the fiducial point.
            If None, the function will interactively allow the user to pick a point.
        distance_around_fiducial (int, optional): The size of the region to crop around the fiducial point,
            in pixels. Defaults to 100.

    Returns:
        tuple:
            - np.ndarray: Cropped image (fiducial template).
            - tuple[int, int]: Fiducial coordinate used.

    Raises:
        ValueError: If no fiducial point is provided and the interactive point picker fails.
    """
    if len(image.shape) != 2:
        raise ValueError("Only grayscale images are supported.")

    if fiducial_coordinate is None:
        points = points_picker(image)
        if len(points) == 0:
            raise ValueError("No fiducial point was selected interactively.")
        fiducial_coordinate = points[0]

    x, y = fiducial_coordinate
    x_L = max(0, x - distance_around_fiducial)
    x_R = min(image.shape[1], x + distance_around_fiducial)
    y_T = max(0, y - distance_around_fiducial)
    y_B = min(image.shape[0], y + distance_around_fiducial)

    return image[y_T:y_B, x_L:x_R], fiducial_coordinate


# def create_pseudofiducials_templates_from_image(
#     image: cv2.typing.MatLike,
#     fiducials_coordinates: list[tuple[int, int]] | None = None,
#     distance_around_fiducial: int = 250,
#     threshold: int = 50,
# ) -> list[cv2.typing.MatLike]:
#     if fiducials_coordinates is None:
#         coords = points_picker(image, point_count=4)
#     else:
#         coords = fiducials_coordinates
#     if len(coords) != 4:
#         raise ValueError("Not enough fiducials coordinate, need to have 4")

#     results = []

#     for coord in coords:
#         fiducial_image = create_fiducial_template_from_image(image, coord, distance_around_fiducial)
#         _, binary_mask = cv2.threshold(fiducial_image, threshold, 255, cv2.THRESH_BINARY)
#         mask_bool = binary_mask == 255
#         masked_values = fiducial_image[mask_bool]

#         # calculate the mean and the standard deviation of the masked value
#         mean_val = np.mean(masked_values)
#         std_val = np.std(masked_values)

#         # generate some gaussian noise with the same values
#         noisy_values = np.random.normal(loc=mean_val, scale=std_val, size=masked_values.shape)

#         # clip value to keep them between 0 and 255
#         noisy_values = np.clip(noisy_values, 0, 255).astype(np.uint8)

#         fiducial_image[mask_bool] = noisy_values
#         results.append(fiducial_image)

#     return results


def detect_fiducials(
    image_path: str,
    corner_fiducial: cv2.typing.MatLike | None = None,
    midside_fiducial: cv2.typing.MatLike | None = None,
    subpixel_corner_fiducial: cv2.typing.MatLike | None = None,
    subpixel_midside_fiducial: cv2.typing.MatLike | None = None,
    subpixel_factor: float = 8,
    grid_size: int = 3,
) -> tuple[FiducialsCoordinate, Fiducials[float], Fiducials[float]]:
    """
    Detects fiducial markers in a large image by reading only specific blocks, optimizing memory usage.

    This function divides the image into a grid and reads only the blocks where fiducials are expected
    (corners and midsides), based on the presence of corresponding templates. It applies standard and
    subpixel template matching to locate the fiducials precisely.

    Args:
        image_path (str): Path to the input image file (e.g., TIFF). Only required regions are read.
        corner_fiducial (cv2.typing.MatLike, optional): Template used to detect fiducials at the image corners.
        midside_fiducial (cv2.typing.MatLike, optional): Template used to detect fiducials at the image midsides.
        subpixel_corner_fiducial (cv2.typing.MatLike, optional): Higher-resolution corner template for subpixel refinement.
        subpixel_midside_fiducial (cv2.typing.MatLike, optional): Higher-resolution midside template for subpixel refinement.
        subpixel_factor (float, optional): Upscaling factor used for subpixel template matching. Defaults to 8.
        grid_size (int): Size of the virtual grid used to divide the image (e.g., 3 for a 3×3 grid). Must be odd.

    Returns:
        tuple:
            - Fiducials: Dictionary mapping fiducial names (e.g., "corner_top_left") to their subpixel-corrected
              coordinates in image space: (float, float).
            - dict[str, float]: Approximate template matching scores for each fiducial.
            - dict[str, float]: Subpixel refinement confidence scores for each fiducial.

    Raises:
        ValueError: If `grid_size` is not an odd number.

    Notes:
        - The image is processed block-wise, avoiding full image loading—suitable for very large files.
        - Detected coordinates are adjusted from local block space to full image coordinates.
        - Fiducial positions are inferred from the grid: corners and midsides are computed based on index.
        - Both `corner_fiducial` and `midside_fiducial` must be provided to detect all positions.
        - Subpixel templates are mandatory if standard templates are given.
    """
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")
    blocs = {}
    if corner_fiducial is not None:
        blocs.update(
            {
                "corner_top_left": (0, 0),
                "corner_top_right": (0, grid_size - 1),
                "corner_bottom_left": (grid_size - 1, 0),
                "corner_bottom_right": (grid_size - 1, grid_size - 1),
            }
        )

    if midside_fiducial is not None:
        center = grid_size // 2
        blocs.update(
            {
                "midside_top": (0, center),
                "midside_bottom": (grid_size - 1, center),
                "midside_left": (center, 0),
                "midside_right": (center, grid_size - 1),
            }
        )
    fiducials_coord = FiducialsCoordinate()
    scores, subpixel_scores = Fiducials[float](), Fiducials[float]()
    with rasterio.open(image_path) as src:
        for bloc_name, (bloc_row, block_col) in blocs.items():
            bloc, (offset_x, offset_y) = hipp.image.read_image_block_grayscale(src, bloc_row, block_col, grid_size)

            fiducial = corner_fiducial if "corner" in bloc_name else midside_fiducial
            subpixel_fiducial = subpixel_corner_fiducial if "corner" in bloc_name else subpixel_midside_fiducial
            assert fiducial is not None  # for mypy
            assert subpixel_fiducial is not None

            fiducial_detection = detect_fiducial(bloc, fiducial, subpixel_fiducial, subpixel_factor)

            # We translate the coordinate of the sub bloc to the full image
            sx, sy = fiducial_detection["subpixel_center"]
            fiducials_coord[bloc_name] = (sx + offset_x, sy + offset_y)

            scores[bloc_name] = fiducial_detection["approx_score"]
            subpixel_scores[bloc_name] = fiducial_detection["subpixel_score"]

    fiducials_coord.compute_principal_point()
    return fiducials_coord, scores, subpixel_scores


def detect_fiducial(
    image: cv2.typing.MatLike,
    fiducial: cv2.typing.MatLike,
    subpixel_fiducial: cv2.typing.MatLike,
    subpixel_factor: float = 8,
) -> FiducialDetection:
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
    resized_crop = hipp.image.resize_img(crop, factor=subpixel_factor)

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

    return {
        "approx_center": (max_loc[0] + w / 2, max_loc[1] + h / 2),
        "approx_score": max_val,
        "subpixel_center": refined_center,
        "subpixel_score": sub_max_val,
    }


def detect_pseudofiducials(image_path: str, fiducials: list[cv2.typing.MatLike], grid_size: int = 3) -> None:
    """TODO"""
    pass


class MetadataImageRestituion(TypedDict, total=False):
    transformation_matrix: cv2.typing.MatLike
    rmse_before_transformation: float
    rmse_after_transformation: float


def image_restitution(
    detected_fiducials: FiducialsCoordinate,
    true_fiducials_mm: FiducialsCoordinate | dict[str, tuple[float, float]] | None,
    image_path: str | None = None,
    output_image_path: str | None = None,
    scanning_resolution_mm: float = 0.02,
    image_square_dim: int | None = 10800,
    interpolation_flag: int = cv2.INTER_CUBIC,
    clahe_enhancement: bool = True,
) -> MetadataImageRestituion:
    """
    Perform geometric image restitution using detected and reference fiducial markers.

    This function estimates the affine transformation between the detected fiducials and the reference
    positions (in millimeters), optionally applies it to an input image, crops the result around the
    principal point, and enhances the image using CLAHE. It also returns a metadata dictionary containing
    transformation matrices and registration quality metrics.

    Parameters
    ----------
    detected_fiducials : FiducialsCoordinate
        Fiducial marker coordinates detected in the raw image (in pixel units).
    true_fiducials_mm : FiducialsCoordinate | dict[str, tuple[float, float]] | None
        Ground truth positions of fiducials in millimeters. If None, no registration is performed.
    image_path : str | None, optional
        Path to the input grayscale image to be transformed. If None, no image processing is performed.
    output_image_path : str | None, optional
        Path to save the output image. Required if `image_path` is provided.
    scanning_resolution_mm : float, default=0.02
        Physical size of one pixel in millimeters (used to convert reference coordinates).
    image_square_dim : int | None, default=10800
        Size of the output square image (for cropping around the principal point). If None, no cropping is done.
    interpolation_flag : int, default=cv2.INTER_CUBIC
        OpenCV interpolation method used for image warping.
    clahe_enhancement : bool, default=True
        If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization) after transformation.

    Returns
    -------
    MetadataImageRestituion
        A dictionary containing the following keys (when available):
        - "transformation_matrix": 3x3 affine transformation applied to the image.
        - "rmse_before_transformation": RMSE between detected and reference fiducials (before alignment).
        - "rmse_after_transformation": RMSE after alignment.
        - Other intermediate fiducial sets if computed.

    Notes
    -----
    - The function applies transformation and cropping in a single warp if possible, which avoids
      multiple image resampling steps.
    - If no `true_fiducials_mm` is provided, the image is simply cropped using the detected principal point.
    - Fiducials that are not detected (i.e., `None`) are ignored in RMSE computation.
    """
    metadata: MetadataImageRestituion = {}
    transformed_fiducials = None

    # Compute transformation matrix and transform coordinates if requested
    if true_fiducials_mm is not None:
        true_fiducials_mm = FiducialsCoordinate(true_fiducials_mm)

        assert detected_fiducials["principal_point"] is not None

        img_ref_matrix = affine_matrix(
            scale_x=1 / scanning_resolution_mm,
            scale_y=-1 / scanning_resolution_mm,
            translate_x=detected_fiducials["principal_point"][0],
            translate_y=detected_fiducials["principal_point"][1],
        )
        true_fiducials = true_fiducials_mm.transform(img_ref_matrix)

        metadata["transformation_matrix"] = detected_fiducials.estimate_transformation_matrix(true_fiducials)

        transformed_fiducials = detected_fiducials.transform(metadata["transformation_matrix"])

        # calculate the rmse before and after the transformation
        metadata["rmse_before_transformation"] = qc.compute_rmse(true_fiducials, detected_fiducials)
        metadata["rmse_after_transformation"] = qc.compute_rmse(true_fiducials, transformed_fiducials)

    # cropping block
    if image_square_dim is not None:
        center = (
            transformed_fiducials["principal_point"]
            if transformed_fiducials is not None
            else detected_fiducials["principal_point"]
        )
        assert center is not None
        # calculate the transformation matrix for the crop (translation only)
        top_left_x = int(center[0] - image_square_dim // 2)
        top_left_y = int(center[1] - image_square_dim // 2)
        crop_transformation_matrix = affine_matrix(translate_x=-top_left_x, translate_y=-top_left_y)

        if "transformation_matrix" in metadata:
            metadata["transformation_matrix"] = crop_transformation_matrix @ metadata["transformation_matrix"]  # type: ignore [typeddict-item]
        else:
            metadata["transformation_matrix"] = crop_transformation_matrix

    # we transform the image only if the image_path and the output_image are set
    if image_path is not None and output_image_path is not None and "transformation_matrix" in metadata:
        output_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = output_image.shape[:2]

        # if a crop need to be apply we update the width and the height
        if image_square_dim is not None:
            width, height = image_square_dim, image_square_dim

        output_image = cv2.warpAffine(
            output_image, metadata["transformation_matrix"][:2], dsize=(width, height), flags=interpolation_flag
        )
        if clahe_enhancement:
            output_image = hipp.image.apply_clahe(output_image)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, output_image)

    return metadata


def filter_detected_fiducials(
    all_detections: dict[str, FiducialsCoordinate],
    all_scores: dict[str, Fiducials[float]],
    all_subpixel_scores: dict[str, Fiducials[float]],
    degree_threshold: float = 0.05,
    score_margin: float = 0.1,
) -> dict[str, FiducialsCoordinate]:
    """
    Process and validate fiducial detections based on angle consistency and scoring thresholds.

    This function performs a validation step on a set of detected fiducials using two criteria:
    1. Angular consistency between fiducial points (e.g. geometric layout).
    2. Confidence scores (both raw detection and subpixel refinement) compared to category-specific thresholds.

    Parameters:
        all_detections (dict[str, FiducialsCoordinate]):
            A dictionary mapping image IDs (or categories) to their detected fiducial coordinates.
        all_scores (dict[str, FiducialsScore]):
            A dictionary mapping image IDs to their detection confidence scores per fiducial.
        all_subpixel_scores (dict[str, FiducialsScore]):
            A dictionary mapping image IDs to subpixel refinement scores per fiducial.
        degree_threshold (float, optional):
            The maximum angular deviation (in degrees) allowed for geometric validation.
        score_margin (float, optional):
            The margin to subtract from the median score per category to establish a minimum threshold.

    Returns:
        dict[str, FiducialsCoordinate]:
            A deep-copied and cleaned version of the input detections, where invalid fiducials are set to `None`.
    """
    # Compute per-category score medians (used to set thresholds)
    score_median_by_categ = compute_median_score_by_category(all_scores)
    subpixel_score_median_by_categ = compute_median_score_by_category(all_subpixel_scores)

    # Compute score thresholds by subtracting a margin from the medians
    score_threshold_by_category = score_median_by_categ.apply(lambda x: x - score_margin)
    subpixel_score_threshold_by_category = subpixel_score_median_by_categ.apply(lambda x: x - score_margin)

    # Create a deep copy of detections to avoid mutating original input
    result = copy.deepcopy(all_detections)

    # Iterate over each category/image key
    for key in all_detections:
        # Validate angular consistency (returns a dict[name, bool])
        validation_angles = all_detections[key].validate_angles(degree_threshold)

        # Validate score thresholds (both raw and subpixel scores must exceed thresholds)
        validation_score = {
            name: all_scores[key][name] > score_threshold_by_category[name]
            and all_subpixel_scores[key][name] > subpixel_score_threshold_by_category[name]
            for name in all_scores[key]
        }

        # Combine angle validation and score validation (logical OR)
        validation = validation_angles.apply_with_key(lambda name, value: value or validation_score[name])

        # Invalidate any fiducial not passing the combined validation
        for fiducial_name in validation:
            if not validation[fiducial_name]:
                result[key][fiducial_name] = None

        # Recompute the principal point from remaining valid fiducials
        result[key].compute_principal_point()

    return result


def compute_median_score_by_category(all_scores: dict[str, Fiducials[float]]) -> Fiducials[float]:
    scores_by_category = defaultdict(list)

    for image_scores in all_scores.values():
        for fiducial_name, score in image_scores.items():
            scores_by_category[fiducial_name].append(score)

    median_by_category = Fiducials[float]()
    for fiducial_name, scores in scores_by_category.items():
        median_by_category[fiducial_name] = statistics.median(scores)

    return median_by_category
