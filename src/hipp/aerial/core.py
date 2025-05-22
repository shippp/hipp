"""
Copyright (c) 2025 HIPP developers
Description: functions for aerial fiducials manipulation
"""

import copy
import statistics
from collections import defaultdict
from typing import TypedDict

import cv2
import numpy as np
import rasterio

import hipp.image
import hipp.math
from hipp.aerial.fiducials import Fiducials, FiducialsCoordinate
from hipp.tools import points_picker


class MetadataImageRestituion(TypedDict, total=False):
    transformation_matrix: cv2.typing.MatLike
    fiducials_mm: FiducialsCoordinate
    transformed_fiducials: FiducialsCoordinate
    transformed_fiducials_mm: FiducialsCoordinate
    true_fiducials_mm_centered: FiducialsCoordinate


class FiducialDetection(TypedDict):
    approx_center: tuple[float, float]
    approx_score: float
    subpixel_center: tuple[float, float]
    subpixel_score: float


def create_fiducial_template_from_image(
    image: cv2.typing.MatLike,
    fiducial_coordinate: tuple[int, int] | None = None,
    distance_around_fiducial: int = 100,
) -> cv2.typing.MatLike:
    """
    Create a fiducial template by cropping a portion of an input image around a fiducial point.

    Args:
        image (np.ndarray): The input grayscale image.
        fiducial_coordinate (tuple[int, int] | None, optional): The coordinate (x, y) of the fiducial point.
            If None, the function will interactively allow the user to pick a point.
        distance_around_fiducial (int, optional): The size of the region to crop around the fiducial point,
            in pixels. Defaults to 100.

    Returns:
        np.ndarray: Cropped image (fiducial template).

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

    return image[y_T:y_B, x_L:x_R]


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


def image_restitution(
    image_path: str,
    detected_fiducials: FiducialsCoordinate,
    true_fiducials_mm: FiducialsCoordinate | dict[str, tuple[float, float]],
    scanning_resolution_mm: float = 0.025,
    image_square_dim: int = 10800,
    interpolation_flag: int = cv2.INTER_CUBIC,
    transform_coords: bool = True,
    transform_image: bool = True,
    crop_image: bool = True,
    clahe_enhancement: bool = True,
) -> tuple[cv2.typing.MatLike | None, MetadataImageRestituion]:
    """
    Performs geometric rectification, spatial alignment, and contrast enhancement of a scanned image
    based on detected and known fiducial markers.

    This function computes an affine transformation matrix to align the image with a physical reference frame
    defined by known fiducial positions (in millimeters). It optionally applies this transformation to the image,
    recenters it around the principal point, and enhances local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image_path (str): Path to the grayscale input image (typically high-resolution scan).
        detected_fiducials (dict[str, tuple[float, float]]): Pixel-space coordinates of detected fiducial markers
            (e.g., corners and midsides), usually obtained from template matching.
        true_fiducials_mm (dict[str, tuple[float, float]]): Known physical positions (in millimeters) of the same fiducials
            in a reference frame centered on the geometric principal point of the object.
        scanning_resolution_mm (float, optional): Pixel resolution of the scan in millimeters per pixel. Default is 0.025 mm/px.
        image_square_dim (int, optional): Final dimension (in pixels) of the cropped image. Image is cropped to a square
            centered on the principal point. Default is 10800.
        interpolation_flag (int, optional): OpenCV interpolation method used in image warping. Default is `cv2.INTER_CUBIC`.
        transform_coords (bool, optional): If True, compute and apply coordinate transformations (both pixel → mm and affine alignment).
        transform_image (bool, optional): If True, apply the affine transformation to the image.
        crop_image (bool, optional): If True, crop the image around the (transformed) principal point.
        clahe_enhancement (bool, optional): If True, apply CLAHE for local contrast enhancement.

    Returns:
        tuple:
            - np.ndarray | None: The processed image if any transformation is applied, else `None`.
            - MetadataImageRestituion: Dictionary of intermediate metadata for traceability and diagnostics, including:
                - 'transformation_matrix' (np.ndarray | None): The affine matrix used to rectify the image.
                - 'fiducials_mm' (dict | None): Detected fiducials converted from pixel coordinates to millimeters.
                - 'true_fiducials_mm_centered' (dict | None): Reference fiducials centered on the geometric principal point.
                - 'transformed_fiducials' (dict | None): Pixel fiducials after transformation.
                - 'transformed_fiducials_mm' (dict | None): Transformed fiducials expressed in physical (mm) space.

    Technical Notes:
        - The function first aligns the *true* fiducial positions with the origin by subtracting the geometric principal point.
          This ensures that the transformation matrix computed later aligns the detected image frame with a centered physical model.
        - The affine transformation is estimated using similarity transform between
          detected pixel positions and centered millimeter reference positions.
        - Once the transformation is computed, all fiducial coordinates can be mapped to their rectified positions in pixel and mm space.
        - When `transform_image=True`, `cv2.warpAffine` is applied using the transformation matrix. The image is geometrically corrected
          (shear, rotation, scale, translation) to align with the reference.
        - Cropping is centered on the principal point (either original or transformed) and ensures a fixed output size suitable for further analysis.
        - CLAHE (applied via `cv2.createCLAHE`) improves local contrast, particularly useful for uneven lighting or low dynamic range scans.
    """
    output_image = None
    metadata: MetadataImageRestituion = {}
    # Compute transformation matrix and transform coordinates if requested
    if transform_coords:
        true_fiducials_mm = FiducialsCoordinate(true_fiducials_mm)
        # here we align the true fiducials coordinate with the geometric principal point, to be in the same reference as detected fiducials
        metadata["true_fiducials_mm_centered"] = true_fiducials_mm.convert_in_camera_reference(1, False)

        # convert the detected fiducials in camera reference
        metadata["fiducials_mm"] = detected_fiducials.convert_in_camera_reference(scanning_resolution_mm)

        metadata["transformation_matrix"] = metadata["fiducials_mm"].estimate_transformation_matrix(
            metadata["true_fiducials_mm_centered"]
        )

        metadata["transformed_fiducials"] = detected_fiducials.transform(metadata["transformation_matrix"])
        metadata["transformed_fiducials_mm"] = metadata["fiducials_mm"].transform(metadata["transformation_matrix"])

    # Load image if any processing is requested
    if transform_image or crop_image or clahe_enhancement:
        output_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply geometric transformation to the image
        if transform_image:
            height, width = output_image.shape[:2]
            inverse_matrix = np.linalg.inv(metadata["transformation_matrix"])  # type: ignore[arg-type]
            output_image = cv2.warpAffine(
                output_image, inverse_matrix[:2], dsize=(width, height), flags=interpolation_flag
            )

        # Crop image around the principal point if requested
        if crop_image:
            center = (
                metadata["transformed_fiducials"]["principal_point"]
                if transform_coords
                else detected_fiducials["principal_point"]
            )
            assert center is not None
            output_image = hipp.image.crop_image_around_point(output_image, center, image_square_dim)

        # Apply CLAHE enhancement
        if clahe_enhancement:
            output_image = hipp.image.apply_clahe(output_image)
    return output_image, metadata


def process_fiducials_detection(
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
