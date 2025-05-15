"""
Module: fiducials.py
Author: godinlu
Date: 28
Description: functions for aerial fiducials manipulation
"""

from typing import cast

import cv2
import numpy as np
import rasterio
from skimage.transform import SimilarityTransform

import hipp.image
from hipp.tools import points_picker
from hipp.typing import DetectedFiducials, FiducialDetection, MetadataImageRestituion


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
) -> tuple[DetectedFiducials, dict[str, float], dict[str, float]]:
    """
    Detects fiducial markers efficiently from a large image by reading only specific blocks from disk.

    This function avoids loading the full image into memory. Instead, it divides the image into a grid
    and reads only the relevant blocks (corners and midsides) using lightweight I/O operations.
    Each block is scanned using OpenCV's template matching to detect fiducial markers, and optionally refined
    using higher-resolution subpixel templates.

    Args:
        image_path (str): Path to the image file (e.g., TIFF). Only the needed regions will be read.
        corner_fiducial (cv2.typing.MatLike, optional): Template for detecting corner fiducials.
        midside_fiducial (cv2.typing.MatLike, optional): Template for detecting midside fiducials.
        subpixel_corner_fiducial (cv2.typing.MatLike, optional): Higher-res corner fiducial template for subpixel refinement.
        subpixel_midside_fiducial (cv2.typing.MatLike, optional): Higher-res midside template for subpixel refinement.
        subpixel_factor (float, optional): Upscaling factor for subpixel refinement. Defaults to 8.
        grid_size (int): Size of the grid used to divide the image (e.g., 3 means 3x3 grid). Must be odd.

    Returns:
        DetectedFiducials: A dictionary mapping fiducial labels (e.g., "corner_top_left") to:
            - "approx_center": (float, float), approximate center in full image coordinates.
            - "approx_score": float, score of basic template matching.
            - "subpixel_center": (float, float or None), refined center coordinates if available.
            - "subpixel_score": float or None, confidence score for subpixel refinement.

    Raises:
        ValueError: If grid_size is not an odd number.

    Notes:
        - This function is optimized for memory efficiency and is ideal for large TIFF images.
        - It assumes that fiducials are located at fixed positions (corners and midsides) in the grid.
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
    fiducials_detection = {}
    scores, subpixel_scores = {}, {}
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
            fiducials_detection[bloc_name] = (sx + offset_x, sy + offset_y)

            scores[bloc_name] = fiducial_detection["approx_score"]
            subpixel_scores[bloc_name] = fiducial_detection["subpixel_score"]

    return fiducials_detection, scores, subpixel_scores


def detect_fiducial(
    image: cv2.typing.MatLike,
    fiducial: cv2.typing.MatLike,
    subpixel_fiducial: cv2.typing.MatLike,
    subpixel_factor: float = 8,
) -> FiducialDetection:
    """
    Detects a fiducial marker in a given image using template matching. It locates the fiducial marker
    and refines the detection using subpixel accuracy if a subpixel fiducial image is provided. The function
    returns both the approximate and subpixel-level detection centers, along with their corresponding confidence scores.

    Args:
        image (cv2.typing.MatLike): The input image (grayscale or color) in which the fiducial marker is to be detected.
        fiducial (cv2.typing.MatLike): The template image of the fiducial marker that is used for initial template matching.
        subpixel_fiducial (cv2.typing.MatLike | None): The high-resolution subpixel template of the fiducial marker used
                                                      for refining the center detection. If `None`, subpixel refinement is skipped.
        subpixel_factor (float, optional): A scaling factor used when resizing the cropped image to increase the precision
                                           of subpixel-level matching. Default is 8.

    Returns:
        FiducialDetection: A dictionary containing the following keys:
            - "approx_center" (tuple[float, float]): The approximate (x, y) coordinates of the fiducial marker's center
              based on the initial template matching.
            - "approx_score" (float): The correlation score of the initial template matching, indicating the quality of the match.
            - "subpixel_center" (tuple[float, float] | None): The refined (x, y) coordinates of the fiducial marker's center
              with subpixel accuracy, or `None` if no subpixel fiducial is provided.
            - "subpixel_score" (float | None): The correlation score of the subpixel-level matching, indicating the quality
              of the subpixel refinement, or `None` if no subpixel fiducial is provided.

    Notes:
        - The initial fiducial detection is performed using OpenCV's `cv2.matchTemplate` method with normalized cross-correlation.
        - If a subpixel fiducial template is provided, the function first performs template matching at the approximate location,
          then extracts and resizes the cropped region for subpixel-level refinement using the same matching method.
        - The function assumes the fiducial marker is located within the region where the template matching is performed,
          and that subpixel detection requires a smaller, refined image region.
        - The function will return `None` for the subpixel-related values if no subpixel fiducial is provided.

    Example:
        detection = detect_fiducial(image, fiducial, subpixel_fiducial, subpixel_factor=8)
        print(detection["approx_center"])  # Output: (x, y) of approximate center
        print(detection["subpixel_center"])  # Output: (x, y) of subpixel-level center, or None
        print(detection["approx_score"])    # Output: Score from the initial matching
        print(detection["subpixel_score"])  # Output: Score from subpixel matching, or None

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
    detected_fiducials: dict[str, tuple[float, float]],
    true_fiducials_mm: dict[str, tuple[float, float]],
    scanning_resolution_mm: float = 0.025,
    image_square_dim: int = 10800,
    interpolation_flag: int = cv2.INTER_CUBIC,
    transform_coords: bool = True,
    transform_image: bool = True,
    crop_image: bool = True,
    clahe_enhancement: bool = True,
) -> tuple[cv2.typing.MatLike | None, MetadataImageRestituion]:
    """
    Performs image rectification and enhancement based on detected fiducials and known reference positions.

    This function aligns an image using fiducial markers, applies geometric transformations,
    optionally crops around the principal point, and enhances contrast using CLAHE.

    Args:
        image_path (str): Path to the input image (grayscale).
        detected_fiducials (dict[str, tuple[float, float]]): Coordinates (in pixels) of detected fiducial markers.
        true_fiducials_mm (dict[str, tuple[float, float]]): Ground truth coordinates (in millimeters) of fiducials.
        scanning_resolution_mm (float, optional): Pixel resolution in mm. Defaults to 0.025.
        image_square_dim (int, optional): Target image size for cropping (square). Defaults to 10800.
        interpolation_flag (int, optional): Interpolation method for warping. Defaults to cv2.INTER_CUBIC.
        transform_coords (bool, optional): Whether to compute and apply coordinate transformations. Defaults to True.
        transform_image (bool, optional): Whether to warp the image using the transformation matrix. Defaults to True.
        crop_image (bool, optional): Whether to crop the image around the principal point. Defaults to True.
        clahe_enhancement (bool, optional): Whether to apply CLAHE enhancement. Defaults to True.

    Returns:
        tuple:
            - np.ndarray or None: The final processed image (or None if not processed).
            - dict: Dictionary containing metadata:
                - 'transformation_matrix': Affine transformation matrix (or None),
                - 'fiducials_mm': Detected fiducials converted to mm (or None),
                - 'transformed_fiducials': Transformed fiducial pixel coordinates (or None),
                - 'transformed_fiducials_mm': Transformed fiducials in mm (or None).
    """
    output_image = None
    metadata: MetadataImageRestituion = {}
    # Compute transformation matrix and transform coordinates if requested
    if transform_coords:
        geometric_pp = compute_principal_point_from_valid_segments(true_fiducials_mm)
        metadata["true_fiducials_mm_centered"] = {
            k: (coord[0] - geometric_pp[0], coord[1] - geometric_pp[1]) for k, coord in true_fiducials_mm.items()
        }
        metadata["transformation_matrix"] = estimate_transformation_matrix(
            detected_fiducials,
            metadata["true_fiducials_mm_centered"],
            scanning_resolution_mm,
        )
        metadata["fiducials_mm"] = convert_coordinate_in_camera_reference(detected_fiducials, scanning_resolution_mm)
        metadata["transformed_fiducials"] = {
            key: transform_coord(coord, metadata["transformation_matrix"]) for key, coord in detected_fiducials.items()
        }
        metadata["transformed_fiducials_mm"] = {
            key: transform_coord(coord, metadata["transformation_matrix"])
            for key, coord in metadata["fiducials_mm"].items()
        }

    # Load image if any processing is requested
    if transform_image or crop_image or clahe_enhancement:
        output_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply geometric transformation to the image
        if transform_image:
            height, width = output_image.shape[:2]
            output_image = cv2.warpAffine(
                output_image, metadata["transformation_matrix"][:2], dsize=(width, height), flags=interpolation_flag
            )

        # Crop image around the principal point if requested
        if crop_image:
            center = (
                metadata["transformed_fiducials"]["principal_point"]
                if transform_coords
                else detected_fiducials["principal_point"]
            )
            output_image = hipp.image.crop_image_around_point(output_image, center, image_square_dim)

        # Apply CLAHE enhancement
        if clahe_enhancement:
            output_image = hipp.image.apply_clahe(output_image)
    return output_image, metadata


def estimate_transformation_matrix(
    detected_fiducials: dict[str, tuple[float, float]],
    true_fiducials_mm: dict[str, tuple[float, float]],
    scanning_resolution_mm: float = 0.025,
) -> cv2.typing.MatLike:
    """
    Estimate a 3x3 similarity transformation matrix aligning detected fiducials (in pixels)
    to true fiducials (in mm), excluding the principal point which is already the origin.

    Args:
        detected_fiducials (dict): Detected fiducials in pixel coordinates (may include 'principal_point').
        true_fiducials_mm (dict): Ground truth fiducials in mm (without 'principal_point').
        scanning_resolution_mm (float): Resolution to convert pixels to mm.

    Returns:
        np.ndarray: 3x3 similarity transformation matrix.
    """
    used_keys = [k for k in detected_fiducials if k != "principal_point"]

    if set(used_keys) != set(true_fiducials_mm.keys()):
        raise ValueError(
            f"true_fiducials_mm must have keys: {list(detected_fiducials.keys())}, including 'principal_point'"
        )

    # Convert detected fiducial coordinates to millimeters using the scanning resolution
    detected_fiducials_mm = convert_coordinate_in_camera_reference(detected_fiducials, scanning_resolution_mm)

    detected_points = np.array([detected_fiducials_mm[k] for k in used_keys])
    true_points = np.array([true_fiducials_mm[k] for k in used_keys])

    # Estimate transformation
    transform = SimilarityTransform()
    transform.estimate(detected_points, true_points)

    return cast(cv2.typing.MatLike, transform.params)


def convert_coordinate_in_camera_reference(
    detected_fiducials: dict[str, tuple[float, float]], scanning_resolution_mm: float = 0.025
) -> dict[str, tuple[float, float]]:
    """
    Converts the coordinates of detected fiducials from pixel space to camera reference space
    (in millimeters), using the principal point as the origin. The conversion is performed
    by subtracting the principal point from each detected fiducial and scaling the result
    by the scanning resolution in millimeters.

    The conversion also inverts the Y-axis to align the coordinates with the camera reference system.

    Args:
        detected_fiducials (DetectedFiducials): A dictionary containing detected fiducial points,
                                                 where each key is a fiducial name (e.g.,
                                                 "corner_top_left") and the value is a dictionary
                                                 containing fiducial coordinates.
        scanning_resolution_mm (float, optional): The scanning resolution in millimeters per pixel.
                                                  Defaults to 0.025 mm/pixel.

    Returns:
        dict: A dictionary where each key is a fiducial name and the corresponding value is a tuple
              of (x, y) coordinates in camera reference space, in millimeters.

    Notes:
        - The principal point is computed using the midpoints of valid fiducial pairs, and it
          is used as the origin for the conversion.
        - The Y-axis is inverted after the conversion to match the camera reference system's coordinate system.
    """
    if "principal_point" not in detected_fiducials:
        raise ValueError("'principal_point' need to be in detected_fiducials.")
    # Extraire le point principal
    principal_point = np.array(detected_fiducials["principal_point"])

    # Préparer la sortie
    converted = {}

    for key, coord in detected_fiducials.items():
        delta = np.array(coord) - principal_point
        delta_mm = delta * scanning_resolution_mm
        delta_mm[1] *= -1  # Inversion de l'axe Y
        converted[key] = tuple(delta_mm)

    return converted


def compute_principal_point_from_valid_segments(
    detected_fiducials: DetectedFiducials,
) -> tuple[float, float]:
    """
    Computes the principal point based on the midpoints of valid fiducial marker pairs.
    The principal point is calculated as the average of the midpoints of the following pairs:
    - corner_top_left ↔ corner_bottom_right
    - corner_top_right ↔ corner_bottom_left
    - mid_left ↔ mid_right
    - mid_top ↔ mid_bottom

    Args:
        detected_fiducials (dict): Dictionary with keys as fiducial names and values as (x, y) tuples.

    Returns:
        tuple or None: The computed principal point as a (x, y) tuple, or None if no valid pairs found.
    """
    pairs_keys = [
        ("corner_top_left", "corner_bottom_right"),
        ("corner_top_right", "corner_bottom_left"),
        ("mid_left", "mid_right"),
        ("mid_top", "mid_bottom"),
    ]

    midpoints = []

    for key1, key2 in pairs_keys:
        if key1 in detected_fiducials and key2 in detected_fiducials:
            p1 = detected_fiducials[key1]
            p2 = detected_fiducials[key2]
            if p1 is not None and p2 is not None:
                midpoint = (np.array(p1) + np.array(p2)) / 2
                midpoints.append(midpoint)

    if midpoints:
        return tuple(np.mean(midpoints, axis=0))
    else:
        raise ValueError("Principal point computing error.")


def transform_coord(coord: tuple[float, float], transformation_matrix: cv2.typing.MatLike) -> tuple[float, float]:
    vec = np.append(np.array(coord), 1)  # homogénéisation
    transformed = transformation_matrix @ vec
    return float(transformed[0]), float(transformed[1])
