"""
Module: fiducials.py
Author: godinlu
Date: 28
Description: functions for aerial fiducials manipulation
"""

import copy
import statistics
from collections import defaultdict
from typing import cast

import cv2
import numpy as np
import rasterio
from skimage.transform import SimilarityTransform

import hipp.image
import hipp.math
from hipp.tools import points_picker
from hipp.typing import FiducialDetection, Fiducials, MetadataImageRestituion


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
) -> tuple[Fiducials, dict[str, float], dict[str, float]]:
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

    return cast(Fiducials, fiducials_detection), scores, subpixel_scores


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
    detected_fiducials: Fiducials,
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
        # here we align the true fiducials coordinate with the geometric principal point, to be in the same reference as detected fiducials
        geometric_pp = compute_principal_point_from_valid_segments(true_fiducials_mm)  # type: ignore[arg-type]
        assert geometric_pp is not None
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
            key: transform_coord(coord, metadata["transformation_matrix"]) if coord is not None else None
            for key, coord in detected_fiducials.items()
        }
        metadata["transformed_fiducials_mm"] = {
            key: transform_coord(coord, metadata["transformation_matrix"]) if coord is not None else None
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
            assert center is not None
            output_image = hipp.image.crop_image_around_point(output_image, center, image_square_dim)

        # Apply CLAHE enhancement
        if clahe_enhancement:
            output_image = hipp.image.apply_clahe(output_image)
    return output_image, metadata


def estimate_transformation_matrix(
    detected_fiducials: Fiducials,
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
    # Exclude 'principal_point' and None values
    used_keys = [k for k, v in detected_fiducials.items() if k != "principal_point" and v is not None]

    if not used_keys:
        raise ValueError("No valid detected fiducials found for transformation estimation.")

    if not all(k in true_fiducials_mm for k in used_keys):
        raise ValueError(
            f"true_fiducials_mm must contain the same keys as detected fiducials (excluding 'principal_point'), "
            f"but only found: {list(true_fiducials_mm.keys())}"
        )

    # Convert valid detected fiducials to mm
    detected_fiducials_mm = convert_coordinate_in_camera_reference(detected_fiducials, scanning_resolution_mm)

    detected_points = np.array([detected_fiducials_mm[k] for k in used_keys])
    true_points = np.array([true_fiducials_mm[k] for k in used_keys])

    if len(detected_points) < 2:
        raise ValueError("At least two valid fiducials are required to estimate the transformation.")

    # Estimate transformation
    transform = SimilarityTransform()
    transform.estimate(detected_points, true_points)

    return cast(cv2.typing.MatLike, transform.params)


def convert_coordinate_in_camera_reference(
    detected_fiducials: Fiducials, scanning_resolution_mm: float = 0.025
) -> Fiducials:
    """
    Converts the coordinates of detected fiducials from pixel space to camera reference space
    (in millimeters), using the principal point as the origin. Ignores fiducials with None values.

    The conversion also inverts the Y-axis to align the coordinates with the camera reference system.

    Args:
        detected_fiducials (dict): A dictionary mapping fiducial names to (x, y) tuples or None.
        scanning_resolution_mm (float): The scanning resolution in millimeters per pixel.

    Returns:
        dict: A dictionary where each key is a fiducial name and the corresponding value is a tuple
              of (x, y) coordinates in camera reference space, in millimeters.
    """
    if "principal_point" not in detected_fiducials:
        raise ValueError("'principal_point' need to be in detected_fiducials.")
    principal_coord = detected_fiducials["principal_point"]
    if principal_coord is None:
        raise ValueError("'principal_point' cannot be None.")

    principal_point = np.array(principal_coord)
    converted: Fiducials = {}

    for key, coord in detected_fiducials.items():
        if coord is None:
            converted[key] = None
        else:
            delta = np.array(coord) - principal_point
            delta_mm = delta * scanning_resolution_mm
            delta_mm[1] *= -1  # Invert Y-axis
            converted[key] = tuple(delta_mm)

    return converted


def compute_principal_point_from_valid_segments(
    detected_fiducials: Fiducials,
) -> tuple[float, float] | None:
    """
    Estimates the principal point of an image based on detected fiducial markers.

    The principal point is computed using both diagonal midpoints and perpendicular offsets
    from adjacent fiducial segments. The algorithm uses two types of fiducial markers:
    - Corners: ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
    - Midsides: ["mid_left", "mid_top", "mid_right", "mid_bottom"]

    For each group (corners and midsides), the following logic is applied:
    1. For each fiducial and its diagonal counterpart (i and (i+2)%4), compute the midpoint if both exist.
    2. For each adjacent pair (i and (i+1)%4), compute the midpoint of the segment and create a point
       perpendicular to the segment direction, offset by half the segment length.

    All valid midpoints and orthogonal points are averaged to return the final principal point estimate.

    Args:
        detected_fiducials (dict): A dictionary mapping fiducial names to (x, y) coordinates or None.

    Returns:
        tuple[float, float] or None: The estimated principal point as an (x, y) tuple,
                                     or None if no valid points were available.
    """
    corners = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
    midsides = ["midside_left", "midside_top", "midside_right", "midside_bottom"]

    orthogonal_points = []
    midpoints = []
    for fiducial_names in [corners, midsides]:
        for i in range(4):
            # Get the points and check they are not None
            p_ortho_1 = detected_fiducials.get(fiducial_names[i])
            p_ortho_2 = detected_fiducials.get(fiducial_names[(i + 1) % 4])
            p_diag_1 = detected_fiducials.get(fiducial_names[i])
            p_diag_2 = detected_fiducials.get(fiducial_names[(i + 2) % 4])

            # Diagonal: compute the midpoint
            if p_diag_1 is not None and p_diag_2 is not None:
                midpoint_diag = (np.array(p_diag_1) + np.array(p_diag_2)) / 2
                midpoints.append(midpoint_diag)

            # Orthogonal: compute the orthogonal point at the center of the adjacent segment
            if p_ortho_1 is not None and p_ortho_2 is not None:
                p1 = np.array(p_ortho_1)
                p2 = np.array(p_ortho_2)
                mid = (p1 + p2) / 2

                # Direction vector of the segment
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    # Unit orthogonal vector
                    perp = np.array([-direction[1], direction[0]]) / norm

                    # Scale the orthogonal offset (here: half the length of the segment)
                    orth_point = mid + perp * (norm / 2)
                    orthogonal_points.append(orth_point)

    # Compute the average of all valid points (diagonals + orthogonals)
    all_points = midpoints + orthogonal_points
    if all_points:
        principal_point = np.mean(all_points, axis=0)
        return float(principal_point[0]), float(principal_point[1])
    else:
        return None


def transform_coord(coord: tuple[float, float], transformation_matrix: cv2.typing.MatLike) -> tuple[float, float]:
    """
    Applies a 2D affine transformation to a single (x, y) coordinate.

    Args:
        coord (tuple[float, float]): The input coordinate to transform, as (x, y).
        transformation_matrix (cv2.typing.MatLike): A 2x3 affine transformation matrix.

    Returns:
        tuple[float, float]: The transformed coordinate as (x', y').
    """
    vec = np.append(np.array(coord), 1)
    transformed = transformation_matrix @ vec
    return float(transformed[0]), float(transformed[1])


def process_fiducials_detection(
    all_detections: dict[str, Fiducials],
    all_scores: dict[str, dict[str, float]],
    all_subpixel_scores: dict[str, dict[str, float]],
    degree_threshold: float = 0.05,
    score_margin: float = 0.1,
) -> dict[str, Fiducials]:
    score_median_by_categ = compute_median_score_by_category(all_scores)
    subpixel_score_median_by_categ = compute_median_score_by_category(all_subpixel_scores)

    score_threshold_by_category = {key: value - score_margin for key, value in score_median_by_categ.items()}
    subpixel_score_threshold_by_category = {
        key: value - score_margin for key, value in subpixel_score_median_by_categ.items()
    }

    result = copy.deepcopy(all_detections)
    for key in all_detections:
        validation_angles = validate_detection_points_with_angle(all_detections[key], degree_threshold)
        validation_score = validate_detection_points_with_matching_score(
            all_scores[key], all_subpixel_scores[key], score_threshold_by_category, subpixel_score_threshold_by_category
        )

        validation = {name: validation_angles[name] or validation_score[name] for name in validation_angles}
        for fiducial_name in validation:
            if not validation[fiducial_name]:
                result[key][fiducial_name] = None

        result[key]["principal_point"] = compute_principal_point_from_valid_segments(result[key])

    return result


def compute_median_score_by_category(all_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    scores_by_category = defaultdict(list)

    for image_scores in all_scores.values():
        for fiducial_name, score in image_scores.items():
            scores_by_category[fiducial_name].append(score)

    median_by_category = {
        fiducial_name: statistics.median(scores) for fiducial_name, scores in scores_by_category.items()
    }
    return median_by_category


def validate_detection_points_with_angle(detection: Fiducials, degree_threshold: float = 0.05) -> dict[str, bool]:
    """
    Evaluate which corner or midside points in a quadrilateral detection are geometrically valid
    based on angle closeness to 90 degrees.

    This function dynamically adapts to whether the input contains corners, midsides, or both.

    Args:
        detection (dict): Dictionary of 4 or more fiducial coordinates (corners and/or midsides).
        degree_threshold (float): Allowed deviation from 90° to consider an angle valid.

    Returns:
        dict[str, bool]: Dictionary mapping each evaluated point name to True (valid) or False (suspect).
    """
    corners = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
    midsides = ["mid_left", "mid_top", "mid_right", "mid_bottom"]

    result: dict[str, bool | None] = {key: None for key in detection if key != "principal_point"}

    # Detect whether to use corners, midsides, or both
    groups_to_check = []
    if all(name in detection and detection[name] is not None for name in corners):
        groups_to_check.append(corners)
    if all(name in detection and detection[name] is not None for name in midsides):
        groups_to_check.append(midsides)

    for group in groups_to_check:
        for i in range(4):
            point_names = [group[(i - 1) % 4], group[i], group[(i + 1) % 4]]
            points = [detection[name] for name in point_names]
            angle = hipp.math.angle_between_three_points(*points)  # type: ignore[arg-type]

            for name in point_names:
                if abs(90 - angle) < degree_threshold:
                    result[name] = True
                elif result[name] is None:
                    result[name] = False
    return cast(dict[str, bool], result)


def validate_detection_points_with_matching_score(
    scores: dict[str, float],
    subpixel_scores: dict[str, float],
    score_threshold_by_category: dict[str, float],
    subpixel_score_threshold_by_category: dict[str, float],
) -> dict[str, bool]:
    result = {}
    for key in scores:
        result[key] = (
            scores[key] > score_threshold_by_category[key]
            and subpixel_scores[key] > subpixel_score_threshold_by_category[key]
        )
    return result
