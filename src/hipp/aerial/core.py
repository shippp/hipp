"""
Module: fiducials.py
Author: godinlu
Date: 28
Description: functions for aerial fiducials manipulation
"""

import cv2

import hipp.image
from hipp.tools import points_picker
from hipp.typing import DetectedFiducials, FiducialDetection


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
    image: cv2.typing.MatLike,
    corner_fiducial: cv2.typing.MatLike | None = None,
    midside_fiducial: cv2.typing.MatLike | None = None,
    subpixel_corner_fiducial: cv2.typing.MatLike | None = None,
    subpixel_midside_fiducial: cv2.typing.MatLike | None = None,
    subpixel_factor: float = 8,
    grid_size: int = 3,
) -> DetectedFiducials:
    """
    Detects fiducial markers in a grid-divided image using template matching, with optional subpixel refinement.

    The function divides the input image into a grid of blocks based on the specified grid size,
    and searches for fiducials in predefined locations using template matching with OpenCV.
    Each detected fiducial is assigned a label based on its expected position (e.g., "top_left_corner", "right_midside").

    For each match, the approximate center and match score are computed. If corresponding high-resolution
    subpixel templates are provided, a refinement step is applied by upsampling the matched image region
    and reapplying template matching to obtain a more precise center location.

    Args:
        image: The input image in which to search for fiducials. Must be a valid OpenCV image format.
        corner_fiducial: Template image for detecting corner fiducials. Used in specific grid positions.
        midside_fiducial: Template image for detecting midside fiducials. Used in specific grid positions.
        subpixel_corner_fiducial: High-resolution version of the corner fiducial template for subpixel refinement.
        subpixel_midside_fiducial: High-resolution version of the midside fiducial template for subpixel refinement.
        subpixel_factor: Factor by which to upscale image regions for subpixel template matching.
        grid_size: The number of divisions along one dimension of the image. Must be an odd number
                   to ensure a central block exists.

    Returns:
        A dictionary mapping each detected fiducial label to another dictionary with:
            - "approx_center": Tuple[float, float], the approximate center of the match in image coordinates.
            - "approx_score": float, the template matching confidence score.
            - "subpixel_center": Tuple[float, float], refined center coordinates (if subpixel template is used).
            - "subpixel_score": float, confidence score from subpixel template matching (if applicable).

    Raises:
        ValueError: If the provided grid size is not an odd number.
    """
    result: DetectedFiducials = {}

    # Process the corners fiducials detection
    if corner_fiducial is not None:
        corner_blocs, corner_coordinates = hipp.image.get_corner_blocks(image, grid_size)
        for key in corner_blocs:
            fiducial_detection = detect_fiducial(
                corner_blocs[key], corner_fiducial, subpixel_corner_fiducial, subpixel_factor
            )

            # We translate the coordinate of the sub bloc to the full image
            x0, y0 = fiducial_detection["approx_center"]
            dx, dy = corner_coordinates[key]
            fiducial_detection["approx_center"] = (x0 + dx, y0 + dy)
            if fiducial_detection["subpixel_center"] is not None:
                sx, sy = fiducial_detection["subpixel_center"]
                fiducial_detection["subpixel_center"] = (sx + dx, sy + dy)
            result[f"corner_{key}"] = fiducial_detection

    # Process the midside fiducials detection
    if midside_fiducial is not None:
        edge_blocs, edge_coordinates = hipp.image.get_edge_middle_blocks(image, grid_size)
        for key in edge_blocs:
            fiducial_detection = detect_fiducial(
                edge_blocs[key], midside_fiducial, subpixel_midside_fiducial, subpixel_factor
            )

            # We translate the coordinate of the sub bloc to the full image
            x0, y0 = fiducial_detection["approx_center"]
            dx, dy = edge_coordinates[key]
            fiducial_detection["approx_center"] = (x0 + dx, y0 + dy)
            if fiducial_detection["subpixel_center"] is not None:
                sx, sy = fiducial_detection["subpixel_center"]
                fiducial_detection["subpixel_center"] = (sx + dx, sy + dy)
            result[f"edge_{key}"] = fiducial_detection
    return result


def detect_fiducial(
    image: cv2.typing.MatLike,
    fiducial: cv2.typing.MatLike,
    subpixel_fiducial: cv2.typing.MatLike | None,
    subpixel_factor: float = 8,
) -> FiducialDetection:
    h, w = fiducial.shape[:2]

    matching_template_res = cv2.matchTemplate(image, fiducial, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_template_res)

    if subpixel_fiducial is not None:
        crop = image[max_loc[1] : max_loc[1] + h, max_loc[0] : max_loc[0] + w]
        resized_image = hipp.image.resize_img(crop, factor=subpixel_factor)

        # Perform template matching using normalized cross-correlation and get the maximum correlation
        matching_template_res = cv2.matchTemplate(resized_image, subpixel_fiducial, cv2.TM_CCOEFF_NORMED)
        sub_min_val, sub_max_val, sub_min_loc, sub_max_loc = cv2.minMaxLoc(matching_template_res)

        sub_h, sub_w = subpixel_fiducial.shape[:2]

        # Subpixel center in resized crop, converted to crop coordinates
        sub_center_x = (sub_max_loc[0] + sub_w / 2) / subpixel_factor
        sub_center_y = (sub_max_loc[1] + sub_h / 2) / subpixel_factor

    return {
        "approx_center": (max_loc[0] + w / 2, max_loc[1] + h / 2),
        "approx_score": max_val,
        "subpixel_center": (
            max_loc[0] + sub_center_x,
            max_loc[1] + sub_center_y,
        )
        if subpixel_fiducial is not None
        else None,
        "subpixel_score": sub_max_val if subpixel_fiducial is not None else None,
    }
