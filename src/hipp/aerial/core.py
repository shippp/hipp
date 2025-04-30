"""
Module: fiducials.py
Author: godinlu
Date: 28
Description: functions for aerial fiducials manipulation
"""

import cv2

import hipp.image
from hipp.tools import points_picker


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
) -> dict[str, dict[str, object]]:
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
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")

    splited_image = hipp.image.divide_image_into_blocks(image, grid_size, grid_size)
    block_height, block_width = splited_image[0][0].shape

    results = {}
    positions = {}

    # Define fiducial positions based on available templates
    if corner_fiducial is not None:
        positions.update(
            {
                (0, 0): "top_left_corner",
                (0, grid_size - 1): "top_right_corner",
                (grid_size - 1, grid_size - 1): "bottom_right_corner",
                (grid_size - 1, 0): "bottom_left_corner",
            }
        )
    if midside_fiducial is not None:
        positions.update(
            {
                (0, grid_size // 2): "top_midside",
                (grid_size // 2, grid_size - 1): "right_midside",
                (grid_size - 1, grid_size // 2): "bottom_midside",
                (grid_size // 2, 0): "left_midside",
            }
        )
    for (i, j), label in positions.items():
        template = corner_fiducial if label.endswith("_corner") else midside_fiducial
        assert template is not None  # Ensure mypy knows this can't be None
        h, w = template.shape[:2]

        # Template matching to find fiducial location
        result = cv2.matchTemplate(splited_image[i][j], template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        global_x = j * block_width + max_loc[0]
        global_y = i * block_height + max_loc[1]
        results[label] = {"approx_center": (global_x + w / 2, global_y + h / 2), "approx_score": max_val}

        # Optional subpixel refinement
        if (label.endswith("_corner") and subpixel_corner_fiducial is not None) or (
            label.endswith("_midside") and subpixel_midside_fiducial is not None
        ):
            crop = image[global_y : global_y + h, global_x : global_x + w]
            subpixel_template = subpixel_corner_fiducial if label.endswith("_corner") else subpixel_midside_fiducial
            assert subpixel_template is not None  # for mypy

            crop_center, subpixel_score = subpixel_center(crop, subpixel_template, subpixel_factor)
            fiducial_center = (global_x + crop_center[0], global_y + crop_center[1])
            results[label].update({"subpixel_center": fiducial_center, "subpixel_score": subpixel_score})

    return results


def subpixel_center(
    image: cv2.typing.MatLike, subpixel_template: cv2.typing.MatLike, subpixel_factor: float = 8
) -> tuple[tuple[float, float], float]:
    """
    Computes the subpixel-accurate center of a template match within an image.

    This function upsamples the input image by a given `subpixel_factor` to estimate the
    center of the provided high-resolution template (`subpixel_template`) with subpixel accuracy,
    using normalized cross-correlation via OpenCV's `matchTemplate`.

    Args:
        image: The source image in which to search for the template.
        subpixel_template: The high-resolution template to match against the image.
        subpixel_factor: The upscaling factor used to increase spatial accuracy.

    Returns:
        A tuple containing:
        - (x, y): The subpixel-accurate coordinates of the template's center in the original image scale.
        - score: The maximum correlation score of the template match.
    """
    # Upsample the input image to allow subpixel-accurate matching
    resized_image = hipp.image.resize_img(image, factor=subpixel_factor)

    # Perform template matching using normalized cross-correlation and get the maximum correlation
    result = cv2.matchTemplate(resized_image, subpixel_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    h, w = subpixel_template.shape[:2]

    # Compute the template center in upsampled coordinates and scale it back
    center_x = (max_loc[0] + w / 2) / subpixel_factor
    center_y = (max_loc[1] + h / 2) / subpixel_factor

    return (center_x, center_y), max_val
