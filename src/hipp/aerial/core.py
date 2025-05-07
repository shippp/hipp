"""
Module: fiducials.py
Author: godinlu
Date: 28
Description: functions for aerial fiducials manipulation
"""

import cv2
import rasterio

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
    image_path: str,
    corner_fiducial: cv2.typing.MatLike | None = None,
    midside_fiducial: cv2.typing.MatLike | None = None,
    subpixel_corner_fiducial: cv2.typing.MatLike | None = None,
    subpixel_midside_fiducial: cv2.typing.MatLike | None = None,
    subpixel_factor: float = 8,
    grid_size: int = 3,
) -> DetectedFiducials:
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
    result = {}
    with rasterio.open(image_path) as src:
        for bloc_name, (bloc_row, block_col) in blocs.items():
            bloc, (offset_x, offset_y) = hipp.image.read_image_block_grayscale(src, bloc_row, block_col, grid_size)

            fiducial = corner_fiducial if "corner" in bloc_name else midside_fiducial
            subpixel_fiducial = subpixel_corner_fiducial if "corner" in bloc_name else subpixel_midside_fiducial
            assert fiducial is not None  # for mypy

            fiducial_detection = detect_fiducial(bloc, fiducial, subpixel_fiducial, subpixel_factor)

            # We translate the coordinate of the sub bloc to the full image
            x0, y0 = fiducial_detection["approx_center"]
            fiducial_detection["approx_center"] = (x0 + offset_x, y0 + offset_y)
            if fiducial_detection["subpixel_center"] is not None:
                sx, sy = fiducial_detection["subpixel_center"]
                fiducial_detection["subpixel_center"] = (sx + offset_x, sy + offset_y)
            result[bloc_name] = fiducial_detection
    return result


def detect_fiducial(
    image: cv2.typing.MatLike,
    fiducial: cv2.typing.MatLike,
    subpixel_fiducial: cv2.typing.MatLike | None,
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
