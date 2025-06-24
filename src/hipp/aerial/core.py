"""
Copyright (c) 2025 HIPP developers
Description: functions for aerial fiducials manipulation
"""

import os
from typing import TypedDict

import cv2
import pandas as pd
import rasterio

import hipp.aerial.quality_control as qc
from hipp.aerial.fiducials import compute_transformation, warp_fiducial_coordinates
from hipp.image import apply_clahe, read_image_block_grayscale, resize_img
from hipp.math import affine_matrix
from hipp.tools import points_picker


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


def detect_fiducials(
    image_path: str,
    corner_fiducial_path: str | None = None,
    midside_fiducial_path: str | None = None,
    subpixel_corner_fiducial_path: str | None = None,
    subpixel_midside_fiducial_path: str | None = None,
    subpixel_factor: float = 8,
    grid_size: int = 3,
    qc_output_path: str | None = None,
) -> dict[str, tuple[float, float] | float | str]:
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")

    # Load fiducial templates once
    corner_fiducial = cv2.imread(corner_fiducial_path, cv2.IMREAD_GRAYSCALE) if corner_fiducial_path else None
    subpixel_corner_fiducial = (
        cv2.imread(subpixel_corner_fiducial_path, cv2.IMREAD_GRAYSCALE) if subpixel_corner_fiducial_path else None
    )
    midside_fiducial = cv2.imread(midside_fiducial_path, cv2.IMREAD_GRAYSCALE) if midside_fiducial_path else None
    subpixel_midside_fiducial = (
        cv2.imread(subpixel_midside_fiducial_path, cv2.IMREAD_GRAYSCALE) if subpixel_midside_fiducial_path else None
    )

    fiducial_map = {
        "corner": (corner_fiducial, subpixel_corner_fiducial),
        "midside": (midside_fiducial, subpixel_midside_fiducial),
    }

    # Prepare block positions
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
        blocs.update(
            {
                "midside_top": (0, grid_size // 2),
                "midside_bottom": (grid_size - 1, grid_size // 2),
                "midside_left": (grid_size // 2, 0),
                "midside_right": (grid_size // 2, grid_size - 1),
            }
        )

    result: dict[str, tuple[float, float] | float | str] = {
        "image_id": os.path.basename(image_path).replace(".tif", "")
    }
    qc_crops = []

    with rasterio.open(image_path) as src:
        for bloc_name, (bloc_row, block_col) in blocs.items():
            block, (offset_x, offset_y) = read_image_block_grayscale(src, bloc_row, block_col, (grid_size, grid_size))

            kind = "corner" if "corner" in bloc_name else "midside"
            fiducial, subpixel_fiducial = fiducial_map[kind]

            # Defensive programming
            if fiducial is None or subpixel_fiducial is None:
                raise ValueError(f"Missing fiducial or subpixel fiducial for {bloc_name}")

            center, score = detect_fiducial(block, fiducial, subpixel_fiducial, subpixel_factor)

            result[f"{bloc_name}_x"] = center[0] + offset_x
            result[f"{bloc_name}_y"] = center[1] + offset_y
            result[f"{bloc_name}_score"] = score

            # For QC image
            if qc_output_path:
                qc_crops.append(qc.create_qc_crop(block, fiducial.shape, center, bloc_name))

    # for qc image
    if qc_output_path and qc_crops:
        grid = qc.concat_images(qc_crops)
        cv2.imwrite(qc_output_path, grid)

    return result


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


class MetadataImageRestituion(TypedDict, total=False):
    transformation_matrix: cv2.typing.MatLike
    rmse_before_transformation: float
    rmse_after_transformation: float


def image_restitution(
    detected_fiducials: pd.Series,
    true_fiducials_mm: pd.Series | dict[str, tuple[float, float]] | None,
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

    **Notes**
    - The function applies transformation and cropping in a single warp if possible, which avoids
      multiple image resampling steps.
    - If no `true_fiducials_mm` is provided, the image is simply cropped using the detected principal point.
    - Fiducials that are not detected (i.e., `None`) are ignored in RMSE computation.
    """

    metadata: MetadataImageRestituion = {}
    if pd.isna(detected_fiducials["principal_point_x"]) or pd.isna(detected_fiducials["principal_point_y"]):
        raise ValueError(f"Can't restitute {detected_fiducials.name} without principal point")

    principal_point = (detected_fiducials["principal_point_x"], detected_fiducials["principal_point_y"])

    # Compute transformation matrix and transform coordinates if requested
    if true_fiducials_mm is not None:
        true_fiducials_mm = pd.Series(true_fiducials_mm)

        img_ref_matrix = affine_matrix(
            scale_x=1 / scanning_resolution_mm,
            scale_y=-1 / scanning_resolution_mm,
            translate_x=principal_point[0],
            translate_y=principal_point[1],
        )
        true_fiducials = warp_fiducial_coordinates(true_fiducials_mm, img_ref_matrix)

        metadata["transformation_matrix"] = compute_transformation(detected_fiducials, true_fiducials)

        transformed_fiducials = warp_fiducial_coordinates(detected_fiducials, metadata["transformation_matrix"])
        principal_point = (transformed_fiducials["principal_point_x"], transformed_fiducials["principal_point_y"])

        # calculate the rmse before and after the transformation
        metadata["rmse_before_transformation"] = qc.compute_rmse(true_fiducials, detected_fiducials)
        metadata["rmse_after_transformation"] = qc.compute_rmse(true_fiducials, transformed_fiducials)

    # cropping block
    if image_square_dim is not None:
        # calculate the transformation matrix for the crop (translation only)
        top_left_x = int(principal_point[0] - image_square_dim // 2)
        top_left_y = int(principal_point[1] - image_square_dim // 2)
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
            output_image = apply_clahe(output_image)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, output_image)

    return metadata
