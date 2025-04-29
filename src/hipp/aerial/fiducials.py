"""
Module: fiducials.py
Author: godinlu
Date: 28
Description: functions for aerial fiducials manipulation
"""

import os
from typing import cast

import cv2
import numpy.typing as npt
import numpy as np

from hipp.tools import points_picker


def create_fiducial_template(
    image_file: str,
    fiducial_coordinate: tuple[int, int] | None = None,
    output_file: str = "fiducial.tif",
    distance_around_fiducial: int = 100,
) -> None:
    """
    Create a fiducial template by cropping a portion of the input image around a fiducial point.

    This function reads an image, optionally picks a fiducial point, and then crops a square region
    around the fiducial point. The cropped image is then saved to the specified output file.

    Args:
        image_file (str): Path to the input image file.
        fiducial_coordinate (tuple[int, int] | None, optional): The coordinate (x, y) of the fiducial point.
            If None, the function will interactively allow the user to pick a point.
        output_file (str, optional): Path to save the cropped image. Defaults to "fiducial.tif".
        distance_around_fiducial (int, optional): The size of the region to crop around the fiducial point,
            in pixels. Defaults to 100.

    Raises:
        FileNotFoundError: If the input image file does not exist.
        ValueError: If no fiducial point is provided and the interactive point picker fails.
    """
    # Check if the input image file exists
    if not os.path.isfile(image_file):
        raise FileNotFoundError(f"The image file '{image_file}' does not exist.")

    # Extract the directory from the output file path
    output_dir = os.path.dirname(output_file)

    # Create the output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load the image in grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load the image from '{image_file}'.")

    # If no fiducial coordinate is provided, allow the user to pick a point interactively
    if fiducial_coordinate is None:
        points = points_picker(
            cast(npt.NDArray[np.uint8], image)
        )  # Assume the first point picked is the fiducial point
        if len(points) == 0:
            raise ValueError("No fiducial point was selected interactively.")
        else:
            fiducial_coordinate = points[0]

    # Extract the fiducial coordinates and calculate the cropping boundaries
    x_L = fiducial_coordinate[0] - distance_around_fiducial
    x_R = fiducial_coordinate[0] + distance_around_fiducial
    y_T = fiducial_coordinate[1] - distance_around_fiducial
    y_B = fiducial_coordinate[1] + distance_around_fiducial

    # Make sure the crop boundaries are within the image dimensions
    x_L = max(0, x_L)
    x_R = min(image.shape[1], x_R)
    y_T = max(0, y_T)
    y_B = min(image.shape[0], y_B)

    # Crop the image using the calculated boundaries
    cropped_image = image[y_T:y_B, x_L:x_R]

    # Save the cropped image to the output file
    cv2.imwrite(output_file, cropped_image)
