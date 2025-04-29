"""
Module: image.py
Author: godinlu
Date: 28
Description: some function for the image processing
"""

import cv2
import numpy as np
import numpy.typing as npt
from typing import cast


def apply_clahe(
    image: npt.NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> npt.NDArray[np.uint8]:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance the contrast of a grayscale or color image.

    Args:
        image (np.ndarray): Input image (grayscale or BGR color) with dtype uint8.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (tuple[int, int], optional): Size of grid for histogram equalization. Defaults to (8, 8).

    Returns:
        np.ndarray: Contrast-enhanced image (same number of channels as input).
    """
    # Create a CLAHE object with given clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        # If the image is grayscale, apply CLAHE directly
        return cast(npt.NDArray[np.uint8], clahe.apply(image))
    else:
        # If the image is color (assumed BGR), apply CLAHE to the L channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert BGR to LAB
        l_channel, a_channel, b_channel = cv2.split(lab)  # Split LAB channels
        l_channel_eq = clahe.apply(
            l_channel
        )  # Apply CLAHE on the L channel (lightness)
        lab_eq = cv2.merge(
            (l_channel_eq, a_channel, b_channel)
        )  # Merge back the modified L with A and B
        return cast(
            npt.NDArray[np.uint8], cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        )  # Convert back LAB to BGR
