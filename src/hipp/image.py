"""
Module: image.py
Author: godinlu
Date: 28
Description: some function for the image processing
"""

import warnings
from typing import Mapping

import cv2
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window


def apply_clahe(
    image: cv2.typing.MatLike,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> cv2.typing.MatLike:
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
        return clahe.apply(image)
    else:
        # If the image is color (assumed BGR), apply CLAHE to the L channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert BGR to LAB
        l_channel, a_channel, b_channel = cv2.split(lab)  # Split LAB channels
        l_channel_eq = clahe.apply(l_channel)  # Apply CLAHE on the L channel (lightness)
        lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))  # Merge back the modified L with A and B
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)  # Convert back LAB to BGR


def resize_img(
    image: cv2.typing.MatLike, factor: float = 8, interpolation: int = cv2.INTER_CUBIC
) -> cv2.typing.MatLike:
    """
    Resize an image by a specified scaling factor using a given interpolation method.

    Parameters:
        image (np.ndarray): The input image as a NumPy array with dtype uint8.
        factor (float): The scaling factor to resize the image. Defaults to 8.
        interpolation (int): OpenCV interpolation method (e.g., cv2.INTER_CUBIC). Defaults to cv2.INTER_CUBIC.

    Returns:
        np.ndarray: The resized image as a NumPy array.
    """
    # Compute the new width and height based on the scaling factor
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)

    # Resize the image using the specified interpolation method
    resized = cv2.resize(image, (width, height), interpolation=interpolation)

    # Return the resized image
    return resized


def get_corner_blocks(
    image: cv2.typing.MatLike, grid_size: int = 3
) -> tuple[Mapping[str, cv2.typing.MatLike], Mapping[str, tuple[int, int]]]:
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")

    h, w = image.shape[:2]
    block_h = h // grid_size
    block_w = w // grid_size

    # Calculate the top_right corner of each bloc
    coords = {
        "top_left": (0, 0),
        "top_right": (w - block_w, 0),
        "bottom_left": (0, h - block_h),
        "bottom_right": (w - block_w, h - block_h),
    }

    # Extract blocs
    blocks = {
        "top_left": image[:block_h, :block_w],
        "top_right": image[:block_h, -block_w:],
        "bottom_left": image[-block_h:, :block_w],
        "bottom_right": image[-block_h:, -block_w:],
    }

    return blocks, coords


def get_edge_middle_blocks(
    image: cv2.typing.MatLike, grid_size: int = 3
) -> tuple[Mapping[str, cv2.typing.MatLike], Mapping[str, tuple[int, int]]]:
    """
    Retourne les 4 parties situées au centre des bords (haut, bas, gauche, droite)
    en découpant l'image en une grille tile x tile (obligatoirement impair).
    """
    if grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd number.")

    h, w = image.shape[:2]
    block_h = h // grid_size
    block_w = w // grid_size

    center_col = grid_size // 2
    center_row = grid_size // 2

    # Calculate the top_right corner of each bloc
    coords = {
        "top_middle": (block_w * center_col, 0),
        "bottom_middle": (block_w * center_col, h - block_h),
        "left_middle": (0, block_h * center_row),
        "right_middle": (w - block_w, block_h * center_row),
    }

    # Extract blocs
    blocks = {
        "top_middle": image[:block_h, block_w * center_col : block_w * (center_col + 1)],
        "bottom_middle": image[-block_h:, block_w * center_col : block_w * (center_col + 1)],
        "left_middle": image[block_h * center_row : block_h * (center_row + 1), :block_w],
        "right_middle": image[block_h * center_row : block_h * (center_row + 1), -block_w:],
    }

    return blocks, coords


def read_image_block_grayscale(
    dataset_reader: rasterio.io.DatasetReader, row_index: int, col_index: int, grid_size: int = 3
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    """
    Reads a specific block from a grayscale version of a large TIFF image using rasterio.

    Parameters:
        dataset_reader (DatasetReader): An open rasterio DatasetReader object for the image.
        row_index (int): Row index of the desired block (0-based).
        col_index (int): Column index of the desired block (0-based).
        grid_size (int): Number of blocks the image is divided into along each dimension.
                         Must be >= 1. Default is 3 (i.e., a 3x3 grid).

    Returns:
        tuple:
            - block (ndarray): The extracted image block in grayscale (2D numpy array).
            - top_left_coords (tuple): (x, y) pixel coordinates of the top-left corner of the block
              in the full image.

    Notes:
        - Only the first image channel is read to save memory (grayscale).
        - The image does not need to be georeferenced.
        - Blocks at the edges may be slightly larger if the image dimensions are not divisible by grid_size.
    """
    # Suppress warnings about missing georeferencing information
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    block_height = dataset_reader.height // grid_size
    block_width = dataset_reader.width // grid_size

    # Compute top-left coordinates of the target block
    x_offset = col_index * block_width
    y_offset = row_index * block_height

    # Adjust block size for edge blocks (to include all pixels)
    if col_index == grid_size - 1:
        block_width = dataset_reader.width - x_offset
    if row_index == grid_size - 1:
        block_height = dataset_reader.height - y_offset

    # Define the window (region) to read
    window = Window(x_offset, y_offset, block_width, block_height)

    # Read only the first channel (grayscale) from the specified window
    block = dataset_reader.read(1, window=window)

    return block, (x_offset, y_offset)


# def wallis_filter(
#     image: cv2.typing.MatLike, window_size: int, enhance_factor: float = 0.0, mean_balance: float = 0.0
# ) -> cv2.typing.MatLike:
#     """
#     Optimized Wallis Filter for image enhancement using local contrast normalization.

#     Parameters:
#         image : Input grayscale image (uint8)
#         window_size : Size of the local neighborhood window (must be odd)
#         enhance_factor : Regularization to avoid division by small std-dev (A, default = 0.0)
#         mean_balance : Blend between global mean and local mean (B, default = 0.0)

#     Returns:
#         Enhanced image (uint8)
#     """
#     # Ensure float32 precision for processing
#     image = image.astype(np.float32)

#     # Precompute global statistics
#     global_mean = np.mean(image)
#     global_std = np.std(image)

#     # Uniform kernel for local averaging
#     kernel = np.full((window_size, window_size), 1.0 / (window_size**2), dtype=np.float32)

#     # Compute local mean and variance in a single pass
#     local_mean = cv2.filter2D(image, -1, kernel)
#     local_sq_mean = cv2.filter2D(image * image, -1, kernel)
#     local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 1e-6))  # avoids sqrt of negative

#     # Apply the Wallis filter formula
#     contrast_term = global_std * (image - local_mean) / (local_std + enhance_factor)
#     brightness_term = global_mean * mean_balance + local_mean * (1.0 - mean_balance)
#     enhanced = contrast_term + brightness_term

#     # Clip and convert back to uint8
#     return np.clip(enhanced, 0, 255).astype(np.uint8)


# def sobel_filter(image: cv2.typing.MatLike, ksize: int = 3) -> cv2.typing.MatLike:
#     sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     gradient = np.sqrt(sobel_x**2 + sobel_y**2)
#     gradient = np.uint8(np.clip(gradient, 0, 255))
#     return gradient
