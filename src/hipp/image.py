"""
Module: image.py
Author: godinlu
Date: 28
Description: some function for the image processing
"""

import cv2


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


def divide_image_into_blocks(image: cv2.typing.MatLike, rows: int, cols: int) -> list[list[cv2.typing.MatLike]]:
    """
    Divides an image into smaller blocks with specified number of rows and columns.
    The last row/column blocks will be smaller if the division isn't even.

    Parameters:
        image (np.ndarray): The input image to be divided.
        rows (int): The number of horizontal blocks (rows).
        cols (int): The number of vertical blocks (columns).

    Returns:
        list[list[np.ndarray]]: A list of lists containing the divided image blocks.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate block height and width
    block_height = height // rows
    block_width = width // cols

    # Initialize the list to store image blocks
    blocks = []

    for i in range(rows):
        row_blocks = []
        for j in range(cols):
            # Calculate coordinates for the block
            top = i * block_height
            left = j * block_width
            bottom = (i + 1) * block_height if i < rows - 1 else height  # Handle last row
            right = (j + 1) * block_width if j < cols - 1 else width  # Handle last column

            # Append the block to the current row
            row_blocks.append(image[top:bottom, left:right])

        # Append the row of blocks to the final list
        blocks.append(row_blocks)

    return blocks
