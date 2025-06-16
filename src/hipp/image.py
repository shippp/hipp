"""
Copyright (c) 2025 HIPP developers
Description: some function for the image processing
"""

import warnings

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


def read_image_block_grayscale(
    image_source: rasterio.io.DatasetReader | str, row_index: int, col_index: int, grid_size: int = 3
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    """
    Reads a specific block from a grayscale version of a large TIFF image using rasterio.

    Parameters:
        image_source (Union[DatasetReader, str]): Either an open rasterio DatasetReader object
            or the file path to a TIFF image.
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

    # Open the dataset if a path is provided
    if isinstance(image_source, str):
        with rasterio.open(image_source) as dataset:
            return _read_block(dataset, row_index, col_index, grid_size)
    else:
        return _read_block(image_source, row_index, col_index, grid_size)


def _read_block(
    dataset: rasterio.io.DatasetReader, row_index: int, col_index: int, grid_size: int
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    block_height = dataset.height // grid_size
    block_width = dataset.width // grid_size

    # Compute top-left coordinates of the target block
    x_offset = col_index * block_width
    y_offset = row_index * block_height

    # Adjust block size for edge blocks (to include all pixels)
    if col_index == grid_size - 1:
        block_width = dataset.width - x_offset
    if row_index == grid_size - 1:
        block_height = dataset.height - y_offset

    # Define the window (region) to read
    window = Window(x_offset, y_offset, block_width, block_height)

    # Read only the first channel (grayscale) from the specified window
    block = dataset.read(1, window=window)

    return block, (x_offset, y_offset)


def crop_image_around_point(
    img: cv2.typing.MatLike, center: tuple[float, float], image_square_dim: int = 11250
) -> cv2.typing.MatLike:
    h, w = img.shape[:2]
    x_center, y_center = int(round(center[0])), int(round(center[1]))

    half_dim = image_square_dim // 2

    # Compute crop boundaries
    x1 = max(0, x_center - half_dim)
    y1 = max(0, y_center - half_dim)
    x2 = min(w, x_center + half_dim)
    y2 = min(h, y_center + half_dim)

    # Handle border padding if crop goes outside image
    cropped = img[y1:y2, x1:x2]

    # If the crop is smaller than image_square_dim (near border), pad it
    pad_y = image_square_dim - cropped.shape[0]
    pad_x = image_square_dim - cropped.shape[1]

    if pad_x > 0 or pad_y > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            top=pad_y // 2,
            bottom=pad_y - (pad_y // 2),
            left=pad_x // 2,
            right=pad_x - (pad_x // 2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Black padding
        )

    return cropped
