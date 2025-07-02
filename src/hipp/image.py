"""
Copyright (c) 2025 HIPP developers
Description: some function for the image processing
"""

import warnings

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


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


def resize_raster_blockwise(
    input_path: str,
    output_path: str,
    scale_factor: float = 0.2,
    block_size: int = 256,
    interpolation: int = cv2.INTER_LINEAR,
    pbar: bool = False,
) -> None:
    """
    Resize a raster image with multiple bands (e.g., RGB) using block-wise processing to limit memory usage,
    and save the result as an image.

    Notes
    -----
    - The function reads and processes the image in blocks to avoid loading the entire image into memory.
    - Supports multi-band images (e.g., RGB with 3 bands).
    - Uses low-level interpolation suitable for visualization (not for high-quality resizing).
    - Converts RGB bands to BGR before saving with OpenCV to ensure correct color representation.
    """
    with rasterio.open(input_path) as src:
        src_width, src_height = src.width, src.height
        src_count = src.count
        # Compute output size
        dst_width = int(src_width * scale_factor)
        dst_height = int(src_height * scale_factor)

        # Prepare an empty array for the output image (single band assumed here)
        dst_img = np.zeros((dst_height, dst_width, src_count), dtype=src.dtypes[0])

        blocks = [(x, y) for y in range(0, dst_height, block_size) for x in range(0, dst_width, block_size)]
        iterator = tqdm(blocks, desc="resizing", unit="block") if pbar else blocks

        for x_out, y_out in iterator:
            w_out = min(block_size, dst_width - x_out)
            h_out = min(block_size, dst_height - y_out)

            x_src_start = int(x_out / scale_factor)
            y_src_start = int(y_out / scale_factor)

            x_src_width = int(w_out / scale_factor)
            y_src_height = int(h_out / scale_factor)

            # read src window
            src_window = Window(x_src_start, y_src_start, x_src_width, y_src_height)

            # Read all bands at once: shape (bands, y, x)
            src_block = src.read(window=src_window)

            # Transpose to (y, x, bands) for cv2.resize
            src_block = np.transpose(src_block, (1, 2, 0))

            resized_block = cv2.resize(
                src_block,
                (w_out, h_out),
                interpolation=interpolation,
            )
            if resized_block.ndim == 2:
                resized_block = resized_block[:, :, np.newaxis]

            dst_img[y_out : y_out + h_out, x_out : x_out + w_out, :] = resized_block

        if dst_img.shape[2] == 3:
            cv2.imwrite(output_path, dst_img[..., ::-1])  # RGB to BGR
        else:
            cv2.imwrite(output_path, dst_img[..., 0])  # write first band


def read_image_block_grayscale(
    image_source: rasterio.io.DatasetReader | str, row_index: int, col_index: int, grid_shape: tuple[int, int] = (3, 3)
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    """
    Reads a specific block from a grayscale version of a large TIFF image using rasterio.

    Parameters:
        image_source (Union[DatasetReader, str]): Either an open rasterio DatasetReader object
            or the file path to a TIFF image.
        row_index (int): Row index of the desired block (0-based).
        col_index (int): Column index of the desired block (0-based).
        grid_shape (tuple[int, int]): Number of blocks in (rows, cols) used to divide the image.
            Must be (>= 1, >= 1). Default is (3, 3).

    Returns:
        tuple:
            - block (ndarray): The extracted image block in grayscale (2D numpy array).
            - top_left_coords (tuple): (x, y) pixel coordinates of the top-left corner of the block
              in the full image.

    Notes:
        - Only the first image channel is read to save memory (grayscale).
        - The image does not need to be georeferenced.
        - Edge blocks may be slightly larger to accommodate image dimensions not divisible by the grid.
    """
    # Open the dataset if a path is provided
    if isinstance(image_source, str):
        with rasterio.open(image_source) as dataset:
            return _read_block(dataset, row_index, col_index, grid_shape)
    else:
        return _read_block(image_source, row_index, col_index, grid_shape)


def _read_block(
    dataset: rasterio.io.DatasetReader, row_index: int, col_index: int, grid_shape: tuple[int, int]
) -> tuple[cv2.typing.MatLike, tuple[int, int]]:
    """
    Internal function to read a block from a raster based on a (rows, cols) grid.
    """
    num_rows, num_cols = grid_shape

    # Compute standard block dimensions
    block_height = dataset.height // num_rows
    block_width = dataset.width // num_cols

    # Compute top-left corner of the block
    x_offset = col_index * block_width
    y_offset = row_index * block_height

    # Adjust size for edge blocks
    if col_index == num_cols - 1:
        block_width = dataset.width - x_offset
    if row_index == num_rows - 1:
        block_height = dataset.height - y_offset

    # Define rasterio window to read
    window = Window(x_offset, y_offset, block_width, block_height)

    # Read only the first band (grayscale)
    block = dataset.read(1, window=window)

    return block, (x_offset, y_offset)


def warp_tif_blockwise(
    input_path: str,
    output_path: str,
    transformation_matrix: cv2.typing.MatLike,
    output_size: tuple[int, int],
    block_size: int = 256,
    interpolation: int = cv2.INTER_CUBIC,
    pbar: bool = True,
    pbar_desc: str = "Warping blocks",
) -> None:
    """
    Applies a geometric transformation (warping) to a raster image in block-wise fashion
    using a provided transformation matrix, and writes the result to a new file.

    Args:
        input_path (str): Path to the input raster image (GeoTIFF).
        output_path (str): Path where the warped image will be saved.
        transformation_matrix (cv2.typing.MatLike): 3x3 homogeneous transformation matrix to apply.
        output_size (tuple[int, int]): Dimensions (width, height) of the output image.
        block_size (int, optional): Size of the processing blocks (in pixels). Defaults to 256.
        interpolation (int, optional): Interpolation method for remapping (e.g., cv2.INTER_LINEAR or cv2.INTER_CUBIC).
        pbar (bool, optional): Whether to display a progress bar with tqdm. Defaults to True.
        pbar_desc (str, optional): Description label for the progress bar. Defaults to "Warping blocks".
    """
    out_width, out_height = output_size

    # Compute the inverse of the transformation matrix for mapping output to input coordinates
    M_inv = np.linalg.inv(transformation_matrix)[0:2, :]  # type: ignore[arg-type]

    with rasterio.open(input_path) as src:
        # Copy the input image profile and update it for the output image
        profile = src.profile.copy()
        profile.update(
            {
                "width": out_width,
                "height": out_height,
                "transform": rasterio.Affine.identity(),  # reset spatial transform, since we're applying a custom one
                "compress": "lzw",
                "driver": "GTiff",
                "BIGTIFF": "YES",  # allow large output files
            }
        )

        # Open output image for writing
        with rasterio.open(output_path, "w", **profile) as dst:
            # Create a list of output block coordinates (top-left corners)
            blocks = [
                (x_out, y_out)
                for y_out in range(0, out_height, block_size)
                for x_out in range(0, out_width, block_size)
            ]
            # Wrap block iterator with a tqdm progress bar if enabled
            iterator = tqdm(blocks, desc=pbar_desc, unit="block") if pbar else blocks

            for x_out, y_out in iterator:
                # Determine the actual block dimensions (handle border cases)
                w = min(block_size, out_width - x_out)
                h = min(block_size, out_height - y_out)

                # Generate grid of destination (output) coordinates
                dst_grid_x, dst_grid_y = np.meshgrid(np.arange(x_out, x_out + w), np.arange(y_out, y_out + h))

                # Stack destination points in homogeneous coordinates (3xN)
                dst_pts = np.stack([dst_grid_x.ravel(), dst_grid_y.ravel(), np.ones(dst_grid_x.size)], axis=0)

                # Apply inverse transformation to get corresponding input coordinates
                src_pts = (M_inv @ dst_pts).T
                x_src = src_pts[:, 0].reshape(h, w).astype(np.float32)
                y_src = src_pts[:, 1].reshape(h, w).astype(np.float32)

                # Compute bounding box of source region to read
                x_min = int(np.floor(x_src.min()))
                y_min = int(np.floor(y_src.min()))
                x_max = int(np.ceil(x_src.max()))
                y_max = int(np.ceil(y_src.max()))

                read_width = x_max - x_min + 1
                read_height = y_max - y_min + 1

                # Skip block if invalid region (completely outside bounds)
                if read_width <= 0 or read_height <= 0:
                    continue

                # Apply padding for better interpolation at block edges
                padding = 2  # safe margin for INTER_CUBIC
                x_pad_min = max(x_min - padding, 0)
                y_pad_min = max(y_min - padding, 0)
                x_pad_max = x_max + padding
                y_pad_max = y_max + padding

                pad_width = x_pad_max - x_pad_min + 1
                pad_height = y_pad_max - y_pad_min + 1

                # Define padded read window in source image
                read_window = Window(x_pad_min, y_pad_min, pad_width, pad_height)

                # Read source data block with boundless mode and padding
                src_block = src.read(1, window=read_window, boundless=True, fill_value=0)

                # Shift coordinates relative to padded window origin
                x_src_shifted = x_src - x_pad_min
                y_src_shifted = y_src - y_pad_min

                # Remap the source block to the warped output using OpenCV
                warped = cv2.remap(  # type: ignore[call-overload]
                    src_block,
                    x_src_shifted,
                    y_src_shifted,
                    interpolation=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                # Write the warped block to the corresponding location in the output image
                dst.write(warped.astype(src.dtypes[0]), 1, window=Window(x_out, y_out, w, h))


def apply_clahe_to_tif_blockwise(
    input_tif_path: str,
    output_tif_path: str,
    block_size: int = 256,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> None:
    """
    Apply CLAHE on a GeoTIFF image block by block and save the result.

    Args:
        input_tif_path (str): Path to input .tif image.
        output_tif_path (str): Path to save the output .tif image.
        block_size (int): Size of the square block/window to process.
        clip_limit (float): CLAHE clip limit parameter.
        tile_grid_size (tuple[int, int]): CLAHE tile grid size.

    """

    # Open source image with rasterio
    with rasterio.open(input_tif_path) as src:
        profile = src.profile.copy()

        # Update profile for output
        profile.update(
            dtype=rasterio.uint8,  # CLAHE output is uint8
            count=src.count,
            compress="lzw",
            bigtiff="TRUE",
        )

        # Create CLAHE object from OpenCV
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        with rasterio.open(output_tif_path, "w", **profile) as dst:
            height = src.height
            width = src.width

            # Read only the first band (grayscale)
            for row_off in range(0, height, block_size):
                for col_off in range(0, width, block_size):
                    # Define window dimensions (may be smaller on edges)
                    win_width = min(block_size, width - col_off)
                    win_height = min(block_size, height - row_off)

                    window = Window(col_off, row_off, win_width, win_height)

                    # Read block from source (as ndarray)
                    block = src.read(1, window=window)

                    # Write result block to destination
                    dst.write(clahe.apply(block), 1, window=window)
