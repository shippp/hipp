"""
Copyright (c) 2025 HIPP developers
Description: some function for the image processing
"""

import logging
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

logger = logging.getLogger(__name__)


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


def generate_quickview(
    raster_filepath: str | Path,
    output_path: str | Path | None = None,
    scale_factor: float = 0.2,
    interpolation: int = Resampling.average,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> None:
    raster_filepath = Path(raster_filepath)

    if output_path is None:
        output_path = raster_filepath.parent / "quickviews" / raster_filepath.with_suffix(".jpg").name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        logger.info("Skipping generate_quickview: %s (already exists, overwrite=False)", str(output_path))
        return

    with rasterio.open(raster_filepath) as src:
        width = int(src.width * scale_factor)
        height = int(src.height * scale_factor)
        count = src.count  # number of bands

        # Read all bands and resize in one call
        qv_img = src.read(out_shape=(count, height, width), resampling=interpolation)

    # rasterio reads arrays as (bands, height, width), OpenCV expects (height, width, channels)
    if count == 1:
        img_cv2 = qv_img[0]  # 2D array for single band
    else:
        # transpose (b, H, W) -> (H, W, b) and convert rgb to bgr
        img_cv2 = cv2.cvtColor(np.transpose(qv_img, (1, 2, 0)), cv2.COLOR_RGB2BGR)

    if img_cv2.dtype != np.uint8:
        img_cv2 = img_cv2.astype(np.uint8)

    encode_params = []
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    cv2.imwrite(str(output_path), img_cv2, encode_params)


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


def remap_tif_blockwise(
    input_path: str | Path,
    output_path: str | Path,
    inverse_remap_function: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    output_size: tuple[int, int] | None = None,
    block_size: int = 256,
    interpolation: int = cv2.INTER_CUBIC,
    padding: int = 2,
    lowres_step: int | None = None,
) -> None:
    """
    Apply a geometric remapping on a GeoTIFF file in a memory-efficient, blockwise manner.

    This function divides the output image into small blocks, computes a local inverse
    remapping function for each block, reads only the required input data from the source
    raster, applies a geometric transformation (via OpenCV `cv2.remap`), and writes the
    result to the output GeoTIFF. This approach is optimized for large raster files that
    cannot fit into memory entirely.

    Parameters
    ----------
    input_path : str or Path
        Path to the input GeoTIFF file to be remapped.
    output_path : str or Path
        Destination path for the output GeoTIFF file.
    inverse_remap_function : Callable[[NDArray[np.float32]], NDArray[np.float32]]
        A user-defined function that takes an array of (x, y) coordinates (in float32)
        and returns the corresponding inverse-transformed coordinates of the same shape.
        The mapping is applied in image coordinate space (column, row).
    output_size : tuple[int, int], optional
        Output raster dimensions as (width, height). If None, the input raster dimensions
        are used by default.
    block_size : int, default=256
        Size (in pixels) of each processing block. Increasing this value can improve
        performance but also memory usage.
    interpolation : int, default=cv2.INTER_CUBIC
        OpenCV interpolation flag (e.g., `cv2.INTER_LINEAR`, `cv2.INTER_CUBIC`, `cv2.INTER_NEAREST`).
        Defines how pixel values are interpolated during remapping.
    padding : int, default=2
        Number of extra pixels to read around the computed source window, to reduce
        border artifacts caused by bicubic interpolation.
    lowres_step : int or None, default=None
        Optional subsampling factor for the coordinate grid used during remapping.
        If greater than 1, the remapping grid is computed on a lower resolution
        and interpolated back using `RectBivariateSpline`. This can significantly
        speed up processing but slightly reduce geometric accuracy.

    Notes
    -----
    - The function uses blockwise processing to minimize memory footprint.
    - Source windows that fall entirely outside the input raster are skipped.
    - The output file uses LZW compression and supports large (>4 GB) GeoTIFFs (BIGTIFF=YES).
    - If `lowres_step` > 1, make sure that `block_size` is sufficiently large
      compared to the sampling step to ensure interpolation stability.

    Example
    -------
    >>> import numpy as np
    >>> def inverse_mapping(coords: np.ndarray) -> np.ndarray:
    ...     # Example: simple translation by (dx, dy)
    ...     dx, dy = 10.0, -5.0
    ...     return np.column_stack((coords[:, 0] - dx, coords[:, 1] - dy)).astype(np.float32)
    >>>
    >>> remap_tif_blockwise(
    ...     input_path="input.tif",
    ...     output_path="output_remapped.tif",
    ...     inverse_remap_function=inverse_mapping,
    ...     block_size=512,
    ...     lowres_step=4
    ... )
    """
    with rasterio.open(input_path) as src:
        input_size = (src.width, src.height)
        output_size = output_size if output_size else input_size
        profile = src.profile.copy()
        profile.update(
            {
                "compress": "lzw",
                "BIGTIFF": "YES",
                "width": output_size[0],
                "height": output_size[1],
            }
        )
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            blocks = [
                (dst_x0, dst_y0)
                for dst_x0 in range(0, output_size[0], block_size)
                for dst_y0 in range(0, output_size[1], block_size)
            ]
            n_blocks = len(blocks)
            log_every = max(1, n_blocks // 100000)

            for block_idx, (dst_x0, dst_y0) in enumerate(blocks):
                if block_idx % log_every == 0:
                    logger.debug("  remapping %d%%", block_idx * 100 // n_blocks)
                dst_x1 = min(dst_x0 + block_size, output_size[0])
                dst_y1 = min(dst_y0 + block_size, output_size[1])

                # Full-res grid for this block
                ygrid, xgrid = np.mgrid[dst_y0:dst_y1, dst_x0:dst_x1]

                if lowres_step is not None and lowres_step > 1:
                    # subsample
                    y_idx = list(range(0, ygrid.shape[0], lowres_step))
                    x_idx = list(range(0, xgrid.shape[1], lowres_step))
                    if y_idx[-1] != ygrid.shape[0] - 1:
                        y_idx.append(ygrid.shape[0] - 1)
                    if x_idx[-1] != xgrid.shape[1] - 1:
                        x_idx.append(xgrid.shape[1] - 1)

                    ygrid_lr = ygrid[y_idx][:, x_idx]
                    xgrid_lr = xgrid[y_idx][:, x_idx]

                    points_lr = np.column_stack((xgrid_lr.ravel(), ygrid_lr.ravel()))
                    points_transformed_lr = inverse_remap_function(points_lr.astype(np.float32))

                    # reshape for interpolation
                    tf_x_lr = points_transformed_lr[:, 0].reshape(len(y_idx), len(x_idx))
                    tf_y_lr = points_transformed_lr[:, 1].reshape(len(y_idx), len(x_idx))

                    rbsx = RectBivariateSpline(ygrid_lr[:, 0], xgrid_lr[0, :], tf_x_lr)
                    rbsy = RectBivariateSpline(ygrid_lr[:, 0], xgrid_lr[0, :], tf_y_lr)
                    tf_xgrid = rbsx(ygrid[:, 0], xgrid[0, :]).astype(np.float32)
                    tf_ygrid = rbsy(ygrid[:, 0], xgrid[0, :]).astype(np.float32)
                else:
                    points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
                    points_transformed = inverse_remap_function(points.astype(np.float32))
                    tf_xgrid = points_transformed[:, 0].reshape(xgrid.shape).astype(np.float32)
                    tf_ygrid = points_transformed[:, 1].reshape(ygrid.shape).astype(np.float32)

                # Compute source window bounds with a padding to avoid artefact on edge caused of bicubic interpolation
                src_x0 = int(np.floor(tf_xgrid.min())) - padding
                src_y0 = int(np.floor(tf_ygrid.min())) - padding
                src_x1 = int(np.ceil(tf_xgrid.max())) + padding
                src_y1 = int(np.ceil(tf_ygrid.max())) + padding

                # Rasterio window (row_off, col_off)
                src_window = Window(col_off=src_x0, row_off=src_y0, width=src_x1 - src_x0, height=src_y1 - src_y0)
                src_block = src.read(1, window=src_window)

                if src_block.size == 0:
                    continue
                # Convert global coordinates to local block
                grid_x_local = tf_xgrid - src_x0
                grid_y_local = tf_ygrid - src_y0

                # Remap
                remapped_block = cv2.remap(
                    src_block,
                    grid_x_local,
                    grid_y_local,
                    interpolation=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                # Write destination block
                dst_window = Window(col_off=dst_x0, row_off=dst_y0, width=dst_x1 - dst_x0, height=dst_y1 - dst_y0)
                dst.write(remapped_block, 1, window=dst_window)


def match_multiple_templates(
    image: cv2.typing.MatLike,
    templates: list[cv2.typing.MatLike],
    threshold: float = 0.8,
    method: int = cv2.TM_CCOEFF_NORMED,
    nms_threshold: float = 0.3,
) -> tuple[list[tuple[int, int, int, int]], list[float], list[int]]:
    """
    Match multiple templates and return filtered detections.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    templates : list of np.ndarray
        List of templates.
    threshold : float
        Matching score threshold.
    method : int
        OpenCV template matching method.
    nms_threshold : float
        IoU threshold for NMS.

    Returns
    -------
    boxes : list of tuple
        (x, y, w, h)
    scores : list of float
    template_ids : list of int
        Index of template for each detection
    """

    all_boxes: list[tuple[int, int, int, int]] = []
    all_scores: list[float] = []
    all_template_ids: list[int] = []

    # --- Collect detections ---
    for tid, template in enumerate(templates):
        res = cv2.matchTemplate(image, template, method)
        h, w = template.shape[:2]

        ys, xs = np.where(res >= threshold)

        for x, y in zip(xs, ys):
            all_boxes.append((int(x), int(y), int(w), int(h)))
            all_scores.append(float(res[y, x]))
            all_template_ids.append(tid)

    if not all_boxes:
        return [], [], []

    # --- Apply global NMS ---
    indices = cv2.dnn.NMSBoxes(
        all_boxes,
        all_scores,
        score_threshold=threshold,
        nms_threshold=nms_threshold,
    )

    if len(indices) == 0:
        return [], [], []

    indices_np = np.array(indices).reshape(-1)

    boxes = [all_boxes[i] for i in indices_np]
    scores = [all_scores[i] for i in indices_np]
    template_ids = [all_template_ids[i] for i in indices_np]

    return boxes, scores, template_ids
