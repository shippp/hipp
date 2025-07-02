"""
Copyright (c) 2025 HIPP developers
Description: Functions to recreate in python the image_mosaic function from ASP
"""

import os

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
from tqdm import tqdm

from hipp.math import transform_coord


####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################
def compute_sequential_alignment(
    images_path: list[str],
    overlap_width: int = 3000,
    bloc_height: int = 256,
    nfeature_per_block: int = 500,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 3,
    verbose: bool = True,
) -> dict[str, cv2.typing.MatLike]:
    """
    Compute sequential geometric alignment transformations between a list of images.

    This function aligns each image in the list to its subsequent image by:
    - Extracting matched keypoints from overlapping image regions in blocks.
    - Transforming matched points to a global coordinate system.
    - Estimating a robust Euclidean transformation using RANSAC to handle outliers.
    - Accumulating transformation matrices relative to the first image.

    Args:
        images_path (list[str]): List of file paths to input images to align sequentially.
        overlap_width (int, optional): Width in pixels of the overlapping area between consecutive images used for keypoint matching. Defaults to 3000.
        bloc_height (int, optional): Height of blocks (in pixels) to split the overlap region for local keypoint detection. Defaults to 256.
        nfeature_per_block (int, optional): Number of ORB features to detect per block. Defaults to 500.
        ransac_max_trials (int, optional): Maximum number of RANSAC iterations for robust transformation estimation. Defaults to 1000.
        ransac_residual_threshold (float, optional): Maximum allowed residual to classify a point as an inlier in RANSAC. Defaults to 3.
        verbose (bool, optional): Whether to print progress and debug information. Defaults to True.

    Returns:
        dict[str, cv2.typing.MatLike]: Dictionary mapping each image path to its cumulative 3x3 homogenous transformation matrix.
            The first image is assigned the identity matrix.
    """
    # Initialize dictionary with identity transformation for the first image (reference)
    transformation_matrixs = {images_path[0]: np.eye(3)}

    # Iterate through consecutive image pairs to compute relative transformations
    for i in range(len(images_path) - 1):
        if verbose:
            print(f"Matching '{images_path[i]}' with '{images_path[i + 1]}' ...")

        # Extract globally matched keypoints from the overlap area between images
        points_a, points_b = extract_global_matches_from_overlap(
            images_path[i], images_path[i + 1], overlap_width, bloc_height, nfeature_per_block
        )
        # Transform matched points from image A to global coordinate system using accumulated transformation
        points_a_tf = [transform_coord(coord, transformation_matrixs[images_path[i]]) for coord in points_a]

        # Estimate robust Euclidean transformation using RANSAC to filter out outliers
        model_robust, inliers = ransac(
            (np.array(points_b, dtype=np.float32), np.array(points_a_tf, dtype=np.float32)),
            EuclideanTransform,
            min_samples=3,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
        )
        if verbose:
            print(f"\t- Number of matching points before versus after ransac : {np.sum(inliers)}/{len(points_a)}")

        # Store cumulative transformation for the next image in the sequence
        transformation_matrixs[images_path[i + 1]] = model_robust.params
    return transformation_matrixs  # type: ignore[return-value]


def stitch_images_with_transformations(
    transformation_matrixs_dict: dict[str, cv2.typing.MatLike], output_tif: str
) -> None:
    """
    Stitch multiple images into a single output raster using provided transformation matrices.

    This function:
    - Determines the output raster size based on the last image and its transformation.
    - Creates an empty output GeoTIFF with appropriate metadata.
    - Warps each input image onto the output raster using its corresponding transformation matrix.
    - Writes the warped data blockwise to efficiently handle large images.

    Args:
        transformation_matrixs_dict (dict[str, cv2.typing.MatLike]):
            Dictionary mapping image file paths to their 3x3 transformation matrices.
            These matrices transform each image into the output coordinate system.
        output_tif (str): File path where the stitched GeoTIFF will be saved.

    Returns:
        None. The result is saved directly to the output_tif file.
    """
    # Get the last image path to determine the output size and metadata
    last_image_path = next(reversed(transformation_matrixs_dict))

    # Open the last image to base the output profile on
    with rasterio.open(last_image_path) as src:
        # Calculate output width by adding translation component from transformation matrix
        width = src.width + int(np.round(transformation_matrixs_dict[last_image_path][0, 2]))
        height = src.height

        # Copy source profile and update parameters for output raster
        profile = src.profile.copy()
        profile.update(
            {
                "width": width,
                "height": height,
                "transform": rasterio.Affine.identity(),
                "compress": "lzw",
                "driver": "GTiff",
                "BIGTIFF": "YES",
                "count": 1,
            }
        )
    # Create and open the output raster for writing warped images
    with rasterio.open(output_tif, "w+", **profile) as dst:
        # Iterate through each image and its transformation matrix
        for image_path, transformation_matrix in transformation_matrixs_dict.items():
            pbar_desc = f"Warping {os.path.basename(image_path)}"
            # Warp the image blockwise and write into the output dataset
            warp_tif_blockwise_to_dst(image_path, dst, transformation_matrix, pbar_desc=pbar_desc)


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def warp_tif_blockwise_to_dst(
    input_path: str,
    dst: rasterio.io.DatasetWriter,
    transformation_matrix: cv2.typing.MatLike,
    block_size: int = 256,
    interpolation: int = cv2.INTER_CUBIC,
    overlap: int = 16,
    pbar: bool = True,
    pbar_desc: str = "Warping blocks",
) -> None:
    """
    Applies a geometric transformation (warping) to a raster image in a memory-efficient,
    block-wise manner, with overlap between blocks to avoid seam artifacts and safe
    in-place writing to the output dataset to prevent overwriting with invalid pixels.

    The function processes the output image in fixed-size blocks, computing the
    corresponding source pixel coordinates for each block using the inverse of the
    provided transformation matrix. Each block is extended by an 'overlap' margin
    on all sides to ensure smooth transitions between adjacent blocks when warping.

    The warped pixels are then remapped from the source to the destination block using
    OpenCV's `remap` function with the specified interpolation method. To handle edges
    properly and avoid invalid pixels overwriting valid data, the function reads the
    existing destination data, and combines it with the warped block pixels using a mask.

    Notes
    -----
    - The function reads blocks from the source raster with a margin ('overlap') to avoid
      artifacts at block edges after warping.
    - The inverse transformation matrix is used to compute source pixel coordinates for
      each destination block's extended region.
    - Pixels mapped outside the source image bounds are filled with the source nodata value.
    - The function combines newly warped pixels with existing destination pixels to avoid
      overwriting valid data with nodata values.
    - This method is designed to be memory efficient for processing large rasters that
      cannot fit entirely into memory.
    """
    out_width, out_height = dst.width, dst.height

    # Compute the inverse transformation matrix to map output coordinates back to source image coordinates.
    M_inv = np.linalg.inv(
        transformation_matrix  # type: ignore[arg-type]
    )[0:2, :]  # Extract first two rows for 2D affine transform usable by cv2.remap

    with rasterio.open(input_path) as src:
        src_dtype = src.dtypes[0]  # Data type of source raster, uint8 for grayscale
        src_nodata = src.nodata if src.nodata is not None else 0  # NoData value, fallback to 0 if undefined

        # Generate a list of blocks covering the entire output raster by stepping through width and height
        blocks = [(x, y) for y in range(0, out_height, block_size) for x in range(0, out_width, block_size)]
        # Wrap blocks with a progress bar if enabled
        iterator = tqdm(blocks, desc=pbar_desc, unit="block") if pbar else blocks

        # Process each output block independently to limit memory use
        for x_out, y_out in iterator:
            # Compute block size in x and y (handle edge blocks smaller than block_size)
            w = min(block_size, out_width - x_out)
            h = min(block_size, out_height - y_out)

            # Extend the block boundaries by 'overlap' pixels on all sides, clipped to image bounds,
            # to avoid edge artifacts when warping and enable smooth blending between blocks
            x_ext = max(0, x_out - overlap)
            y_ext = max(0, y_out - overlap)
            w_ext = min(out_width - x_ext, w + 2 * overlap)
            h_ext = min(out_height - y_ext, h + 2 * overlap)

            # Create meshgrid of pixel coordinates in the extended output block area
            dst_grid_x, dst_grid_y = np.meshgrid(np.arange(x_ext, x_ext + w_ext), np.arange(y_ext, y_ext + h_ext))

            # Prepare homogeneous coordinates (x, y, 1) for transformation
            dst_pts = np.stack([dst_grid_x.ravel(), dst_grid_y.ravel(), np.ones(dst_grid_x.size)], axis=0)

            # Map output coordinates back to source image coordinates using inverse transform
            src_pts = (M_inv @ dst_pts).T

            # Reshape source coordinates to 2D grids matching the extended block size
            x_src = src_pts[:, 0].reshape(h_ext, w_ext).astype(np.float32)
            y_src = src_pts[:, 1].reshape(h_ext, w_ext).astype(np.float32)

            # Determine the bounding box of the source pixels needed to sample for the current block
            x_min = int(np.floor(x_src.min()))
            x_max = int(np.ceil(x_src.max()))
            y_min = int(np.floor(y_src.min()))
            y_max = int(np.ceil(y_src.max()))

            # Skip block if it lies completely outside the source image boundaries
            if x_max < 0 or y_max < 0 or x_min >= src.width or y_min >= src.height:
                continue  # Tout est hors champ

            # Clip the read window to source image bounds with a small margin of 2 pixels
            x_min_clip = max(x_min - 2, 0)
            y_min_clip = max(y_min - 2, 0)
            x_max_clip = min(x_max + 2, src.width - 1)
            y_max_clip = min(y_max + 2, src.height - 1)

            # Define a rasterio Window object to read the required block from source image
            read_window = Window(x_min_clip, y_min_clip, x_max_clip - x_min_clip + 1, y_max_clip - y_min_clip + 1)

            # Read the source block pixels with boundless=True to allow reading outside boundaries if needed,
            # filling missing values with the nodata value.
            src_block = src.read(1, window=read_window, boundless=True, fill_value=src_nodata)

            # Adjust source coordinates to be relative to the read window top-left corner
            x_src_shifted = x_src - x_min_clip
            y_src_shifted = y_src - y_min_clip

            # Warp (remap) the source block pixels to the destination coordinate grid
            warped_ext = cv2.remap(
                src_block,
                x_src_shifted,
                y_src_shifted,
                interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT,  # Use constant border mode to fill out-of-bounds with nodata
                borderValue=src_nodata,  # type: ignore[arg-type]
            )

            # Define core block area inside the extended block by removing overlap margins on edges
            x_start = overlap if x_out - overlap >= 0 else 0
            y_start = overlap if y_out - overlap >= 0 else 0
            x_end = x_start + w
            y_end = y_start + h

            # Extract the central (non-overlapping) region of the warped block to avoid duplication during writing
            warped_core = warped_ext[y_start:y_end, x_start:x_end]

            # Read the existing pixels from destination at the current block location
            existing = dst.read(1, window=Window(x_out, y_out, w, h))

            # Create mask where newly warped pixels are valid (not nodata)
            mask_new = warped_core != src_nodata

            # Combine the warped pixels with existing destination pixels,
            # giving priority to valid new warped pixels to avoid overwriting with black/empty pixels
            combined = np.where(mask_new, warped_core, existing)

            # Write the combined result back to the destination raster at the current block window
            dst.write(combined.astype(src_dtype), 1, window=Window(x_out, y_out, w, h))


def extract_global_matches_from_overlap(
    image_a_path: str,
    image_b_path: str,
    overlap_width: int = 3000,
    bloc_height: int = 256,
    nfeature_per_block: int = 500,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Extracts matched keypoints between the overlapping edge of two georeferenced images,
    by processing them in horizontal blocks. Assumes that image A is on the left and image B is on the right.

    This function is useful to compute global tie points between adjacent raster strips (e.g., satellite or aerial images).
    """
    points_a, points_b = [], []

    with rasterio.open(image_a_path) as src_a, rasterio.open(image_b_path) as src_b:
        width_a = src_a.width
        height_a = src_a.height
        height_b = src_b.height

        # Ensure both images have the same height for block-wise processing
        assert height_a == height_b, "Both images must have the same height for block-wise matching."

        # Iterate over horizontal blocks
        for i in range(0, src_a.height, bloc_height):
            current_block_height = min(bloc_height, height_a - i)

            # Define overlapping windows:
            # - image A: right edge
            # - image B: left edge
            window_a = Window(
                col_off=width_a - overlap_width, row_off=i, width=overlap_width, height=current_block_height
            )
            window_b = Window(col_off=0, row_off=i, width=overlap_width, height=current_block_height)

            # Read corresponding blocks
            block_a = src_a.read(1, window=window_a)
            block_b = src_b.read(1, window=window_b)

            # Match keypoints using ORB
            pts_a, pts_b = match_orb_keypoints(block_a, block_b, nfeatures=nfeature_per_block)

            # Reproject local coordinates to global coordinates
            pts_a_global = [(pt[0] + (width_a - overlap_width), pt[1] + i) for pt in pts_a]
            pts_b_global = [(pt[0], pt[1] + i) for pt in pts_b]

            # Accumulate results
            points_a.extend(pts_a_global)
            points_b.extend(pts_b_global)

    return points_a, points_b


def match_orb_keypoints(
    image_a: cv2.typing.MatLike, image_b: cv2.typing.MatLike, nfeatures: int = 500
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Detect ORB keypoints and return matched coordinates between two grayscale image.
    Returns
    -------
    pts_a : list of tuple[float, float]
        Matched keypoint coordinates from image A.
    pts_b : list of tuple[float, float]
        Matched keypoint coordinates from image B.
    """
    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=nfeatures)  # type: ignore[attr-defined]

    # Detect and compute descriptors
    kp_a, des_a = orb.detectAndCompute(image_a, None)
    kp_b, des_b = orb.detectAndCompute(image_b, None)

    if des_a is None or des_b is None:
        return [], []

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)

    # Sort by match distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched coordinates
    pts_a = [kp_a[m.queryIdx].pt for m in matches]
    pts_b = [kp_b[m.trainIdx].pt for m in matches]

    return pts_a, pts_b
