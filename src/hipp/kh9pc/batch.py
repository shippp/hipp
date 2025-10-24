"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict
from pathlib import Path

# from hipp.image import warp_tif_blockwise_to_dst
from hipp.kh9pc.core import collimation_rectification, image_mosaic
from hipp.kh9pc.image_mosaic import compute_sequential_alignment, mosaic_images


def join_images_asp(
    images_directory: str,
    output_directory: str,
    overwrite: bool = False,
    threads: int = 0,
    cleanup: bool = True,
    verbose: bool = True,
    dryrun: bool = False,
) -> None:
    """
    Groups and mosaics TIF image tiles from a directory by scene ID.

    Each group of images is identified by the prefix before the first underscore in the filename.
    Images must be named in a way that ensures alphabetical ordering corresponds to spatial/temporal logic
    (e.g., img_a.tif, img_b.tif, etc.).

    Parameters:
        images_directory (str): Path to the directory containing .tif image tiles.
        output_directory (str): Path where the output mosaicked images will be saved.
        overwrite (bool): If False and an output file already exists, it will be skipped. Default is False.
        threads (int): Number of threads to use for mosaicking. Default is 0 (auto).
        cleanup (bool): If True, temporary log/auxiliary files will be deleted after mosaicking. Default is True.
        verbose (bool): If True, prints progress and command details. Default is True.
        dryrun (bool): If True, simulates the process without executing commands. Default is False.

    Returns:
        None
    """
    scene_tiles = defaultdict(list)

    # Group image tiles by scene ID (assumed to be the prefix before the first underscore)
    for filename in os.listdir(images_directory):
        if filename.endswith(".tif") and "_" in filename:
            scene_id = filename.split("_")[0]
            scene_tiles[scene_id].append(os.path.join(images_directory, filename))

    # For each scene group, create a mosaicked image
    for scene_id in sorted(scene_tiles):
        output_image_path = os.path.join(output_directory, f"{scene_id}.tif")
        image_paths = sorted(scene_tiles[scene_id])

        # Call image_mosaic for each group
        # Sort image paths alphabetically to ensure consistent mosaicking order
        image_mosaic(image_paths, output_image_path, overwrite, threads, cleanup, verbose, dryrun)


def join_images(
    images_directory: str,
    output_directory: str,
    overwrite: bool = False,
    verbose: bool = True,
    max_workers: int = 5,
) -> None:
    """
    Groups and mosaics TIF image tiles from a directory by scene ID.

    Each group of images is identified by the prefix before the first underscore in the filename.
    Images must be named in a way that ensures alphabetical ordering corresponds to spatial/temporal logic
    (e.g., img_a.tif, img_b.tif, etc.).
    """
    scene_tiles = defaultdict(list)

    # Group image tiles by scene ID (assumed to be the prefix before the first underscore)
    for filename in os.listdir(images_directory):
        if filename.endswith(".tif") and "_" in filename:
            scene_id = filename.split("_")[0]
            scene_tiles[scene_id].append(os.path.join(images_directory, filename))

    # For each scene group, create a mosaicked image
    for scene_id in sorted(scene_tiles):
        output_image_path = os.path.join(output_directory, f"{scene_id}.tif")
        image_paths = sorted(scene_tiles[scene_id])

        if os.path.exists(output_image_path) and not overwrite:
            print(f"Skipping {output_image_path}: output already exists")
        else:
            matrix = compute_sequential_alignment(image_paths, verbose=verbose)
            mosaic_images(matrix, output_image_path, max_workers, verbose)


def iter_collimation_rectification(
    input_dir: str | Path,
    output_dir: str | Path,
    qc_dir: str | Path,
    bg_px_threshold: int = 20,
    collimation_line_dist: int = 21770,
    transformation: str = "tps",
    verbose: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Apply collimation rectification iteratively to all raster images in a directory.

    This function loops over all `.tif` images in the input directory and applies
    the `collimation_rectification()` function to each. The user can choose between
    Thin Plate Spline (TPS) or Affine transformations for geometric correction.
    Quality control (QC) outputs for each image are stored in the specified QC directory.

    Args:
        input_dir (str | Path):
            Directory containing the input raster images to rectify.
        output_dir (str | Path):
            Directory where rectified raster images will be saved.
        qc_dir (str | Path):
            Directory where quality control plots and intermediate data will be stored.
        bg_px_threshold (int, optional):
            Minimum pixel intensity difference used to detect vertical edges. Defaults to 20.
        collimation_line_dist (int, optional):
            Expected distance (in pixels) between the top and bottom collimation lines
            in the rectified image. Defaults to 21770.
        transformation (str, optional):
            Type of geometric transformation to apply.
            - "tps": Thin Plate Spline (non-linear, smooth correction)
            - "affine": Affine (linear correction)
            Defaults to "tps".
        verbose (bool, optional):
            If True, prints progress updates during processing. Defaults to True.
        overwrite (bool, optional):
            If False, skips processing for images that already have a rectified output.
            If True, overwrites existing rectified images. Defaults to False.

    Returns:
        None

    Workflow:
        1. Scan the `input_dir` for all `.tif` files.
        2. For each image:
            a. Check if the output file already exists.
            b. If not (or if `overwrite=True`), perform collimation rectification using
               `collimation_rectification()`.
        3. Store rectified images in `output_dir` and QC data in `qc_dir`.

    Notes:
        - This function is designed for batch rectification of multiple raster scenes.
        - Each image’s intermediate data (collimation lines, grids, QC plots)
          will be organized under its corresponding subdirectories in `qc_dir`.
        - The same transformation type (`transformation`) is applied to all images
          in the batch for consistency.

    Example:
        >>> iter_collimation_rectification(
        ...     input_dir="raw_scenes/",
        ...     output_dir="rectified_scenes/",
        ...     qc_dir="quality_control/",
        ...     bg_px_threshold=25,
        ...     collimation_line_dist=21800,
        ...     transformation="tps",
        ...     verbose=True,
        ...     overwrite=False
        ... )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for input_raster_path in sorted(input_dir.glob("*.tif")):
        output_raster_path = output_dir / input_raster_path.name

        if output_raster_path.exists() and not overwrite:
            if verbose:
                print(f"Skipping {input_raster_path.name} : output already exists")
        else:
            collimation_rectification(
                input_raster_path,
                output_raster_path,
                qc_dir,
                bg_px_threshold,
                collimation_line_dist,
                transformation,
                verbose,
            )
