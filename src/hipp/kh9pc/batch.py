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
    verbose: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Apply collimation rectification iteratively to all raster images in a directory.

    This function scans a directory for raster images (GeoTIFF format) and performs
    collimation rectification on each file using the `collimation_rectification()` function.
    It supports batch processing, optional overwriting of existing results, and
    generates quality control (QC) outputs for each image.

    Args:
        input_dir (str | Path):
            Directory containing input raster images to be rectified.
        output_dir (str | Path):
            Directory where rectified raster images will be saved.
        qc_dir (str | Path):
            Directory where quality control plots and diagnostics will be stored.
        bg_px_threshold (int, optional):
            Minimum pixel intensity difference used for vertical edge detection. Defaults to 20.
        collimation_line_dist (int, optional):
            Expected vertical distance (in pixels) between top and bottom collimation lines
            in the rectified image. Defaults to 21770.
        verbose (bool, optional):
            If True, prints progress updates for each processed image. Defaults to True.
        overwrite (bool, optional):
            If True, overwrites existing rectified images in the output directory. Defaults to False.

    Returns:
        None

    Workflow:
        1. Iterate over all `.tif` files in the input directory.
        2. For each file:
            - Skip processing if the corresponding output file already exists (unless `overwrite=True`).
            - Call `collimation_rectification()` to perform geometric rectification.
            - Save all QC plots to the specified `qc_dir`.
        3. Continue until all images are processed.

    Notes:
        - The input directory must contain valid raster images in TIFF format.
        - The function ensures reproducibility by keeping file names consistent across outputs.
        - Useful for batch rectification of satellite or airborne imagery in a processing pipeline.

    Example:
        >>> iter_collimation_rectification(
        ...     input_dir="raw_scenes/",
        ...     output_dir="rectified_scenes/",
        ...     qc_dir="quality_control/",
        ...     bg_px_threshold=25,
        ...     collimation_line_dist=21800,
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
                verbose,
            )
