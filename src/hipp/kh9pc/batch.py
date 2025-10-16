"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    collimation_lines_detection_kwargs: dict[str, Any] | None = None,
    vertical_edges_detection_kwargs: dict[str, Any] | None = None,
    transformation_kwargs: dict[str, Any] | None = None,
    max_workers: int = 4,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Batch processing of raster rectification using collimation line detection.

    This function iterates through all `.tif` rasters in a given directory and applies
    the `collimation_rectification()` function to each file. The process corrects
    optical or geometric distortions in camera images or scanned rasters by detecting
    horizontal and vertical collimation lines, computing a geometric transformation,
    and warping the image.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input raster files (`.tif`) to be rectified.
    output_dir : str or Path
        Directory where rectified rasters will be written. It must exist or be creatable.
    qc_dir : str or Path
        Directory where quality control (QC) plots and diagnostics will be saved.
    collimation_lines_detection_kwargs : dict, optional
        Additional keyword arguments passed to `detect_horizontal_collimation_lines()`.
    vertical_edges_detection_kwargs : dict, optional
        Additional keyword arguments passed to `detect_vertical_edges()`.
    transformation_kwargs : dict, optional
        Additional keyword arguments passed to `compute_transformation()`.
    max_workers : int, optional
        Number of parallel workers to use for image warping. Default is 4.
    overwrite : bool, optional
        If True, existing output rasters will be overwritten. If False (default),
        files that already exist will be skipped.
    verbose : bool, optional
        If True, prints progress messages to the console. Default is True.

    Returns
    -------
    None
        The function writes rectified rasters and QC plots to disk.

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist or is empty.
    RuntimeError
        If any raster rectification fails unexpectedly.
    ValueError
        If no valid raster files are found in the input directory.

    Notes
    -----
    - Only `.tif` files in the input directory are processed.
    - QC plots are saved in structured subdirectories under `qc_dir`.
    - If `overwrite=False`, already processed rasters will be skipped silently.
    - The function calls `collimation_rectification()` internally for each raster.

    Example
    -------
    >>> iter_collimation_rectification(
    ...     input_dir="raw_images/",
    ...     output_dir="rectified_images/",
    ...     qc_dir="qc_results/",
    ...     collimation_lines_detection_kwargs={"sigma": 2.0},
    ...     transformation_kwargs={"method": "affine"},
    ...     max_workers=8,
    ...     overwrite=False,
    ...     verbose=True
    ... )
    Skipping IMG_0001.tif : output already exists
    Collimation rectification for IMG_0002.tif :
        -[1/4] Estimation of collimation lines...
        -[2/4] Detection of vertical lines...
        -[3/4] Warping image (can take some times)...
        -[4/4] Estimation of collimation lines after transformation...
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
                collimation_lines_detection_kwargs,
                vertical_edges_detection_kwargs,
                transformation_kwargs,
                verbose,
                max_workers,
            )
