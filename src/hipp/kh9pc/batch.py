"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict

# from hipp.image import warp_tif_blockwise_to_dst
from hipp.kh9pc.core import image_mosaic_asp
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
        image_mosaic_asp(image_paths, output_image_path, overwrite, threads, cleanup, verbose, dryrun)


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
