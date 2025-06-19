"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict

import cv2
import pandas as pd

from hipp.kh9pc.core import image_mosaic, pick_points_in_corners


def join_images(
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


def select_all_cropping_points(
    images_directory: str, csv_file: str, grid_shape: tuple[int, int] = (5, 20), clahe_enhancement: bool = True
) -> None:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        done_images = set(df["image_id"])
        records = df.to_dict("records")
    else:
        done_images = set()
        records = []

    for filename in sorted(os.listdir(images_directory)):
        if not filename.endswith(".tif") or filename.replace(".tif", "") in done_images:
            continue

        image_path = os.path.join(images_directory, filename)
        coords = {"image_id": filename.replace(".tif", "")}
        points = pick_points_in_corners(image_path, grid_shape, clahe_enhancement, False)
        if points is None:
            return

        for key, coord in points.items():
            coords[f"{key}_x"], coords[f"{key}_y"] = coord  # type: ignore [assignment]

        # Ajouter la nouvelle ligne
        records.append(coords)

        # Sauvegarde continue
        df_out = pd.DataFrame(records)
        df_out.to_csv(csv_file, index=False)

    try:
        cv2.destroyWindow("Corner Point Picker")
    except cv2.error:
        pass
