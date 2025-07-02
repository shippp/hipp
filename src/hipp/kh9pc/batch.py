"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict

import cv2
import pandas as pd

# from hipp.image import warp_tif_blockwise_to_dst
from hipp.image import warp_tif_blockwise
from hipp.kh9pc.core import compute_cropping_matrix, image_mosaic, pick_points_in_corners
from hipp.kh9pc.image_mosaic import compute_sequential_alignment, stitch_images_with_transformations


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


def join_images(images_directory: str, output_directory: str, overwrite: bool = False, verbose: bool = True) -> None:
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
            stitch_images_with_transformations(matrix, output_image_path)


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


def crop_images(
    images_directory: str,
    csv_file: str,
    output_directory: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Crop and rotate .tif images based on coordinates provided in a CSV file.

    For each image in the input directory, this function looks up its corresponding
    cropping points in the CSV file, rotates the image to align the top edge,
    crops it accordingly, and saves the result in the output directory.

    Args:
        images_directory (str): Path to the directory containing input .tif images.
        csv_file (str): Path to the CSV file containing image IDs and cropping coordinates.
                        The CSV must have an 'image_id' index and columns for each corner point:
                        top_left_x, top_left_y, top_right_x, top_right_y, etc.
        output_directory (str): Directory to save the cropped and rotated images.
        overwrite (bool, optional): If False, skip images whose output already exists. Defaults to False.
    """
    # Load the CSV into a DataFrame indexed by image_id
    df = pd.read_csv(csv_file, index_col="image_id")

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(images_directory):
        if filename.endswith(".tif"):
            image_id = filename.replace(".tif", "")
            input_path = os.path.join(images_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Skip if output already exists and overwrite is disabled
            if os.path.exists(output_path) and not overwrite:
                print(f"[{image_id}] Skipped: output already exists at '{output_path}'")
                continue

            # Skip if image_id is not in the CSV
            if image_id not in df.index:
                print(f"[{image_id}] No cropping points found in CSV. Please update '{csv_file}'")
                continue

            # Retrieve the four corner points from the CSV and convert to int
            row = df.loc[image_id]
            points = [
                (int(row["top_left_x"]), int(row["top_left_y"])),
                (int(row["top_right_x"]), int(row["top_right_y"])),
                (int(row["bottom_right_x"]), int(row["bottom_right_y"])),
                (int(row["bottom_left_x"]), int(row["bottom_left_y"])),
            ]

            cropping_matrix, output_size = compute_cropping_matrix(input_path, points)
            # Print cropping info and perform cropping + rotation
            print(f"Image '{image_id}' :")
            print(f"\t- Cropping points : {points}")

            print(f"\t- Output size : {output_size}")
            print(f"\t- Transformation matrix : \n{cropping_matrix}")

            if not dry_run:
                warp_tif_blockwise(
                    input_path, output_path, cropping_matrix, output_size, pbar=True, pbar_desc=f"[{image_id}] warping"
                )
            print(f"\t- Image saved at '{output_path}'\n")
