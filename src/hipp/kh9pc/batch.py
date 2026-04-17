"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict

from hipp.kh9pc.image_mosaic import compute_sequential_alignments, write_mosaic


def join_images(
    images_directory: str,
    output_directory: str,
    overwrite: bool = False,
    verbose: bool = True,
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
            write_mosaic(compute_sequential_alignments(image_paths), output_image_path)
