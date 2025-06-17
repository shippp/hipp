"""
Copyright (c) 2025 HIPP developers
Description: Functions for applying core preprocessing functions to images batch
"""

import os
from collections import defaultdict

from hipp.kh9pc.core import image_mosaic


def join_images(
    images_directory: str,
    output_directory: str,
    overwrite: bool = False,
    threads: int = 0,
    cleanup: bool = True,
    verbose: bool = True,
    dryrun: bool = False,
) -> None:
    scene_tiles = defaultdict(list)

    for filename in os.listdir(images_directory):
        if filename.endswith(".tif") and "_" in filename:
            scene_id = filename.split("_")[0]
            scene_tiles[scene_id].append(os.path.join(images_directory, filename))

    for scene_id, image_paths in scene_tiles.items():
        output_image_path = os.path.join(output_directory, f"{scene_id}.tif")
        image_mosaic(image_paths, output_image_path, overwrite, threads, cleanup, verbose, dryrun)
