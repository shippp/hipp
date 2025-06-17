"""
Copyright (c) 2025 HIPP developers
Description: core functions for the preprocessing of KH-9 PC images
"""

import glob
import os
import subprocess


def image_mosaic(
    image_paths: list[str],
    output_image_path: str,
    overwrite: bool = False,
    threads: int = 0,
    cleanup: bool = True,
    verbose: bool = True,
    dryrun: bool = False,
) -> None:
    if os.path.exists(output_image_path) and not overwrite:
        if verbose:
            print(f"Skipping {output_image_path}: output already exists")
        return

    if verbose:
        print(f"\nMosaicking {output_image_path} with {len(image_paths)} tiles...\n")

    cmd = [
        "image_mosaic",
        *sorted(image_paths),  # sorted for reproducibility
        "--ot",
        "byte",
        "--overlap-width",
        "3000",
        "--threads",
        str(threads),
        "-o",
        output_image_path,
    ]
    if verbose:
        print(" ".join(cmd))

    if not dryrun:
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {output_image_path}: {e}")

    if cleanup:
        for f in glob.glob(f"{output_image_path}-log-image_mosaic-*.txt") + glob.glob(f"{output_image_path}.aux.xml"):
            os.remove(f)
