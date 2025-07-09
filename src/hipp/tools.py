"""
Copyright (c) 2025 HIPP developers
Description: Generic tools
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import cv2
import rasterio
from rasterio.shutil import copy as rio_copy
from tqdm import tqdm

from hipp.image import apply_clahe, resize_raster_blockwise


def points_picker(
    image: cv2.typing.MatLike, point_count: int = 1, clahe_enhancement: bool = True
) -> list[tuple[int, int]]:
    """Pick points interactively on a image, only when Ctrl is pressed.

    Args:
        image (np.ndarray): The original large image.
        point_count (int, optional): Number of points to pick. Defaults to 1.
        clahe_enhancement (bool, optional): Apply clahe enhancement. Defaults to True.

    Returns:
        list[tuple[int, int]]: Points in coordinates of the original image.
    """
    picked_points: list[tuple[int, int]] = []
    clone = image.copy()

    if clahe_enhancement:
        clone = apply_clahe(clone)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal picked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if Ctrl is held down (flags & cv2.EVENT_FLAG_CTRLKEY)
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if len(picked_points) < point_count:
                    # Scale back coordinates to the original image size
                    picked_points.append((x, y))

                    # Draw a small circle on the resized image
                    if len(clone.shape) == 2:  # Grayscale image (single channel)
                        cv2.circle(clone, (x, y), 5, (255,), -1)  # Use white circle
                    else:  # RGB image (three channels)
                        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)  # Use green circle

                    # Update the window title
                    title = f"Pick Points with 'Ctrl + Click' ({len(picked_points)}/{point_count} points picked)"
                    cv2.setWindowTitle("Pick Points", title)

                    cv2.imshow("Pick Points", clone)

    cv2.namedWindow("Pick Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Pick Points", clone)
    cv2.setMouseCallback("Pick Points", mouse_callback)

    # Initial window title
    title = f"Pick Points with 'Ctrl + Click' (0/{point_count} points picked)"
    cv2.setWindowTitle("Pick Points", title)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(picked_points) >= point_count:
            break
        if key == ord("q"):
            break
        if cv2.getWindowProperty("Pick Points", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return picked_points


def pick_point_from_image(
    image: cv2.typing.MatLike,
    window_name: str = "Point Pick",
    window_title: str = "Pick a point (Ctrl + Click)",
    destroy_window: bool = False,
) -> tuple[int, int] | None:
    """
    Display a single image and wait for the user to Ctrl+Click on one point.

    Args:
        window_name (str): Name of the persistent OpenCV window.
        image (np.ndarray): Image to display.
        clahe_enhancement (bool): Apply CLAHE enhancement. Default is False.

    Returns:
        (x, y): Coordinates of the picked point in the image, or None if aborted.

    Note:
        The function don't close the window so don't hesitate to close it with the ``
    """
    picked_point = None

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal picked_point
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            picked_point = (x, y)

    # Setup window only once
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.setWindowTitle(window_name, window_title)

    while picked_point is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    if destroy_window:
        cv2.destroyWindow(window_name)

    return picked_point


def optimize_geotifs_gdal(
    geotifs_directory: str, keep: bool = False, max_workers: int = 5, show_progress: bool = True
) -> None:
    """
    Optimize GeoTIFF files in a directory by applying compression and using the BigTIFF format if necessary.
    The original files can be deleted or kept depending on the 'keep' argument.

    :param geotifs_directory: Directory containing the GeoTIFF files to optimize.
    :param keep: If False, the original files will be deleted after optimization.
    :param show_progress: If True, displays a progress bar using tqdm.
    """
    # List of files to process
    files = [f for f in os.listdir(geotifs_directory) if f.endswith(".tif")]

    # Function to optimize a single .tif file
    def optimize_file(filename: str) -> None:
        tif = os.path.join(geotifs_directory, filename)
        tif_optimized = os.path.join(geotifs_directory, f"optimized_{filename}")
        command = [
            "gdal_translate",
            tif,
            tif_optimized,
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "BIGTIFF=IF_SAFER",
        ]

        # Run the gdal_translate command, redirecting output to /dev/null
        with open(os.devnull, "w") as devnull:
            subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

        # If 'keep' is False, remove the original file and rename the optimized one
        if not keep:
            os.remove(tif)
            os.rename(tif_optimized, tif)

    # Use ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit optimization tasks for each file
        futures = {executor.submit(optimize_file, filename): filename for filename in files}

        if show_progress:
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Optimizing", unit="file"):
                pass
        else:
            for _ in as_completed(futures):
                pass


def optimize_geotifs(
    geotifs_directory: str, show_progress: bool = True, overwrite: bool = False, max_workers: int = 5
) -> None:
    """
    Optimize GeoTIFF files in a directory by applying compression and using the BigTIFF format if necessary.
    Files are rewritten only if not already optimized unless 'overwrite' is set to True.

    :param geotifs_directory: Directory containing the GeoTIFF files to optimize.
    :param show_progress: Whether to display a progress bar.
    :param overwrite: If False, skip files already optimized.
    """
    files = [os.path.join(geotifs_directory, f) for f in os.listdir(geotifs_directory) if f.endswith(".tif")]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit optimization tasks for each file
        futures = [executor.submit(optimize_geotif_file, f) for f in files]

        if show_progress:
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Optimizing", unit="file"):
                pass
        else:
            for _ in as_completed(futures):
                pass


def optimize_geotif_file(geotif_file: str, overwrite: bool = False) -> None:
    desired_options = {
        "compress": "lzw",
        "tiled": True,
        "driver": "GTiff",
        "blockxsize": 256,
        "blockysize": 256,
    }
    tmp_tif = geotif_file + ".tmp"

    with rasterio.open(geotif_file) as src:
        profile = src.profile.copy()

        # Check if already optimized
        already_optimized = all(str(profile.get(k, "")).lower() == str(v).lower() for k, v in desired_options.items())
        if already_optimized and not overwrite:
            return
        profile.update(desired_options)
        profile.update({"BIGTIFF": "IF_SAFER"})

        # Write to temporary file
        rio_copy(src, tmp_tif, **profile)

    # Replace or keep original
    os.remove(geotif_file)
    os.rename(tmp_tif, geotif_file)


def generate_quickviews(
    directory: str,
    factor: float = 0.2,
    output_directory: str | None = None,
    image_extension: str = ".tif",
    output_image_extension: str = ".jpg",
    max_workers: int = 5,
    overwrite: bool = False,
) -> None:
    if output_directory is None:
        output_directory = os.path.join(directory, "quickviews")
    os.makedirs(output_directory, exist_ok=True)
    # Build task list
    tasks = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extension.lower()):
            input_path = os.path.join(directory, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_directory, f"{base_name}{output_image_extension}")
            if overwrite or not os.path.exists(output_path):
                tasks.append((input_path, output_path, factor))

    # Run with multithreading and progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_raster_blockwise, inp, out, factor) for inp, out, factor in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating quickviews", unit="image"):
            pass
