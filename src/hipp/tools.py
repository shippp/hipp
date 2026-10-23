"""
Copyright (c) 2025 HIPP developers
Description: Generic tools
"""

import logging
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import rasterio
from rasterio.shutil import copy as rio_copy
from tqdm import tqdm

from hipp.image import apply_clahe, generate_quickview

logger = logging.getLogger(__name__)


def skip_existing(path: str | Path, overwrite: bool) -> bool:
    """Return True, and log it, if *path* exists and *overwrite* is False."""
    path = Path(path)
    if not overwrite and path.exists():
        logger.info("%s - already exists, skipped (use overwrite to redo)", path.name)
        return True
    return False


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Yield a temporary sibling of *path* to write to; it replaces *path* only if the block succeeds.

    Works for files and directories.
    """
    path = Path(path)
    # keep the suffix so rasterio / savefig still infer the format from the name
    tmp = path.with_name(f".tmp-{path.name}")

    # remove the left over if existing
    _remove(tmp)

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        yield tmp
    except BaseException:
        _remove(tmp)
        raise
    if path.is_dir():
        shutil.rmtree(path)
    tmp.replace(path)


def _remove(path: Path) -> None:
    """Delete a file or directory tree if it exists."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def resolve_wildcards(wildcards: list[str], **kwargs: object) -> list[str]:
    """Fill the `{key}` wildcards, a list value giving one result per element."""
    results = list(wildcards)
    for key, value in kwargs.items():
        values = value if isinstance(value, list) else [value]
        results = [wc.replace(f"{{{key}}}", str(v)) for v in values for wc in results]

    unresolved = {key for result in results for key in re.findall(r"\{([^{}]+)\}", result)}
    if unresolved:
        raise ValueError(f"Unresolved wildcards: {', '.join(sorted(unresolved))}")
    # A list value multiplies even the wildcards that do not use its key
    return list(dict.fromkeys(results))


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
        if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY and len(picked_points) < point_count:
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
        futures = [executor.submit(optimize_geotif_file, f, overwrite) for f in files]

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
        desired_options.update({"BIGTIFF": "IF_SAFER"})

        # Write to temporary file
        rio_copy(src, tmp_tif, **desired_options)

    # Replace or keep original
    os.remove(geotif_file)
    os.rename(tmp_tif, geotif_file)


def extract_archive(archive_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> None:
    """Extract a zip or tar archive (any compression) to *output_dir*, skipped if it exists and *overwrite* is False."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    if skip_existing(output_dir, overwrite):
        return

    logger.info("%s - extracting to %s", archive_path.name, output_dir)

    with atomic_output(output_dir) as tmp:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                tf.extractall(tmp, filter="data")
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.name!r}, expected a zip or tar archive")

    logger.info("%s - extracted to %s", archive_path.name, output_dir)


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

    if not tasks:
        return
    # Run with multithreading and progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_quickview, inp, out, factor) for inp, out, factor in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating quickviews", unit="image"):
            pass
