"""
Copyright (c) 2025 HIPP developers
Description: Generic tools
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import cv2
from tqdm import tqdm

from hipp.image import apply_clahe


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

    return picked_point


def optimize_geotifs(
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


def generate_quickviews(
    directory: str,
    factor: float = 0.2,
    output_directory: str | None = None,
    image_extension: str = ".tif",
    output_image_extension: str = ".jpg",
    max_workers: int = 5,
    overwrite: bool = False,
) -> None:
    """
    Generate downsampled preview images (quickviews) from large GeoTIFF files using `gdal_translate`.

    This function creates JPEG quickviews by resizing input GeoTIFF images based on a scaling factor.
    It runs in parallel using a thread pool to improve performance and disables GDAL auxiliary metadata
    to avoid the creation of `.aux.xml` sidecar files.

    Parameters
    ----------
    directory : str
        Path to the directory containing input GeoTIFF files.
    factor : float, default=0.2
        Downsampling factor (e.g., 0.2 = 20% of the original dimensions).
    output_directory : str or None, optional
        Directory where output quickviews will be saved. If None, a `quickviews` subdirectory is created.
    image_extension : str, default=".tif"
        Extension of the input images to process (case-insensitive).
    output_image_extension : str, default=".jpg"
        Extension/format of the generated quickviews.
    max_workers : int, default=5
        Maximum number of threads used for parallel processing.
    overwrite : bool, default=False
        If False, skip processing files that already have corresponding output files.

    Notes
    -----
    - The function uses `gdal_translate` under the hood with JPEG output format and LZW compression.
    - Auxiliary metadata generation is disabled via the `GDAL_PAM_ENABLED=NO` environment variable.
    - Output images are generated only for files matching the specified input extension.

    Examples
    --------
    >>> generate_quickviews("input_dir", factor=0.1)

    This will create JPEG previews at 10% size for all `.tif` files in "input_dir",
    and store them in "input_dir/quickviews".
    """

    def create_quickview(input_path: str, output_path: str) -> None:
        try:
            command = [
                "gdal_translate",
                "-of",
                "JPEG",
                "-co",
                "QUALITY=85",
                "-outsize",
                f"{int(factor * 100)}%",
                f"{int(factor * 100)}%",
                input_path,
                output_path,
            ]
            os.environ["GDAL_PAM_ENABLED"] = "NO"  # avoid to generate .jpg.aux.xml files
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to process {input_path}: {e}")

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
                tasks.append((input_path, output_path))

    # Run with multithreading and progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_quickview, inp, out) for inp, out in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating quickviews", unit="image"):
            pass
