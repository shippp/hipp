"""
Copyright (c) 2025 HIPP developers
Description: all utils function for the data querying
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests  # type: ignore [import-untyped]
from tqdm import tqdm


def download_file(url: str, output_path: str, overwrite: bool = False) -> None:
    """
    Download a file from a URL and save it to output_path.

    Parameters:
    - url (str): The URL to download.
    - output_path (str): Where to save the downloaded file.
    - overwrite (bool): If False and file exists, skip download.
    """
    if not overwrite and os.path.exists(output_path):
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def thread_downloads(
    output_directory: str,
    urls: list[str],
    file_names: list[str],
    show_progress: bool = False,
    max_workers: int = 5,
    overwrite: bool = False,
) -> None:
    """
    Download multiple files in parallel using ThreadPoolExecutor.

    Parameters:
    - output_directory (str): Path where files will be saved.
    - urls (list of str): List of file URLs to download.
    - file_names (list of str): Corresponding output filenames.
    - show_progress (bool): If True, show a global tqdm progress bar.
    - max_workers (int): Maximum number of parallel threads.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, url, os.path.join(output_directory, filename), overwrite): (url, filename)
            for url, filename in zip(urls, file_names)
        }

        if show_progress:
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
                pass
        else:
            for _ in as_completed(futures):
                pass


def is_optimized_tif(file_path: str) -> bool:
    """
    Check if a GeoTIFF file is already optimized with TILED=YES and COMPRESS=LZW.
    """
    try:
        result = subprocess.run(["gdalinfo", file_path], capture_output=True, text=True, check=True)
        output = result.stdout

        return "Compression=LZW" in output and "Tiled=Yes" in output

    except subprocess.CalledProcessError:
        return False


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
