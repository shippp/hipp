"""
Copyright (c) 2025 HIPP developers
Description: all utils function for the data querying
"""

import os
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
