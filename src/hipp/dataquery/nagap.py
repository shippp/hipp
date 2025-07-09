import os

import pandas as pd

from hipp.dataquery.utils import thread_downloads


def NAGAP_download_images_to_disk(
    image_metadata: pd.DataFrame | str,
    output_directory: str,
    file_name_column: str = "fileName",
    image_type_colum: str = "pid_tiff",
    base_url: str = "https://arcticdata.io/metacat/d1/mn/v2/object/",
    show_progress: bool = True,
    max_workers: int = 5,
    overwrite: bool = False,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    if not isinstance(image_metadata, type(pd.DataFrame())):
        df = pd.read_csv(image_metadata)
    else:
        df = image_metadata

    urls = list(base_url + df[image_type_colum])
    file_names = list(df[file_name_column] + ".tif")

    thread_downloads(output_directory, urls, file_names, show_progress, max_workers, overwrite)
