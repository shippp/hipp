"""
Copyright (c) 2025 HIPP developers
Description: all fixtures for the aerial tests
"""

import os
from collections import namedtuple

import pytest

from hipp.aerial.aerial_preprocessing import AerialPreprocessing
from hipp.dataquery import NAGAP_download_images_to_disk

DatasetInfo = namedtuple("DatasetInfo", ["dataset_path", "raw_images"])


@pytest.fixture(scope="session")  # type: ignore[misc]
def dataset() -> DatasetInfo:
    image_metadata = "examples/fiducials_preproc/1994_06_09_aerial_scg.csv"
    dataset_path = os.path.join(os.path.dirname(__file__), "data")
    raw_images = os.path.join(dataset_path, "raw_images")
    if not os.path.exists(raw_images):
        NAGAP_download_images_to_disk(image_metadata, raw_images)

    return DatasetInfo(dataset_path, raw_images)


@pytest.fixture(scope="session")  # type: ignore[misc]
def fiducials(dataset: DatasetInfo) -> str:
    coords_corner = {"fiducial_coordinate": (1115, 381), "subpixel_center_coordinate": (805, 803)}
    coords_midside = {"fiducial_coordinate": (1040, 5881), "subpixel_center_coordinate": (803, 803)}

    preproc = AerialPreprocessing(dataset.raw_images)
    preproc.create_fiducial_template(corner=True, **coords_corner)  # type: ignore [arg-type]
    preproc.create_fiducial_template(midside=True, **coords_midside)  # type: ignore [arg-type]

    return preproc.fiducials_directory
