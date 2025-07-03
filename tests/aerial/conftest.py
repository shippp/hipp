"""
Copyright (c) 2025 HIPP developers
Description: all fixtures for the aerial tests
"""

import glob
import os
from collections import namedtuple

import pytest

from hipp.aerial.core import create_fiducial_templates
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

    first_image_path = sorted(glob.glob(os.path.join(dataset.raw_images, "*.tif")))[0]
    fiducial_directory = os.path.join(dataset.dataset_path, "fiducial_templates")

    create_fiducial_templates(first_image_path, fiducial_directory, corner=True, **coords_corner)  # type: ignore [arg-type]
    create_fiducial_templates(first_image_path, fiducial_directory, midside=True, **coords_midside)  # type: ignore [arg-type]

    return fiducial_directory
