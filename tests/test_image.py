"""
Module: test_image.py
Author: godinlu
Date: 29
Description: Description of the module
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

import hipp.image


@pytest.mark.parametrize(
    "image_shape,is_color",
    [  # type: ignore[misc]
        ((256, 256), False),  # Grayscale image
        ((256, 256, 3), True),  # Color image (BGR)
    ],
)
def test_apply_clahe_on_fake_image(image_shape: tuple[int, ...], is_color: bool) -> None:
    image = np.full(image_shape, 128, dtype=np.uint8)
    if is_color:
        image[..., 0] = 100  # B
        image[..., 1] = 120  # G
        image[..., 2] = 140  # R

    result = hipp.image.apply_clahe(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, image)


def test_resize_img() -> None:
    img = np.ones((100, 100), dtype=np.uint8) * 255
    resized = hipp.image.resize_img(img, 4)
    assert resized.shape == (400, 400)


@pytest.fixture  # type: ignore[misc]
def mock_dataset_reader() -> MagicMock:
    # Création d'un mock pour rasterio.DatasetReader
    mock = MagicMock()
    mock.width = 300
    mock.height = 300
    mock.read.return_value = np.ones((100, 100), dtype=np.uint8)  # Bloc d'image mocké (en niveaux de gris)
    return mock


def test_read_image_block_grayscale(mock_dataset_reader) -> None:  # type: ignore[no-untyped-def]
    row_index = 1
    col_index = 1
    grid_size = 3

    block, top_left_coords = hipp.image.read_image_block_grayscale(mock_dataset_reader, row_index, col_index, grid_size)

    assert isinstance(block, np.ndarray)

    expected_block_height = mock_dataset_reader.height // grid_size
    expected_block_width = mock_dataset_reader.width // grid_size
    assert block.shape == (expected_block_height, expected_block_width)

    x_offset = col_index * expected_block_width
    y_offset = row_index * expected_block_height
    assert top_left_coords == (x_offset, y_offset)
