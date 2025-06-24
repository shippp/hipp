"""
Module: test_image.py
Author: godinlu
Date: 29
Description: Description of the module
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

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
    grid_shape = (3, 3)

    block, top_left_coords = hipp.image.read_image_block_grayscale(
        mock_dataset_reader, row_index, col_index, grid_shape
    )

    assert isinstance(block, np.ndarray)

    expected_block_height = mock_dataset_reader.height // grid_shape[0]
    expected_block_width = mock_dataset_reader.width // grid_shape[1]
    assert block.shape == (expected_block_height, expected_block_width)

    x_offset = col_index * expected_block_width
    y_offset = row_index * expected_block_height
    assert top_left_coords == (x_offset, y_offset)


@pytest.fixture  # type: ignore[misc]
def synthetic_image(tmp_path: Any) -> tuple[Path, cv2.typing.MatLike]:
    """Crée une image raster synthétique avec une grille diagonale."""
    width, height = 512, 512
    data = np.indices((height, width)).sum(axis=0).astype(np.uint8)

    transform = from_origin(0, 0, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
    }

    path = tmp_path / "input.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)

    return path, data


def test_warp_image_by_block_vs_cv2(synthetic_image: tuple[Path, cv2.typing.MatLike], tmp_path: Path) -> None:
    input_path, original_data = synthetic_image
    h, w = original_data.shape

    # roation of 10 degree around center
    matrix = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0), [0, 0, 1]])

    output_path = tmp_path / "warped_block.tif"

    hipp.image.warp_tif_blockwise(
        input_path=str(input_path),
        output_path=str(output_path),
        transformation_matrix=matrix,
        output_size=(w, h),
        block_size=128,
        interpolation=cv2.INTER_CUBIC,
        pbar=False,
    )

    # Warp with OpenCV for reference
    warped_cv2 = cv2.warpPerspective(  # type: ignore[call-overload]
        original_data, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    with rasterio.open(output_path) as src:
        warped_block = src.read(1)

    # check size and MAE
    assert warped_block.shape == warped_cv2.shape
    mae = np.mean(np.abs(warped_block.astype(np.float32) - warped_cv2.astype(np.float32)))
    assert mae < 0.001, f"MAE trop élevée: {mae}"
