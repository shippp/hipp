"""
Module: test_image.py
Author: godinlu
Date: 29
Description: Description of the module
"""

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


def test_divide_image_into_blocks() -> None:
    image = np.ones((700, 500), dtype=np.uint8) * 255
    blocks = hipp.image.divide_image_into_blocks(image, 3, 3)

    assert len(blocks) == 3
    assert len(blocks[0]) == 3

    assert blocks[0][0].shape == (233, 166)
    assert blocks[2][2].shape == (234, 168)
