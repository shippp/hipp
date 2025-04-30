"""
Module: test_fiducials.py
Author: godinlu
Date: 29
Description: Tests function of hipp.aerial.fiducials
"""

import numpy as np

from hipp.aerial.core import create_fiducial_template_from_image  # adapte le chemin


def test_create_fiducial_template_from_image() -> None:
    img = np.ones((200, 200), dtype=np.uint8) * 255
    fiducial_coord = (100, 100)
    cropped = create_fiducial_template_from_image(img, fiducial_coord, 50)
    assert cropped.shape == (100, 100)
