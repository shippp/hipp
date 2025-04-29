"""
Module: test_fiducials.py
Author: godinlu
Date: 29
Description: Tests function of hipp.aerial.fiducials
"""

import os
import numpy as np
import cv2
import tempfile
from hipp.aerial.fiducials import create_fiducial_template  # adapte le chemin


def test_create_fiducial_template_from_given_coordinate() -> None:
    image = np.full((512, 512), 150, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "fake_image.tif")
        output_path = os.path.join(tmpdir, "fiducial_output.tif")

        cv2.imwrite(input_path, image)

        create_fiducial_template(
            image_file=input_path,
            fiducial_coordinate=(256, 256),
            output_file=output_path,
            distance_around_fiducial=50,
        )

        assert os.path.exists(output_path)

        cropped = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        assert cropped.shape == (100, 100)
