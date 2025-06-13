"""
Module: test_fiducials.py
Author: godinlu
Date: 29
Description: Tests function of hipp.aerial.fiducials
"""

import math
import os

import cv2
import numpy as np

import hipp.aerial.core as core  # adapte le chemin
from hipp.aerial.aerial_preprocessing import AerialPreprocessing
from hipp.image import read_image_block_grayscale


def assert_distance_from_2_points(
    point1: tuple[int | float, int | float], point2: tuple[int | float, int | float], distance_treshold: float = 5
) -> None:
    distance = math.hypot(point1[0] - point2[0], point1[1] - point2[1])
    assert distance <= distance_treshold, f"Detected point too far from expected: {distance:.2f} pixels"


def test_create_fiducial_template_from_image() -> None:  # type: ignore[no-untyped-def]
    img = np.ones((200, 200), dtype=np.uint8) * 255
    fiducial_coord = (100, 100)
    cropped, coord = core.create_fiducial_template_from_image(img, fiducial_coord, 50)
    assert cropped.shape == (100, 100)


def test_detect_fiducial(dataset, fiducials) -> None:  # type: ignore[no-untyped-def]
    corner_fiducial = cv2.imread(os.path.join(fiducials, "corner_fiducial.png"), cv2.IMREAD_GRAYSCALE)
    subpixel_corner_fiducial = cv2.imread(os.path.join(fiducials, "subpixel_corner_fiducial.png"), cv2.IMREAD_GRAYSCALE)
    image_path = os.path.join(dataset.raw_images, "NAGAP_94V3_196.tif")
    bloc, _ = read_image_block_grayscale(image_path, 0, 0, 5)

    detection = core.detect_fiducial(bloc, corner_fiducial, subpixel_corner_fiducial)

    assert_distance_from_2_points(detection["subpixel_center"], (1108, 385))
    assert detection["approx_score"] >= 0.9


def test_detect_fiducials(dataset, fiducials) -> None:  # type: ignore[no-untyped-def]
    preproc = AerialPreprocessing(dataset.raw_images, fiducials_directory=fiducials)
    fiducial_templates = preproc.load_fiducials_template()
    image_path = os.path.join(dataset.raw_images, "NAGAP_94V3_196.tif")
    detections, scores, _ = core.detect_fiducials(
        image_path,
        **fiducial_templates,
        subpixel_factor=8,
        grid_size=5,
    )
    points = [
        (1108, 385),
        (6607, 259),
        (12106, 332),
        (12232, 5831),
        (12159, 11331),
        (6660, 11458),
        (1161, 11384),
        (1035, 5885),
    ]

    detection_keys = [
        "corner_top_left",
        "midside_top",
        "corner_top_right",
        "midside_right",
        "corner_bottom_right",
        "midside_bottom",
        "corner_bottom_left",
        "midside_left",
    ]
    for key, point in zip(detection_keys, points):
        assert_distance_from_2_points(detections[key], point)  # type: ignore[arg-type]
        assert scores[key] >= 0.7
