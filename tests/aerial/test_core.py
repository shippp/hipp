"""
Module: test_fiducials.py
Author: godinlu
Date: 29
Description: Tests function of hipp.aerial.fiducials
"""

import math
import os

import cv2

import hipp.aerial.core as core  # adapte le chemin
from hipp.image import read_image_block_grayscale


def assert_distance_from_2_points(
    point1: tuple[int | float, int | float], point2: tuple[int | float, int | float], distance_treshold: float = 5
) -> None:
    distance = math.hypot(point1[0] - point2[0], point1[1] - point2[1])
    assert distance <= distance_treshold, f"Detected point too far from expected: {distance:.2f} pixels"


def test_detect_fiducial(dataset, fiducials) -> None:  # type: ignore[no-untyped-def]
    corner_fiducial = cv2.imread(os.path.join(fiducials, "corner_fiducial.png"), cv2.IMREAD_GRAYSCALE)
    subpixel_corner_fiducial = cv2.imread(os.path.join(fiducials, "subpixel_corner_fiducial.png"), cv2.IMREAD_GRAYSCALE)
    image_path = os.path.join(dataset.raw_images, "NAGAP_94V3_196.tif")
    bloc, _ = read_image_block_grayscale(image_path, 0, 0, (5, 5))

    center, score = core.detect_fiducial(bloc, corner_fiducial, subpixel_corner_fiducial)

    assert_distance_from_2_points(center, (1108, 385))
    assert score >= 0.9


def test_detect_fiducials(dataset, fiducials) -> None:  # type: ignore[no-untyped-def]
    image_path = os.path.join(dataset.raw_images, "NAGAP_94V3_196.tif")
    detection = core.detect_fiducials(image_path, fiducials, grid_size=5)
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
        detected_point = (detection[f"{key}_x"], detection[f"{key}_y"])
        assert_distance_from_2_points(detected_point, point)  # type: ignore[arg-type]
        assert detection[f"{key}_score"] >= 0.7  # type: ignore[operator]
