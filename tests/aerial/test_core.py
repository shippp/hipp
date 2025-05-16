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
import rasterio

import hipp.aerial.core as core  # adapte le chemin
from hipp.aerial.aerial_preprocessing import AerialPreprocessing
from hipp.image import read_image_block_grayscale


def assert_distance_from_2_points(
    point1: tuple[int | float, int | float], point2: tuple[int | float, int | float], distance_treshold: float = 5
) -> None:
    distance = math.hypot(point1[0] - point2[0], point1[1] - point2[1])
    assert distance <= distance_treshold, f"Detected point too far from expected: {distance:.2f} pixels"


def test_create_fiducial_template_from_image() -> None:
    img = np.ones((200, 200), dtype=np.uint8) * 255
    fiducial_coord = (100, 100)
    cropped = core.create_fiducial_template_from_image(img, fiducial_coord, 50)
    assert cropped.shape == (100, 100)


def test_detect_fiducial_on_1978_09_06_aerial() -> None:
    assert os.path.exists("data/test_images/1978_09_06_aerial/fiducials")
    assert os.path.exists("data/test_images/1978_09_06_aerial/raw_images")

    corner_fiducial = cv2.imread(
        "data/test_images/1978_09_06_aerial/fiducials/corner_fiducial.png", cv2.IMREAD_GRAYSCALE
    )
    subpixel_corner_fiducial = cv2.imread(
        "data/test_images/1978_09_06_aerial/fiducials/subpixel_corner_fiducial.png", cv2.IMREAD_GRAYSCALE
    )
    with rasterio.open("data/test_images/1978_09_06_aerial/raw_images/ARBCSRD00010007.tif") as src:
        bloc, _ = read_image_block_grayscale(src, 0, 0, 5)
    detection = core.detect_fiducial(bloc, corner_fiducial, subpixel_corner_fiducial)

    assert_distance_from_2_points(detection["subpixel_center"], (784, 467))
    assert detection["approx_score"] >= 0.9


def test_detect_fiducials_on_1978_09_06_aerial() -> None:
    assert os.path.exists("data/test_images/1978_09_06_aerial/fiducials")
    assert os.path.exists("data/test_images/1978_09_06_aerial/raw_images")

    preproc = AerialPreprocessing(
        "data/test_images/1978_09_06_aerial/raw_images",
        fiducials_directory="data/test_images/1978_09_06_aerial/fiducials",
    )
    fiducials = preproc.load_fiducials_template()
    detections, scores, _ = core.detect_fiducials(
        "data/test_images/1978_09_06_aerial/raw_images/ARBCSRD00010007.tif", **fiducials, subpixel_factor=8, grid_size=5
    )
    points = [(784, 467), (5046, 297), (9307, 448), (9478, 4709), (9324, 8972), (5063, 9141), (801, 8989), (632, 4728)]

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
        assert_distance_from_2_points(detections[key], point)
        assert scores[key] >= 0.9
