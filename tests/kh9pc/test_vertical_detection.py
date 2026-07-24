"""
Copyright (c) 2026 HIPP developers
Description: tests for the class VerticalDetector
"""

import pytest

from conftest import JoinedImage
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector

EDGE_ABS_TOLERANCE = 100
WIDTH_REL_TOLERANCE = 0.02


@pytest.mark.slow()
def test_vertical_detection(fitted_vertical_detector: VerticalDetector, joined_image: JoinedImage) -> None:
    left, right = fitted_vertical_detector.edges_
    expected_left, expected_right, _, _ = joined_image.expected_edges

    expected_width = KH9ImageSpec.from_raster_filepath(joined_image.path).expected_size[0]

    assert left == pytest.approx(expected_left, abs=EDGE_ABS_TOLERANCE)
    assert right == pytest.approx(expected_right, abs=EDGE_ABS_TOLERANCE)
    assert fitted_vertical_detector.detected_width_ == pytest.approx(expected_width, rel=WIDTH_REL_TOLERANCE)
