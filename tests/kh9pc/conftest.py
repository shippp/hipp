"""
Copyright (c) 2026 HIPP developers
Description: some fixture for kh9pc tests
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from hipp.kh9pc.restitution.poly_strategy import PolyStrategy
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector


@dataclass
class JoinedImage:
    """A sample joined-image scan with its reference (left, right, top, bottom) edges."""

    path: Path
    expected_edges: tuple[int, int, int, int]


@pytest.fixture(scope="session")
def joined_image_1() -> JoinedImage:
    """Image to test the vertical detection with a dark right edge, with its reference
    (left, right, top, bottom) edges"""
    return JoinedImage(
        path=Path("/mnt/summer/USERS/DEHECQA/DIVERGENCE/outputs/KH9_PC/work/joined_images/D3C1201-100059F032.tif"),
        expected_edges=(4880, 234065, 945, 22966),  # top/bottom are placeholder values
    )


@pytest.fixture(scope="session")
def joined_image_2() -> JoinedImage:
    """Image to test the vertical detection with the next image at the right side of the film, with its
    reference (left, right, top, bottom) edges"""
    return JoinedImage(
        path=Path("/mnt/summer/USERS/DEHECQA/DIVERGENCE/outputs/KH9_PC/work/joined_images/D3C1204-300473A013.tif"),
        expected_edges=(5610, 235169, 1642, 23823),  # top/bottom are placeholder values
    )


@pytest.fixture(scope="session")
def joined_image_3() -> JoinedImage:
    """Image with a halo on the left side, with its reference (left, right, top, bottom) edges"""
    return JoinedImage(
        path=Path("/mnt/summer/USERS/DEHECQA/DIVERGENCE/outputs/KH9_PC/work/joined_images/D3C1204-300473F013.tif"),
        expected_edges=(3810, 232675, 1494, 23692),  # top/bottom are placeholder values
    )


@pytest.fixture(scope="session", params=["joined_image_1", "joined_image_2", "joined_image_3"])
def joined_image(request: pytest.FixtureRequest) -> JoinedImage:
    """Parametrized indirection over the joined_image_* fixtures, one per sample scan."""
    image: JoinedImage = request.getfixturevalue(request.param)
    return image


@pytest.fixture(scope="session")
def fitted_vertical_detector(joined_image: JoinedImage) -> VerticalDetector:
    """Real VerticalDetector fitted once per joined_image and cached for the session.

    Reused by downstream strategies (PolyStrategy, ...) so the slow edge detection
    runs only once per sample and its actual output feeds the next stage.
    """
    return VerticalDetector().fit(joined_image.path)


@pytest.fixture(scope="session")
def fitted_poly_strategy(fitted_vertical_detector: VerticalDetector, joined_image: JoinedImage) -> PolyStrategy:
    """Real PolyStrategy fitted once per joined_image on top of fitted_vertical_detector.

    Reused by downstream strategies (FiducialStrategy, ...) for the same reason.
    """
    return PolyStrategy(vertical_detector=fitted_vertical_detector).fit(joined_image.path)
