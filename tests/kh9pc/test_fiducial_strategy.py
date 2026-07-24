"""
Copyright (c) 2026 HIPP developers
Description: tests for the class FiducialStrategy
"""

import pytest

from conftest import JoinedImage
from hipp.kh9pc.restitution.fiducial_strategy import FiducialStrategy
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy


@pytest.mark.slow()
def test_fiducial_strategy(fitted_poly_strategy: PolyStrategy, joined_image: JoinedImage) -> None:
    strategy = FiducialStrategy(poly_strategy=fitted_poly_strategy).fit(joined_image.path)

    assert not strategy.is_failed
