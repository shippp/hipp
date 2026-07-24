"""
Copyright (c) 2026 HIPP developers
Description: tests for the class PolyStrategy
"""

import numpy as np
import pytest

from conftest import JoinedImage
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy

ABS_TOLERANCE = 100
REL_TOLERANCE = 0.02


@pytest.mark.slow()
def test_poly_strategy(fitted_poly_strategy: PolyStrategy, joined_image: JoinedImage) -> None:
    left, right, top, bottom = joined_image.expected_edges
    cx = left + (right - left) // 2

    cy_top = fitted_poly_strategy.top_.model.predict(np.array([[cx]])).ravel()[0]
    cy_bottom = fitted_poly_strategy.bottom_.model.predict(np.array([[cx]])).ravel()[0]

    expected_height = KH9ImageSpec.expected_size_from_file(joined_image.path)[1]

    assert not fitted_poly_strategy.is_failed
    assert cy_top == pytest.approx(top, abs=ABS_TOLERANCE)
    assert cy_bottom == pytest.approx(bottom, abs=ABS_TOLERANCE)
    assert cy_bottom - cy_top == pytest.approx(expected_height, rel=REL_TOLERANCE)
