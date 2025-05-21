"""
Copyright (c) 2025 HIPP developers
"""

from typing import TypedDict


class FiducialDetection(TypedDict):
    approx_center: tuple[float, float]
    approx_score: float
    subpixel_center: tuple[float, float]
    subpixel_score: float


Fiducials = dict[str, tuple[float, float] | None]
