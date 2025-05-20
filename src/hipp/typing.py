"""
Copyright (c) 2025 HIPP developers
"""

from typing import TypedDict

import cv2


class FiducialDetection(TypedDict):
    approx_center: tuple[float, float]
    approx_score: float
    subpixel_center: tuple[float, float]
    subpixel_score: float


Fiducials = dict[str, tuple[float, float] | None]


class MetadataImageRestituion(TypedDict, total=False):
    transformation_matrix: cv2.typing.MatLike
    fiducials_mm: Fiducials
    transformed_fiducials: Fiducials
    transformed_fiducials_mm: Fiducials
    true_fiducials_mm_centered: dict[str, tuple[float, float]]
