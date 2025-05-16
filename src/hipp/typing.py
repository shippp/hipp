from typing import TypedDict

import cv2


class FiducialDetection(TypedDict):
    approx_center: tuple[float, float]
    approx_score: float
    subpixel_center: tuple[float, float]
    subpixel_score: float


class MetadataImageRestituion(TypedDict, total=False):
    transformation_matrix: cv2.typing.MatLike
    fiducials_mm: dict[str, tuple[float, float]]
    transformed_fiducials: dict[str, tuple[float, float]]
    transformed_fiducials_mm: dict[str, tuple[float, float]]
    true_fiducials_mm_centered: dict[str, tuple[float, float]]


DetectedFiducials = dict[str, tuple[float, float]]
