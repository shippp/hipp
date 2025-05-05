from typing import TypedDict


class FiducialDetection(TypedDict):
    approx_center: tuple[float, float]
    approx_score: float
    subpixel_center: tuple[float, float] | None
    subpixel_score: float | None


DetectedFiducials = dict[str, FiducialDetection]
