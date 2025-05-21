"""
Copyright (c) 2025 HIPP developers
"""

import math

import cv2
import numpy as np


def angle_between_three_points(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    """
    Compute the angle in degrees formed at point p2 by the segments p1-p2 and p3-p2.

    Args:
        p1 (Tuple[float, float]): First point (x, y)
        p2 (Tuple[float, float]): Vertex point (x, y) at which the angle is formed
        p3 (Tuple[float, float]): Third point (x, y)

    Returns:
        float: The angle in degrees between the two segments.
    """
    # Vector from p2 to p1
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    # Vector from p2 to p3
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    # Dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 == 0 or mag2 == 0:
        raise ValueError("One of the vectors has zero length")

    # Compute angle in radians
    angle_rad = math.acos(dot / (mag1 * mag2))
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def transform_coord(coord: tuple[float, float], transformation_matrix: cv2.typing.MatLike) -> tuple[float, float]:
    """
    Applies a 2D homogeneous transformation (3x3) to a single (x, y) coordinate.

    Args:
        coord (tuple[float, float]): The input coordinate to transform, as (x, y).
        transformation_matrix (cv2.typing.MatLike): A 3x3 homogeneous transformation matrix.

    Returns:
        tuple[float, float]: The transformed coordinate as (x', y').
    """
    vec = np.array([coord[0], coord[1], 1.0], dtype=np.float32)
    transformed = transformation_matrix @ vec

    # Normalize if homogeneous (third coord != 1)
    if transformed[2] != 0 and transformed[2] != 1:
        transformed /= transformed[2]

    return float(transformed[0]), float(transformed[1])
