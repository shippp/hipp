"""
Copyright (c) 2025 HIPP developers
"""

import math
from typing import cast

import cv2
import numpy as np
from skimage.transform import AffineTransform, SimilarityTransform


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


def estimate_transformation_matrix(
    src_points: cv2.typing.MatLike, dst_points: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    num_points = len(src_points)
    transform: SimilarityTransform | AffineTransform

    if num_points == 0:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    if num_points == 1:
        src_pt = src_points[0]
        dst_pt = dst_points[0]
        dx, dy = dst_pt[0] - src_pt[0], dst_pt[1] - src_pt[1]
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    elif num_points == 2:
        transform = SimilarityTransform()
    else:
        transform = AffineTransform()
    success = transform.estimate(src_points, dst_points)
    if not success:
        raise RuntimeError("Transformation estimation failed.")

    return cast(cv2.typing.MatLike, transform.params.astype(np.float64))


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


def affine_matrix(
    rotation_deg: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    shear_deg: float = 0.0,
) -> cv2.typing.MatLike:
    """
    Create a 3x3 affine transformation matrix with rotation, scaling, translation, and shear.

    Args:
        rotation_deg (float): Rotation angle in degrees.
        scale_x (float): Scaling factor along the x-axis.
        scale_y (float): Scaling factor along the y-axis.
        translate_x (float): Translation along the x-axis.
        translate_y (float): Translation along the y-axis.
        shear_deg (float): Shear angle in degrees (applied along x).

    Returns:
        np.ndarray: 3x3 affine transformation matrix.
    """
    theta = np.radians(rotation_deg)
    shear = np.radians(shear_deg)

    # Rotation and scale matrix
    a = scale_x * np.cos(theta)
    b = scale_x * np.sin(theta)
    c = -scale_y * np.sin(theta + shear)
    d = scale_y * np.cos(theta + shear)

    # Compose affine transformation
    matrix = np.array([[a, c, translate_x], [b, d, translate_y], [0, 0, 1]], dtype=np.float32)

    return matrix
