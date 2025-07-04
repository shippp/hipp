"""
Copyright (c) 2025 HIPP developers

Description:
This module provides utility functions for manipulating aerial fiducial markers,
including center computation, geometric transformation estimation, and coordinate warping.

Fiducial coordinates are represented as a `pd.Series`, where each fiducial key
(from `CORNER_KEYS` or `MIDSIDE_KEYS`) is associated with two fields:
one for the x-coordinate and one for the y-coordinate.
For example: `corner_top_left_x`, `corner_top_left_y`.

Note:
The key orders in `CORNER_KEYS` and `MIDSIDE_KEYS` are assumed to be circular.
This ordering is critical for angle-based geometric calculations and transformations.
"""

import os

import cv2
import numpy as np
import pandas as pd

from hipp.math import angle_between_three_points, estimate_transformation_matrix, transform_coord

CORNER_KEYS = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
MIDSIDE_KEYS = ["midside_left", "midside_top", "midside_right", "midside_bottom"]

CORNER_FIDUCIAL_NAME = "corner_fiducial.png"
MIDSIDE_FIDUCIAL_NAME = "midside_fiducial.png"
SUBPIXEL_CORNER_FIDUCIAL_NAME = "subpixel_" + CORNER_FIDUCIAL_NAME
SUBPIXEL_MIDSIDE_FIDUCIAL_NAME = "subpixel_" + MIDSIDE_FIDUCIAL_NAME


####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################


def compute_principal_point(detection: pd.Series) -> tuple[float, float] | None:
    centers = []

    for keys in _get_groups(detection):
        coords = np.array([(detection[f"{key}_x"], detection[f"{key}_y"]) for key in keys], dtype=np.float32)
        center = _compute_center_square(coords)
        if center is not None:
            centers.append(center)

    if not centers:
        return None

    return tuple(np.mean(centers, axis=0))


def compute_fiducial_transformation(detected_fiducials: pd.Series, true_fiducials: pd.Series) -> cv2.typing.MatLike:
    """
    Compute a geometric transformation matrix between detected and reference fiducial keypoints.

    This function compares the positions of detected fiducial markers with their corresponding
    true/reference positions and estimates a 2D transformation matrix (translation, similarity, or affine),
    depending on the number of valid keypoints.

    The transformation is estimated using:
      - Identity matrix if no points are valid.
      - Pure translation if only one point is valid.
      - Similarity transform if two points are valid.
      - Affine transform if three or more points are valid.

    Keypoints are grouped using `_get_groups()` and their coordinates must be present in both
    `detected_fiducials` and `true_fiducials` under the format `{key}_x`, `{key}_y`.

    Args:
        detected_fiducials (pd.Series): A series containing the detected keypoints' coordinates.
        true_fiducials (pd.Series): A series containing the ground-truth/reference keypoints' coordinates.

    Returns:
        np.ndarray: A 3×3 homogeneous transformation matrix (float64) mapping detected points to true points.

    Raises:
        KeyError: If required keys are missing from the `true_fiducials` input.
        RuntimeError: If the transformation estimation fails (e.g. when enough points are present but
                      cannot produce a valid transformation).
    """
    used_keys = [key for group in _get_groups(detected_fiducials) for key in group]
    suffixed_keys = [suffix for key in used_keys for suffix in (f"{key}_x", f"{key}_y")]

    for key in suffixed_keys:
        if key not in true_fiducials:
            raise KeyError(f"The true fiducials need to have the following keys : {suffixed_keys}")

    src_points = []
    dst_points = []
    for key in used_keys:
        if not pd.isna(detected_fiducials[f"{key}_x"]) and not pd.isna(detected_fiducials[f"{key}_y"]):
            src_points.append((detected_fiducials[f"{key}_x"], detected_fiducials[f"{key}_y"]))
            dst_points.append((true_fiducials[f"{key}_x"], true_fiducials[f"{key}_y"]))

    matrix = estimate_transformation_matrix(np.array(src_points), np.array(dst_points))
    return matrix


def warp_fiducial_coordinates(fiducials: pd.Series, transformation_matrix: cv2.typing.MatLike) -> pd.Series:
    """
    Apply a 2D transformation matrix to all valid fiducial keypoints in a detection.

    This function looks for all keypoint coordinate pairs in the form `{key}_x` and `{key}_y`
    in the input Series, applies the given transformation matrix to non-NaN points,
    and updates their positions accordingly.

    Args:
        fiducials (pd.Series): A Series containing 2D keypoint coordinates with keys like 'corner1_x', 'corner1_y', etc.
        transformation_matrix (np.ndarray): A 3×3 homogeneous transformation matrix to apply to the coordinates.

    Returns:
        pd.Series: A new Series with transformed coordinates (original input is not modified).

    Notes:
        - Keypoints with NaN coordinates are ignored.
        - The input is shallow-copied before modification.
    """
    fiducials = fiducials.copy()

    # Extraire toutes les clés se terminant par _x
    x_keys = [k for k in fiducials.index if k.endswith("_x")]

    for x_key in x_keys:
        y_key = x_key[:-2] + "_y"
        if y_key in fiducials.index:
            x_val, y_val = fiducials[x_key], fiducials[y_key]
            if not pd.isna(x_val) and not pd.isna(y_val):
                x_t, y_t = transform_coord((x_val, y_val), transformation_matrix)
                fiducials[x_key] = x_t
                fiducials[y_key] = y_t

    return fiducials


def filter_scores_by_local_median(df: pd.DataFrame, score_threshold: float = 0.1) -> pd.DataFrame:
    """
    Filter out low-confidence coordinates based on per-key local median score.

    This function identifies all coordinate groups (e.g., "corner_top_left") by
    finding all columns ending with "_score". For each group, it calculates the
    median score and sets the associated x and y coordinates to NaN if their
    score is below (median - score_threshold). After filtering, all "_score"
    columns are removed from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing detection results,
                           with columns named like "<key>_x", "<key>_y", and "<key>_score".
        score_threshold (float): The margin below the median to consider a score as low-confidence.

    Returns:
        pd.DataFrame: A new DataFrame with low-confidence coordinates set to NaN
                      and all "_score" columns removed.
    """
    df = df.copy()
    all_keys = [col.replace("_score", "") for col in df.columns if col.endswith("_score")]

    for key in all_keys:
        score_col = f"{key}_score"
        x_col = f"{key}_x"
        y_col = f"{key}_y"

        median_score = df[score_col].median()
        low_score_mask = df[score_col] < median_score - score_threshold

        df.loc[low_score_mask, [x_col, y_col]] = np.nan

    # Drop all columns that end with '_score'
    score_cols = [col for col in df.columns if col.endswith("_score")]
    df.drop(columns=score_cols, inplace=True)

    return df


def filter_by_angle(df: pd.DataFrame, angle_threshold: float = 0.005) -> pd.DataFrame:
    """
    Filters all detections in a DataFrame based on the angle validity of their keypoints.

    Each row in the DataFrame is processed by `filter_detection_by_angle`, which sets invalid keypoints
    (those forming nearly right angles) to NaN based on the provided angle threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing detections with keypoint coordinates.
        angle_threshold (float): Maximum deviation from 90 degrees (in degrees) to consider a point valid.
                                 Defaults to 0.005.

    Returns:
        pd.DataFrame: A DataFrame with invalid keypoints set to NaN and '_score' columns removed.
    """
    df = df.copy()
    df = df.apply(lambda row: _filter_detection_by_angle(row, angle_threshold=angle_threshold), axis=1)

    # Drop all columns that end with '_score'
    score_cols = [col for col in df.columns if col.endswith("_score")]
    df.drop(columns=score_cols, inplace=True)
    return df


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def _get_groups(detection: pd.Series | pd.DataFrame) -> list[list[str]]:
    if isinstance(detection, pd.Series):
        keys = detection.index
    elif isinstance(detection, pd.DataFrame):
        keys = detection.columns
    else:
        raise TypeError("Expected a pandas Series or DataFrame.")

    result = []
    if all(f"{key}_x" in keys and f"{key}_y" in keys for key in MIDSIDE_KEYS):
        result.append(MIDSIDE_KEYS)
    if all(f"{key}_x" in keys and f"{key}_y" in keys for key in CORNER_KEYS):
        result.append(CORNER_KEYS)
    return result


def _filter_detection_by_angle(
    detection: pd.Series,
    angle_threshold: float = 0.005,
) -> pd.Series:
    """
    Filters keypoints in a single detection based on their geometric angle consistency.

    For each triplet of neighboring keypoints, calculates the angle formed. If the angle
    deviates from 90 degrees by less than the threshold, the keypoints are marked as valid.
    Otherwise, their coordinates are set to NaN.

    Parameters:
        detection (pd.Series): A row from the DataFrame representing a single detection, with keypoint coordinates.
        angle_threshold (float): Maximum deviation from 90 degrees (in degrees) to consider a point valid.
                                 Defaults to 0.005.

    Returns:
        pd.Series: The modified detection with invalid keypoints set to NaN.
    """
    detection = detection.copy()
    result = {}
    for group in _get_groups(detection):
        for i in range(4):
            point_names = [group[(i - 1) % 4], group[i], group[(i + 1) % 4]]
            points = [(detection[f"{name}_x"], detection[f"{name}_y"]) for name in point_names]
            angle = angle_between_three_points(*points)  # type: ignore[arg-type]
            for name in point_names:
                if abs(90 - angle) < angle_threshold:
                    result[name] = True
                elif name not in result:
                    result[name] = False

    for key, valid in result.items():
        if not valid:
            detection[f"{key}_x"] = np.nan
            detection[f"{key}_y"] = np.nan
    return detection


def _compute_center_square(points: cv2.typing.MatLike) -> tuple[float, float] | None:
    """
    Compute the approximate center of a square given up to 4 corner points.
    If some corners are missing (np.nan), the function averages the centers
    of the diagonals that can be computed.

    Args:
        points (Sequence[tuple[float, float]]): List of 4 points in the following order:
            [top_left, top_right, bottom_right, bottom_left]

    Returns:
        tuple[float, float] | None: Estimated center of the square, or None if not computable.
    """
    assert len(points) == 4, "Expected 4 points: top_left, top_right, bottom_right, bottom_left"

    diagonals = [(points[0], points[2]), (points[1], points[3])]
    centers = [
        ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        for p1, p2 in diagonals
        if not (np.isnan(p1).any() or np.isnan(p2).any())
    ]

    if not centers:
        return None

    return tuple(np.mean(centers, axis=0))


def _get_fiducial_template_paths(fiducials_directory: str) -> dict[str, str]:
    paths = {
        "corner_fiducial_path": os.path.join(fiducials_directory, CORNER_FIDUCIAL_NAME),
        "midside_fiducial_path": os.path.join(fiducials_directory, MIDSIDE_FIDUCIAL_NAME),
        "subpixel_corner_fiducial_path": os.path.join(fiducials_directory, SUBPIXEL_CORNER_FIDUCIAL_NAME),
        "subpixel_midside_fiducial_path": os.path.join(fiducials_directory, SUBPIXEL_MIDSIDE_FIDUCIAL_NAME),
    }
    return {key: path for key, path in paths.items() if os.path.exists(path)}
