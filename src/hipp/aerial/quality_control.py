"""
Copyright (c) 2025 HIPP developers
Description: All function for quality control
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def create_qc_crop(
    block: cv2.typing.MatLike,
    fiducial_shape: tuple[int, int],
    center: tuple[float, float],
    bloc_name: str,
) -> cv2.typing.MatLike:
    """
    Create a QC crop image from a block and detected fiducial center.

    Args:
        block (np.ndarray): The grayscale image block where detection occurred.
        fiducial (np.ndarray): The template fiducial image (used for crop size).
        center (tuple[float, float]): Detected center of the fiducial in the block.
        bloc_name (str): Name of the block (used for labeling).

    Returns:
        np.ndarray: Annotated BGR crop image for QC.
    """
    crop_height, crop_width = fiducial_shape
    half_h, half_w = crop_height // 2, crop_width // 2
    cx, cy = int(center[0]), int(center[1])

    # Crop region around center
    crop = block[max(0, cy - half_h) : cy + half_h, max(0, cx - half_w) : cx + half_w]
    crop = cv2.resize(crop, (crop_width, crop_height))  # Ensure exact size

    # Convert to BGR for annotation
    crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    # Draw green circle at the center
    center_point = (crop_width // 2, crop_height // 2)
    cv2.circle(crop_color, center_point, 1, (0, 255, 0), 1)

    # Add text label
    cv2.putText(
        crop_color,
        bloc_name,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return crop_color


def concat_images(images: list[cv2.typing.MatLike], grid_cols: int = 4) -> cv2.typing.MatLike:
    """
    Concatenate images into a grid layout (rows x columns).

    Parameters:
        images (list of np.ndarray): List of images (grayscale or BGR).
        grid_cols (int): Number of columns in the grid.

    Returns:
        np.ndarray: Concatenated image as a grid.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)  # default black image in color

    # Get size and number of channels from the first image
    h, w = images[0].shape[:2]
    is_color = len(images[0].shape) == 3

    # Number of rows needed
    grid_rows = (len(images) + grid_cols - 1) // grid_cols

    # Prepare blank canvas
    if is_color:
        canvas = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((grid_rows * h, grid_cols * w), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w
        canvas[y1:y2, x1:x2] = img

    return canvas


def plot_fiducial_deviation(
    detected_fiducial_df: pd.DataFrame, ax: Axes, title: str = "Sum of absolute deviations to mean"
) -> None:
    df = detected_fiducial_df.copy()
    score_cols = [col for col in df.columns if col.endswith("_score")]
    df.drop(columns=score_cols, inplace=True)
    mean_values = df.mean()
    diff_to_mean = (df - mean_values).abs()
    row_sum_diff = diff_to_mean.sum(axis=1)

    ax.plot(row_sum_diff.index, row_sum_diff.values, marker="o", linestyle="-", color="blue")
    ax.set_xticks(row_sum_diff.index)  # définit la position des ticks
    ax.set_xticklabels(row_sum_diff.index, rotation=90)
    ax.grid(True)
    ax.set_title(title)


def plot_detection_score_boxplot(detected_fiducials_df: pd.DataFrame, figure_path: str) -> None:
    score_cols = [col for col in detected_fiducials_df.columns if col.endswith("_score")]

    plt.boxplot(
        [detected_fiducials_df[col] for col in score_cols], tick_labels=score_cols, vert=True, patch_artist=True
    )
    plt.title("Boxplot of Fiducial Matching Scores")
    plt.grid(True)
    plt.ylabel("Matching Score")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Pour éviter que les labels soient coupés
    plt.show()
    plt.savefig(figure_path)


def plot_rmse_after_vs_before(rmse_before: dict[str, float], rmse_after: dict[str, float], figure_path: str) -> None:
    labels = [os.path.splitext(os.path.basename(k))[0] for k in sorted(rmse_before)]
    rmse_before_values = [rmse_before[k] for k in sorted(rmse_before.keys())]
    rmse_after_values = [rmse_after[k] for k in sorted(rmse_before.keys())]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(labels, rmse_before_values, label="RMSE before correction", marker="o", color="blue")
    ax.plot(labels, rmse_after_values, label="RMSE after correction", marker="o", color="red")

    ax.set_ylabel("RMSE (pixels)")
    ax.set_title("RMSE before vs after correction")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(figure_path)


def compute_rmse(s1: pd.Series, s2: pd.Series) -> float:
    common_keys = s1.index.intersection(s2.index)
    diff = s1[common_keys] - s2[common_keys]
    rmse = np.sqrt((diff**2).mean())
    return float(rmse)
