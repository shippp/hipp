"""
Copyright (c) 2025 HIPP developers
Description: Functions to generate some quality control plots
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.linear_model import RANSACRegressor


def process_image_mosaicing_qc(
    qc_directory: str, vmax_percentile: int = 97, scale_factor: int = 8, keep: bool = True
) -> None:
    scene_tiles = defaultdict(list)

    # Group image tiles by scene ID (assumed to be the prefix before the first underscore)
    for filename in sorted(os.listdir(qc_directory)):
        match = re.match(r"diff_[a-z]_[a-z]_(.+)\.tif", filename)
        if match:
            base_name = match.group(1)
            scene_tiles[base_name].append(os.path.join(qc_directory, filename))

    for base_name, paths in scene_tiles.items():
        fig, axes = plt.subplots(1, len(paths), figsize=(15, 10))
        axes = axes.flatten()
        for i, path in enumerate(paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            assert img is not None
            img_resized = cv2.resize(
                img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC
            ).astype(np.uint8)
            vmax = np.percentile(img_resized, vmax_percentile)
            im = axes[i].imshow(img_resized, cmap="viridis", vmin=0, vmax=vmax)
            axes[i].set_title(f"{chr(ord('a') + i)} - {chr(ord('a') + i + 1)}\nMAE={np.mean(img_resized):.2f}")
            axes[i].axis("off")
            fig.colorbar(im, ax=axes[i])

            if not keep:
                os.remove(path)

        plt.suptitle("Overlapping images absolute differences", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(qc_directory, f"diff_{base_name}.png"))
        plt.show()


def plot_src_and_dst_points(
    src_points: NDArray[np.generic],
    dst_points: NDArray[np.generic],
    output_size: tuple[int, int],
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> None:
    """Plot source and destination TPS points with boundaries and legends."""

    # scatter des points source (en bleu) et destination (en rouge)
    plt.scatter(dst_points[:, 0], dst_points[:, 1], c="red", s=1, label="Destination points", alpha=0.5)
    plt.scatter(src_points[:, 0], src_points[:, 1], c="blue", s=1, label="Source points")

    # Rectangle de l’output (dans l’espace dst)
    rect_x = [0, output_size[0], output_size[0], 0, 0]
    rect_y = [0, 0, output_size[1], output_size[1], 0]
    plt.plot(rect_x, rect_y, color="green", linewidth=1, label="Output rectangle", linestyle="--")

    plt.gca().invert_yaxis()  # cohérent avec les coordonnées image
    plt.xlabel("x-coordinate [pixels]")
    plt.ylabel("y-coordinate [pixels]")
    plt.title("Source vs Destination points")
    plt.legend()
    plt.tight_layout()

    # Save or display
    if output_plot_path is not None:
        Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)
    if plot:
        plt.show()
    plt.close()


def plot_collimation_gradient(
    collimation_lines: dict[str, RANSACRegressor],
    tf_collimation_lines: dict[str, RANSACRegressor],
    width: int,
    nb_points: int = 100,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> None:
    """
    Plot the gradient of collimation lines before and after a transformation.

    This function computes and plots the gradients (first derivatives) of the
    top and bottom collimation lines both before and after a geometric transformation.
    It can either display the plot or save it to a specified path.

    Args:
        collimation_lines (dict[str, RANSACRegressor]):
            Dictionary containing RANSAC models for the "top" and "bottom" collimation lines before transformation.
        tf_collimation_lines (dict[str, RANSACRegressor]):
            Dictionary containing RANSAC models for the "top" and "bottom" collimation lines after transformation.
        width (int):
            Image width used to define the x-range for prediction.
        nb_points (int, optional):
            Number of points to sample along the x-axis. Defaults to 100.
        plot (bool, optional):
            If True, the plot will be displayed. If False, the figure will be closed after saving. Defaults to True.
        output_plot_path (str | Path | None, optional):
            Path to save the plot as an image file. If None, the plot is not saved. Defaults to None.

    Returns:
        None
    """
    x = np.linspace(0, width, nb_points)
    X = x.reshape(-1, 1)
    y_top = collimation_lines["top"].predict(X)
    y_bottom = collimation_lines["bottom"].predict(X)
    y_tf_top = tf_collimation_lines["top"].predict(X)
    y_tf_bottom = tf_collimation_lines["bottom"].predict(X)

    plt.plot(x, np.gradient(y_top, x), label="Top before transform")
    plt.plot(x, np.gradient(y_bottom, x), label="Bottom before tranform")
    plt.plot(x, np.gradient(y_tf_top, x), label="Top after transform")
    plt.plot(x, np.gradient(y_tf_bottom, x), label="Bottom after transform")

    # Add title and axis labels
    plt.title("Collimation Line Gradients Before and After Transformation")
    plt.xlabel("Horizontal position (pixels)")
    plt.ylabel("Gradient value")

    plt.legend()

    if output_plot_path is not None:
        Path(output_plot_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_plot_path)

    if plot:
        plt.show()
    else:
        plt.close()


def plot_distance_between_collimation_lines(
    collimation_lines: dict[str, RANSACRegressor],
    tf_collimation_lines: dict[str, RANSACRegressor],
    width: int,
    true_distance_between_collimation: int,
    nb_points: int = 100,
    plot: bool = True,
    output_plot_path: str | Path | None = None,
) -> None:
    """
    Plot the distance between the top and bottom collimation lines before and after transformation.

    This function computes and visualizes the vertical distance between two collimation lines
    (top and bottom) across the image width, both before and after a geometric transformation.
    It also overlays the expected true distance as a reference line to assess rectification accuracy.

    Args:
        collimation_lines (dict[str, RANSACRegressor]):
            Dictionary containing RANSAC models for the "top" and "bottom" collimation lines before transformation.
        tf_collimation_lines (dict[str, RANSACRegressor]):
            Dictionary containing RANSAC models for the "top" and "bottom" collimation lines after transformation.
        width (int):
            Image width used to define the x-range for prediction.
        true_distance_between_collimation (int):
            Expected true distance (in pixels) between the top and bottom collimation lines.
        nb_points (int, optional):
            Number of x-samples used to evaluate the fitted lines. Defaults to 100.
        plot (bool, optional):
            If True, displays the plot. If False, closes it after saving. Defaults to True.
        output_plot_path (str | Path | None, optional):
            Path to save the plot image. If None, the plot is not saved. Defaults to None.

    Returns:
        None
    """
    x = np.linspace(0, width, nb_points)
    X = x.reshape(-1, 1)
    y_top = collimation_lines["top"].predict(X)
    y_bottom = collimation_lines["bottom"].predict(X)
    y_tf_top = tf_collimation_lines["top"].predict(X)
    y_tf_bottom = tf_collimation_lines["bottom"].predict(X)

    dist_before_transformation = np.abs(y_top - y_bottom)
    dist_after_transformation = np.abs(y_tf_top - y_tf_bottom)

    plt.plot(x, dist_before_transformation, label="Before transformation")
    plt.plot(x, dist_after_transformation, label="After transformation")

    plt.axhline(
        y=true_distance_between_collimation,
        color="red",
        linestyle="--",
        label=f"True distance : {true_distance_between_collimation}",
    )
    plt.title("Distance Between Collimation Lines Before and After Transformation")
    plt.xlabel("Horizontal position (pixels)")
    plt.ylabel("Distance between lines (pixels)")

    plt.legend()

    if output_plot_path is not None:
        Path(output_plot_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_plot_path)

    if plot:
        plt.show()
    else:
        plt.close()
