"""
Copyright (c) 2025 HIPP developers
Description: All function for quality control
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.axes import Axes
from rasterio.windows import Window

from hipp.aerial.fiducials import (
    _get_fiducial_template_paths,
    _get_groups,
    compute_principal_point,
    warp_fiducial_coordinates,
)

####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################


def generate_detection_qc_plots(
    images_directory: str,
    detected_fiducials_df: pd.DataFrame,
    output_directory: str | None = None,
    show: bool = False,
    distance_around_fiducial: int = 100,
) -> None:
    """
    Generates quality control plots for detected fiducial markers in aerial images.

    For each image in the given `images_directory`, the function reads the corresponding detection data from
    `detected_fiducials_df`, crops a square region of size `2 * distance_around_fiducial` around each fiducial
    position, and plots these regions in a grid. Detected subpixel coordinates are overlaid using a red cross marker.

    The resulting plots can be saved to `output_directory` if specified, and optionally displayed with the `show` flag.
    """

    d = distance_around_fiducial
    for image_id, detected_fiducials in detected_fiducials_df.iterrows():
        with rasterio.open(os.path.join(images_directory, image_id)) as src:
            groups = _get_groups(detected_fiducials)
            fig, axs = plt.subplots(len(groups), len(groups[0]))
            axs = np.array(axs).reshape(len(groups), len(groups[0]))

            for i, group in enumerate(groups):
                for j, fiducial_name in enumerate(group):
                    x = detected_fiducials[fiducial_name + "_x"]
                    y = detected_fiducials[fiducial_name + "_y"]
                    src_window = Window(int(x - d), int(y - d), d * 2, d * 2)
                    img = src.read(1, window=src_window, boundless=True, fill_value=0)
                    ax = axs[i][j]
                    ax.imshow(img, cmap="gray")
                    ax.plot(x - int(x - d), y - int(y - d), marker="+", color="red", markersize=5, markeredgewidth=1)
                    ax.set_title(fiducial_name, fontsize=10)
                    ax.axis("off")
            fig.tight_layout()
            if not show:
                plt.close()
            if output_directory is not None:
                os.makedirs(output_directory, exist_ok=True)
                fig.savefig(os.path.join(output_directory, image_id.replace(".tif", ".png")))


def plot_fiducials_filtering(detected_fiducials_df: pd.DataFrame, filtered_detected_fiducials_df: pd.DataFrame) -> None:
    """
    Displays a comparison plot showing the effect of filtering on fiducial marker deviations.

    This function compares the deviation of fiducial marker positions before and after filtering. It takes
    `detected_fiducials_df` as the raw detections and `filtered_detected_fiducials_df` as the filtered version.
    Each subplot shows the sum of absolute deviations from the mean position for each fiducial, helping to visualize
    how the filtering impacts detection consistency.
    """

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle("Comparison of Fiducial Deviations")
    fig.supylabel("Sum of absolute deviations to mean (px)")
    _plot_fiducial_deviation(detected_fiducials_df, axs[0], title="before filtering")
    _plot_fiducial_deviation(filtered_detected_fiducials_df, axs[1], title="after filtering")
    plt.tight_layout()
    plt.show()


def plot_detection_score_boxplot(detected_fiducials_df: pd.DataFrame) -> None:
    """
    Generates a boxplot to visualize the distribution of fiducial matching scores across all detected markers.

    This function extracts all columns from `detected_fiducials_df` that end with '_score' and plots their
    distribution using a boxplot. It helps assess the variability and consistency of the matching scores across
    different fiducials, allowing for quick identification of outliers or poor detections.
    """

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


def plot_fiducials_correction(
    detected_fiducials_df: pd.DataFrame,
    transformations: dict[str, cv2.typing.MatLike],
    true_fiducials_mm: pd.Series,
    scanning_resolution_mm: float = 0.02,
) -> None:
    """
    Visualizes the effect of applying geometric corrections on fiducial detections by plotting the Root Mean Square Error (RMSE) before and after correction.

    This function takes detected fiducials, applies provided transformation matrices to correct their coordinates, and compares the corrected and original positions against true fiducial locations.
    It plots RMSE values for each image to illustrate improvements in alignment accuracy after correction.

    The scanning resolution parameter controls the coordinate scaling from physical units to pixels.
    """

    transformed_df = detected_fiducials_df.apply(
        lambda row: warp_fiducial_coordinates(row, transformations[row.name]), axis=1
    )
    rmse1, rmse2 = {}, {}

    for image_id in sorted(detected_fiducials_df.index):
        detected_fiducials1 = detected_fiducials_df.loc[image_id]
        detected_fiducials2 = transformed_df.loc[image_id]

        pp1 = (detected_fiducials1["principal_point_x"], detected_fiducials1["principal_point_y"])
        pp2 = (detected_fiducials2["principal_point_x"], detected_fiducials2["principal_point_y"])

        M1 = np.array(
            [
                [1 / scanning_resolution_mm, 0, pp1[0]],
                [0, -1 / scanning_resolution_mm, pp1[1]],
                [0, 0, 1],
            ]
        )
        M2 = np.array(
            [
                [1 / scanning_resolution_mm, 0, pp2[0]],
                [0, -1 / scanning_resolution_mm, pp2[1]],
                [0, 0, 1],
            ]
        )

        true_fiducials1 = warp_fiducial_coordinates(true_fiducials_mm, M1)
        true_fiducials2 = warp_fiducial_coordinates(true_fiducials_mm, M2)

        rmse1[image_id] = _compute_rmse(detected_fiducials1, true_fiducials1)
        rmse2[image_id] = _compute_rmse(detected_fiducials2, true_fiducials2)

    labels = list(rmse1.keys())
    plt.plot(labels, list(rmse1.values()), label="RMSE before correction", marker="o", color="blue")
    plt.plot(labels, list(rmse2.values()), label="RMSE after correction", marker="o", color="red")

    plt.ylabel("RMSE (pixels)")
    plt.title("RMSE before vs after correction")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_true_fiducials(fiducials: pd.Series) -> None:
    """
    Displays the true fiducial markers on a plot by connecting their coordinates and highlighting key points.

    The function iterates through fiducial groups, plotting each fiducial’s position as scatter points and drawing lines between them to visualize their spatial relationships.
    It specifically differentiates corner fiducials by connecting adjacent points in the group.

    Additionally, it calculates and marks the principal point on the plot for reference, ensuring aspect ratio is equal for accurate spatial representation.
    """

    for group in _get_groups(fiducials):
        for i in range(4):
            coord1 = fiducials[[group[i] + "_x", group[i] + "_y"]]
            coord2 = fiducials[[group[(i + 1) % 4] + "_x", group[(i + 1) % 4] + "_y"]]
            coord3 = fiducials[[group[(i + 2) % 4] + "_x", group[(i + 2) % 4] + "_y"]]
            plt.scatter(*coord1, color="black", zorder=10)
            if "corner" in group[0]:
                plt.plot(
                    [coord1.iloc[0], coord2.iloc[0]], [coord1.iloc[1], coord2.iloc[1]], linestyle="-", color="gray"
                )
            plt.plot([coord1.iloc[0], coord3.iloc[0]], [coord1.iloc[1], coord3.iloc[1]], linestyle="-", color="gray")

    principal_point = compute_principal_point(fiducials)
    assert principal_point is not None
    plt.scatter(*principal_point, color="red", s=100, marker="+", zorder=10)

    plt.title("True fiducials")
    plt.axis("equal")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_fiducial_templates(fiducials_directory: str) -> None:
    """
    Visualizes fiducial and subpixel fiducial template images from a specified directory.

    This function loads the available template images, displaying them in a 2x2 grid layout.
    Each subplot shows a grayscale fiducial template, with titles indicating the type of template for easy identification.
    """

    template_paths = _get_fiducial_template_paths(fiducials_directory)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    # Plot each template if it exists
    for i, key in enumerate(template_paths):
        fiducial_image = cv2.imread(template_paths[key], cv2.IMREAD_GRAYSCALE)
        axes[i].imshow(fiducial_image, cmap="gray")
        axes[i].set_title(key)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def _plot_fiducial_deviation(
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


def _compute_rmse(s1: pd.Series, s2: pd.Series) -> float:
    common_keys = s1.index.intersection(s2.index)
    diff = s1[common_keys] - s2[common_keys]
    rmse = np.sqrt((diff**2).mean())
    return float(rmse)
