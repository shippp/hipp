"""
Module: quality_control.py
Author: godinlu
Date: 30
Description: All function for quality control
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def find_fiducials_quality_control(result: dict[str, dict[str, object]], image: cv2.typing.MatLike) -> Figure:
    """
    Generates a matplotlib figure for visual quality control of fiducial detection results.

    This function overlays detected fiducial centers (either approximate or subpixel-refined) onto
    the original grayscale image (converted to RGB for visualization). If subpixel refinement was applied,
    a secondary bar plot shows the Euclidean distance between the approximate and subpixel centers
    to illustrate the precision gain for each marker.

    Args:
        result: A dictionary containing fiducial detection outputs for each marker. Each entry should include:
                - "approx_center": the approximate center of the matched fiducial.
                - Optionally, "subpixel_center": the refined center from subpixel matching.
        image: The grayscale image used for detection, as an OpenCV-compatible array.

    Returns:
        A matplotlib Figure object containing:
            - An annotated image showing detected fiducials.
            - (If applicable) A bar chart showing subpixel precision gains.

    Notes:
        - Corner fiducials are drawn in red; midside fiducials in green.
        - Subpixel markers improve detection accuracy by evaluating the location at higher resolution.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convertit en RGB pour affichage

    has_subpixel = any("subpixel_center" in data for data in result.values())

    # create the figure
    if has_subpixel:
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax_img = plt.subplots(figsize=(7, 6))
        ax_bar = None

    # creation of the first part with the image and the position of markers
    for label, data in result.items():
        center = data["subpixel_center"] if "subpixel_center" in data else data["approx_center"]
        color = (255, 0, 0) if "corner" in label else (0, 255, 0)
        cv2.circle(image_rgb, (int(center[0]), int(center[1])), 25, color, -1)  # type: ignore[index]
    ax_img.imshow(image_rgb)
    ax_img.set_title("Fiducial Detection Results")
    ax_img.axis("off")

    # if subpixel hase been used, show a plot with the precision gain
    if has_subpixel and ax_bar:
        labels = []
        distances = []
        for label, data in result.items():
            if "subpixel_center" in data and "approx_center" in data:
                approx = np.array(data["approx_center"])
                subpix = np.array(data["subpixel_center"])
                dist = np.linalg.norm(subpix - approx)
                labels.append(label)
                distances.append(dist)

        ax_bar.bar(range(len(labels)), distances, color="orange")
        ax_bar.set_ylabel("Precision gain (px)")
        ax_bar.set_title("Precision subpixel")
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    plt.close(fig)

    return fig
