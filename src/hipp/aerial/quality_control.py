"""
Module: quality_control.py
Author: godinlu
Date: 30
Description: All function for quality control
"""

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import hipp.image
from hipp.typing import DetectedFiducials


def generate_detect_fiducials_qc(
    detected_fiducials: DetectedFiducials, image: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    resized_coef = 0.1
    circle_radius = int((image.shape[0] / 100) * resized_coef)

    resized_image = hipp.image.resize_img(image, resized_coef)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)  # Convertit en RGB pour affichage

    for label, data in detected_fiducials.items():
        center = data["subpixel_center"] if data["subpixel_center"] else data["approx_center"]
        color = (255, 0, 0) if "corner" in label else (0, 255, 0)
        cv2.circle(image_rgb, (int(center[0] * resized_coef), int(center[1] * resized_coef)), circle_radius, color, -1)  # type: ignore[index]

    return image_rgb


def generate_detect_all_fiducials_qc(
    all_detected_fiducials: dict[str, DetectedFiducials], image: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    resized_coef = 0.1
    circle_radius = int((image.shape[0] / 500) * resized_coef)

    resized_image = hipp.image.resize_img(image, resized_coef)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    for detected_fiducials in all_detected_fiducials.values():
        for label, data in detected_fiducials.items():
            center = data["subpixel_center"] if data["subpixel_center"] else data["approx_center"]
            color = (255, 0, 0) if "corner" in label else (0, 255, 0)
            cv2.circle(
                image_rgb, (int(center[0] * resized_coef), int(center[1] * resized_coef)), circle_radius, color, -1
            )  # type: ignore[index]
    return image_rgb


def plot_fiducial_center_deviation_boxplots(
    all_detections: dict[str, DetectedFiducials], use_subpixel: bool = True
) -> Figure:
    """
    Affiche un boxplot pour chaque type de fiducial (corner/edge) représentant
    la distribution de la distance à la moyenne des centres détectés.

    Args:
        all_detections: Un dictionnaire mapping ID -> DetectedFiducials
        use_subpixel: Si True, utilise subpixel_center sinon approx_center
    """
    fiducial_distances = defaultdict(list)

    # Collecte des coordonnées pour chaque type de fiducial
    for detection in all_detections.values():
        for fiducial_key, data in detection.items():
            if use_subpixel and data.get("subpixel_center") is not None:
                center = data["subpixel_center"]
            else:
                center = data["approx_center"]
            fiducial_distances[fiducial_key].append(center)

    # Calcul des distances à la moyenne
    distance_data = {}
    for key, centers in fiducial_distances.items():
        arr = np.array(centers)
        mean = np.mean(arr, axis=0)
        distances = np.linalg.norm(arr - mean, axis=1)
        distance_data[key] = distances

    # Création des boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(list(distance_data.values()), tick_labels=list(distance_data.keys()), vert=True, patch_artist=True)
    ax.set_ylabel("Distance à la moyenne (pixels)")
    ax.set_title("Dispersion des détections de fiduciaux")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def plot_fiducial_score_boxplots(all_detections: dict[str, DetectedFiducials], use_subpixel: bool = True) -> Figure:
    """
    Affiche des boxplots pour chaque type de fiducial (corner/edge),
    représentant la distribution des scores de similarité (approx ou subpixel).

    Args:
        all_detections: Un dictionnaire mapping ID -> DetectedFiducials
        use_subpixel: Si True, utilise subpixel_score sinon approx_score
    """
    fiducial_scores = defaultdict(list)

    # Récupération des scores
    for detection in all_detections.values():
        for fiducial_key, data in detection.items():
            score_key = "subpixel_score" if use_subpixel else "approx_score"
            score = data.get(score_key)
            if score is not None:
                fiducial_scores[fiducial_key].append(score)

    # Création des boxplots
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(list(fiducial_scores.values()), tick_labels=list(fiducial_scores.keys()), vert=True, patch_artist=True)  # type: ignore[arg-type]
    ax.set_ylabel("Score de similarité (template matching)")
    ax.set_title("Distribution des scores de matching ")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig
