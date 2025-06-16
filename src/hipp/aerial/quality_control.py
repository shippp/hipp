"""
Copyright (c) 2025 HIPP developers
Description: All function for quality control
"""

import math
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes

from hipp.aerial.fiducials import Fiducials, FiducialsCoordinate


def save_fiducials_detection_qc(
    all_detections: dict[str, FiducialsCoordinate],
    all_scores: dict[str, Fiducials[float]],
    all_subpixel_scores: dict[str, Fiducials[float]],
    qc_detection_dir: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_principal_points_deviation(all_detections, axes[0])
    plot_fiducial_deviation_boxplots(all_detections, axes[1])
    plt.tight_layout()
    if qc_detection_dir is not None:
        fig.savefig(os.path.join(qc_detection_dir, "Deviation_plot.png"))
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_fiducial_score_boxplots(all_scores, axes[0])
    plot_fiducial_score_boxplots(all_subpixel_scores, axes[1], title="Distribution of subpixel matching score")
    plt.tight_layout()
    if qc_detection_dir is not None:
        fig.savefig(os.path.join(qc_detection_dir, "Scores_boxplot.png"))
    plt.show()


def save_process_fiducials_detection_qc(
    all_detections: dict[str, FiducialsCoordinate],
    processed_detections: dict[str, FiducialsCoordinate],
    qc_detection_dir: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_principal_points_deviation(all_detections, axes[0][0])
    plot_principal_points_deviation(
        processed_detections, axes[0][1], "Principal points Deviation after removing wrong detection"
    )
    plot_fiducial_deviation_boxplots(all_detections, axes[1][0])
    plot_fiducial_deviation_boxplots(
        processed_detections, axes[1][1], "Fiducials Deviation after removing wrong detection"
    )
    plt.tight_layout()
    if qc_detection_dir is not None:
        fig.savefig(os.path.join(qc_detection_dir, "Deviation_correction_plot.png"))
    plt.show()


def generate_fiducial_qc_image_from_detection(
    image_path: str,
    detections: FiducialsCoordinate,
    distance_around_fiducial: int = 100,
    grid_cols: int | None = None,
) -> cv2.typing.MatLike:
    """
    Generates a single QC image composed of fiducial patches centered on subpixel locations.

    Args:
        image_path: Path to the full-resolution image.
        detections: Dictionary from detect_fiducials_fast.
        fiducial_size: Tuple of (height, width) to extract around each subpixel center.
        grid_cols: Number of columns in the QC image grid (optional, auto if None).

    Returns:
        A single image (numpy array) containing all extracted fiducial patches arranged in a grid.
    """
    patch_size = 2 * distance_around_fiducial
    patches = []
    with rasterio.open(image_path) as src:
        for label, coord in detections.items():
            if label != "principal_point" and coord is not None:
                cx, cy = coord
                cx_int, cy_int = int(round(cx)), int(round(cy))

                window = rasterio.windows.Window(
                    cx_int - distance_around_fiducial, cy_int - distance_around_fiducial, patch_size, patch_size
                )
                patch = src.read(1, window=window)

                # Convertir en BGR pour annotations
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

                # Position subpixel locale dans la vignette
                dx = cx - (cx_int - distance_around_fiducial)
                dy = cy - (cy_int - distance_around_fiducial)
                cv2.circle(patch_bgr, (int(round(dx)), int(round(dy))), 2, (0, 255, 0), -1)

                # Annoter le nom
                cv2.putText(patch_bgr, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

                patches.append(patch_bgr)

    # Organisation en grille
    n = len(patches)
    cols = grid_cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid_img = np.zeros((rows * patch_size, cols * patch_size, 3), dtype=np.uint8)

    for idx, patch in enumerate(patches):
        r, c = divmod(idx, cols)
        grid_img[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = patch

    return grid_img


def plot_fiducial_deviation_boxplots(
    all_coordinates: dict[str, FiducialsCoordinate], ax: Axes, title: str = "Deviation of fiducials"
) -> None:
    """
    Affiche un boxplot pour chaque type de fiducial (corner/edge, principal_point) représentant
    la distribution de la distance à la moyenne des centres détectés.

    Args:
        all_detections: Un dictionnaire mapping ID -> Fiducials
        use_subpixel: Si True, utilise subpixel_center sinon approx_center
    """
    # Regrouper toutes les coordonnées valides par nom de fiducial
    fiducial_points = defaultdict(list)
    for detection in all_coordinates.values():
        for name, coord in detection.items():
            if coord is not None:
                fiducial_points[name].append(coord)

    # Calculer les distances à la moyenne pour chaque type de fiducial
    deviations_by_fiducial = {}
    for name, coords in fiducial_points.items():
        if len(coords) < 2:
            # Impossible de calculer une déviation avec moins de deux points
            continue
        coords_array = np.array(coords)
        mean = coords_array.mean(axis=0)
        distances = np.linalg.norm(coords_array - mean, axis=1)
        deviations_by_fiducial[name] = distances

    # Création du plot
    labels = sorted(deviations_by_fiducial.keys())
    data = [deviations_by_fiducial[label] for label in labels]

    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel("Distance from mean (pixels)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")


def plot_fiducial_score_boxplots(
    all_scores: dict[str, Fiducials[float]], ax: Axes, title: str = "Distribution of matching score"
) -> None:
    """
    Affiche des boxplots pour chaque type de fiducial (corner/edge),
    représentant la distribution des scores de similarité (approx ou subpixel).

    Args:
        all_detections: Un dictionnaire mapping ID -> Fiducials
        use_subpixel: Si True, utilise subpixel_score sinon approx_score
    """
    fiducial_scores = defaultdict(list)
    for detection in all_scores.values():
        for name, coord in detection.items():
            fiducial_scores[name].append(coord)

    ax.boxplot(list(fiducial_scores.values()), tick_labels=list(fiducial_scores.keys()), vert=True, patch_artist=True)  # type: ignore[arg-type]
    ax.set_ylabel("Similarity score (template matching)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_principal_points_deviation(
    all_coordinates: dict[str, FiducialsCoordinate],
    ax: Axes,
    title: str = "Principal points Deviation",
) -> None:
    # Récupération des points principaux
    entity_names = sorted(all_coordinates.keys())
    principal_points = np.array([all_coordinates[name]["principal_point"] for name in entity_names])

    # Moyenne des points principaux
    mean_point = np.mean(principal_points, axis=0)

    # Calcul des distances à la moyenne
    distances = np.linalg.norm(principal_points - mean_point, axis=1)

    # Extraire des labels propres
    labels = [os.path.splitext(os.path.basename(name))[0] for name in entity_names]

    # Création de la figure
    ax.bar(range(len(labels)), distances, color="skyblue")

    # Ajout des étiquettes proprement
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)

    # Mise en forme
    ax.set_title(title)
    ax.set_ylabel("Distance (pixels)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)


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


def compute_rmse(detection1: FiducialsCoordinate, detection2: FiducialsCoordinate) -> float:
    squared_errors = []
    for key in detection1.keys():
        coord1 = detection1.get(key)
        coord2 = detection2.get(key)
        if coord1 is not None and coord2 is not None:
            dx = coord1[0] - coord2[0]
            dy = coord1[1] - coord2[1]
            squared_errors.append(dx**2 + dy**2)

    if not squared_errors:
        raise ValueError("No common valid fiducials to compare.")

    mse = sum(squared_errors) / len(squared_errors)
    return math.sqrt(mse)


def compute_intersections_angles(fiducials: dict[str, tuple[float, float]]) -> dict[str, float]:
    angle_defs = {
        "corner_angle": [("corner_top_left", "corner_bottom_right"), ("corner_top_right", "corner_bottom_left")],
        "midside_angle": [("midside_top", "midside_bottom"), ("midside_right", "midside_left")],
    }

    result = {}
    for angle_name, ((p1, p2), (p3, p4)) in angle_defs.items():
        if all(k in fiducials for k in [p1, p2, p3, p4]):
            segment1 = (fiducials[p1], fiducials[p2])
            segment2 = (fiducials[p3], fiducials[p4])
            result[angle_name] = compute_angle_between_segments(segment1, segment2)

    return result


def compute_angle_between_segments(
    segment1: tuple[tuple[float, float], tuple[float, float]], segment2: tuple[tuple[float, float], tuple[float, float]]
) -> float:
    """
    Computes the angle (in degrees) between two segments.

    Args:
        seg1: Segment defined by two points.
        seg2: Segment defined by two points.

    Returns:
        float: The angle in degrees.
    """
    v1 = np.array([segment1[1][0] - segment1[0][0], segment1[1][1] - segment1[0][1]])
    v2 = np.array([segment2[1][0] - segment2[0][0], segment2[1][1] - segment2[0][1]])

    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))
