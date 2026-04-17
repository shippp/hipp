import dataclasses

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling

from hipp.kh9pc.restitution.strategy import CollimationStrategy, FlatStrategy, PolyStrategy, RectificationStrategy
from hipp.kh9pc.restitution.types import StepResult, StrategyAttempt
from hipp.kh9pc.restitution.vertical import VerticalDetector

# ---------------------------------------------------------------------------
# VerticalDetector
# ---------------------------------------------------------------------------


def plot_vertical_ruptures(detector: VerticalDetector) -> Figure:
    """Band profiles with detected rupture positions for left and right edges."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["left", "right"], [detector.left, detector.right]):
        profile = result.sub_image.band.flatten()
        ax.plot(profile, color="gray")
        ax.axvline(x=result.rupture_local, color="red", label=f"rupture (local={result.rupture_local})")
        ax.set_title(f"{side} band profile (global col={result.position})")
        ax.set_xlabel("local column index")
        ax.set_ylabel("intensity")
        ax.legend()

    return fig


def plot_vertical_edges(
    detector: VerticalDetector,
    margin_fraction: float = 0.03,
    plot_res: float = 0.05,
) -> Figure:
    """Thumbnails around the left and right edge positions."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(detector.raster_filepath) as src:
        margin = int(src.width * margin_fraction)

        for ax, side, edge_col in zip(axes, ["left", "right"], detector.edges):
            col_off = max(0, edge_col - margin)
            col_end = min(src.width, edge_col + margin)
            window = Window(col_off, 0, col_end - col_off, src.height)
            out_shape = (1, int(src.height * plot_res), int(window.width * plot_res))
            band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)

            ax.imshow(band, cmap="gray", aspect="auto")
            ax.axvline(x=(edge_col - col_off) * plot_res, color="red")
            ax.set_title(f"{side} edge (col={edge_col})")
            ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# FlatStrategy
# ---------------------------------------------------------------------------


def plot_flat_ruptures(detector: FlatStrategy) -> Figure:
    """Band profiles (collapsed horizontally) with detected rupture row for top and bottom."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top, detector.bottom]):
        profile = result.sub_image.band.flatten()
        ax.plot(profile, color="steelblue", linewidth=1)
        ax.axvline(result.rupture_local, color="red", linewidth=1.5, label=f"rupture={result.rupture_local}")
        ax.set_title(f"{side} band profile")
        ax.set_xlabel("row index (downsampled)")
        ax.set_ylabel("intensity")
        ax.legend(fontsize=8)

    return fig


def plot_flat_edges(detector: FlatStrategy, margin_fraction: float = 0.03) -> Figure:
    """Thumbnails around the top and bottom edge positions with detected line overlaid."""
    left, right = detector.vertical_edges
    roi_w = right - left

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(detector.raster_filepath) as src:
        margin = int(0.03 * src.height)

        for ax, side, result in zip(axes, ["top", "bottom"], [detector.top, detector.bottom]):
            row_off = max(0, result.position - margin)
            row_end = min(src.height, result.position + margin)
            win_h = row_end - row_off
            thumb = src.read(
                1,
                window=Window(left, row_off, roi_w, win_h),
                out_shape=(512, 512),
                resampling=Resampling.average,
            )
            line_row = (result.position - row_off) / win_h * 512
            ax.imshow(thumb, cmap="gray", aspect="auto")
            ax.axhline(line_row, color="yellow", linewidth=1.5)
            ax.set_title(f"{side} edge — position={result.position} px")
            ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# PolyStrategy
# ---------------------------------------------------------------------------


def plot_poly_edges(detector: PolyStrategy) -> Figure:
    """Subimage thumbnails with RANSAC inliers/outliers and polynomial model for top and bottom edges."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top, detector.bottom]):
        ax.imshow(result.sub_image.band, cmap="gray", aspect="auto")

        inlier_mask = result.model.inlier_mask_
        pts = result.ruptures_local
        ax.scatter(pts[~inlier_mask, 0], pts[~inlier_mask, 1], s=12, c="red", label="outliers")
        ax.scatter(pts[inlier_mask, 0], pts[inlier_mask, 1], s=12, c="green", label="inliers")

        x_global = result.ruptures_global[:, 0].astype(float)
        y_global_pred = result.model.predict(x_global.reshape(-1, 1))
        global_pred = np.column_stack([x_global, y_global_pred.ravel()])
        local_pred = result.sub_image.to_local(global_pred)
        ax.plot(local_pred[:, 0], local_pred[:, 1], color="blue", linewidth=1, label="model")

        ax.set_title(f"{side} edge")
        ax.legend(loc="best", fontsize=8)
        ax.axis("off")

    return fig


def plot_poly_distortions(detector: PolyStrategy) -> Figure:
    """Residual distortion curves (deviation from mean) for top and bottom polynomial fits."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ax.plot(detector.top.distortion[:, 0], detector.top.distortion[:, 1], label="top")
    ax.plot(detector.bottom.distortion[:, 0], detector.bottom.distortion[:, 1], label="bottom")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


# ---------------------------------------------------------------------------
# CollimationStrategy
# ---------------------------------------------------------------------------


def plot_collimation_edges(detector: CollimationStrategy) -> Figure:
    """Subimage thumbnails with RANSAC inliers/outliers and polynomial model for top and bottom collimation lines."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top, detector.bottom]):
        ax.imshow(result.sub_img.band, cmap="gray", aspect="auto")

        inliers = result.model.inlier_mask_
        peaks = result.peaks_local
        ax.scatter(peaks[~inliers, 0], peaks[~inliers, 1], s=12, c="red", label="outliers")
        ax.scatter(peaks[inliers, 0], peaks[inliers, 1], s=12, c="green", label="inliers")

        y_global_pred = result.model.predict(result.peaks_global[:, 0].reshape(-1, 1))
        global_pred = np.column_stack([result.peaks_global[:, 0], y_global_pred])
        local_pred = result.sub_img.to_local(global_pred)
        ax.plot(local_pred[:, 0], local_pred[:, 1], color="blue", linewidth=1, label="model")

        ax.set_title(f"{side} collimation line")
        ax.legend(loc="best", fontsize=8)
        ax.axis("off")

    return fig


def plot_collimation_distortions(detector: CollimationStrategy) -> Figure:
    """Residual distortion curves for top and bottom collimation line fits."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for side, result in zip(["top", "bottom"], [detector.top, detector.bottom]):
        ax.plot(result.distortion[:, 0], result.distortion[:, 1], label=side)

    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


# ---------------------------------------------------------------------------
# Report figure builders
# ---------------------------------------------------------------------------

_STATUS_COLOR = {"ran": "#2ecc71", "skipped": "#95a5a6", "failed": "#e74c3c"}
_STRATEGY_DESCRIPTIONS = {
    "CollimationStrategy": (
        "Détecte les lignes de collimation (bandes noires horizontales) par recherche de pics d'intensité "
        "dans chaque colonne, puis ajuste un polynôme RANSAC sur les points détectés."
    ),
    "PolyStrategy": (
        "Détecte les bords film par détection de ruptures d'intensité (transition fond noir → image), "
        "puis ajuste un polynôme RANSAC sur les ruptures détectées."
    ),
    "FlatStrategy": (
        "Détecte les bords film comme des lignes horizontales plates via une rupture d'intensité "
        "globale. Applique une transformation affine (4 points)."
    ),
}


def plot_pipeline_summary(step_results: list[StepResult]) -> Figure:
    """Pipeline step summary table."""
    fig, ax = plt.subplots(figsize=(11, max(3.5, len(step_results) * 0.55 + 1.5)))
    ax.axis("off")
    ax.set_title("Pipeline Summary", fontsize=14, fontweight="bold", pad=16)

    headers = ["Step", "Status", "Started at", "Duration"]
    rows = []
    cell_colors: list[list[str | tuple[float, float, float, float]]] = []
    for r in step_results:
        duration = f"{r.duration:.1f} s" if r.status != "skipped" else "—"
        started = r.started_at.strftime("%H:%M:%S")
        error_suffix = f"  ✗ {r.error}" if r.error else ""
        rows.append([r.name, r.status + error_suffix, started, duration])
        color = mcolors.to_rgba(_STATUS_COLOR.get(r.status, "#ffffff"), alpha=0.25)
        cell_colors.append(["white", color, "white", "white"])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=cell_colors,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3])
    fig.tight_layout()
    return fig


def plot_strategy_header(attempt: StrategyAttempt) -> Figure:
    """Text page describing a strategy attempt (success or failure)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    name = attempt.strategy.__class__.__name__ if attempt.strategy is not None else "Unknown"
    description = _STRATEGY_DESCRIPTIONS.get(name, "")

    if attempt.success:
        status_txt, status_color = "✓  Stratégie retenue", "#27ae60"
    else:
        status_txt, status_color = "✗  Stratégie échouée", "#c0392b"

    ax.text(0.5, 0.88, name, transform=ax.transAxes, fontsize=18, fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.72, status_txt, transform=ax.transAxes, fontsize=13, ha="center", va="top", color=status_color)

    if description:
        ax.text(
            0.5,
            0.55,
            description,
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="top",
            wrap=True,
            style="italic",
            color="#555555",
        )

    if attempt.failure_reason:
        ax.text(
            0.5,
            0.25,
            f"Raison : {attempt.failure_reason}",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="top",
            color="#c0392b",
        )

    fig.tight_layout()
    return fig


def plot_strategy_params(strategy: RectificationStrategy) -> Figure:
    """Two-column table of strategy parameters."""
    params = {f.name: getattr(strategy, f.name) for f in dataclasses.fields(strategy)}  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(8, max(3, len(params) * 0.45 + 1.5)))
    ax.axis("off")
    ax.set_title("Paramètres", fontsize=12, fontweight="bold", pad=12)

    rows = [[k, str(v)] for k, v in params.items()]
    table = ax.table(
        cellText=rows,
        colLabels=["Paramètre", "Valeur"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    fig.tight_layout()
    return fig


def vertical_figures(detector: VerticalDetector) -> list[Figure]:
    """All QC figures for a VerticalDetector."""
    return [plot_vertical_ruptures(detector), plot_vertical_edges(detector)]


def strategy_figures(strategy: RectificationStrategy) -> list[Figure]:
    """All QC figures for a fitted RectificationStrategy."""
    if isinstance(strategy, CollimationStrategy):
        return [plot_collimation_edges(strategy), plot_collimation_distortions(strategy)]
    if isinstance(strategy, PolyStrategy):
        return [plot_poly_edges(strategy), plot_poly_distortions(strategy)]
    if isinstance(strategy, FlatStrategy):
        return [plot_flat_edges(strategy), plot_flat_ruptures(strategy)]
    return []
