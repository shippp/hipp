from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patches
from matplotlib.figure import Figure
from rasterio.warp import Resampling
from rasterio.windows import Window

from hipp.kh9pc.restitution_strategy.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution_strategy.flat_strategy import FlatStrategy
from hipp.kh9pc.restitution_strategy.mixed_strategy import MixedStrategy
from hipp.kh9pc.restitution_strategy.poly_strategy import PolyStrategy
from hipp.kh9pc.types import FittingClass, Transformation
from hipp.kh9pc.vertical_detector import VerticalDetector

# ---------------------------------------------------------------------------
# VerticalDetector
# ---------------------------------------------------------------------------


def plot_vertical_ruptures(detector: VerticalDetector) -> Figure:
    """Band profiles with detected rupture positions for left and right edges."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["left", "right"], [detector.left_, detector.right_]):
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

    with rasterio.open(detector.raster_filepath_) as src:
        margin = int(src.width * margin_fraction)

        for ax, side, edge_col in zip(axes, ["left", "right"], detector.edges_):
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

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
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
    left, right = detector.vertical_detector.edges_
    roi_w = right - left

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(detector.raster_filepath_) as src:
        margin = int(margin_fraction * src.height)

        for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
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

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
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

    ax.plot(detector.top_.distortion[:, 0], detector.top_.distortion[:, 1], label="top")
    ax.plot(detector.bottom_.distortion[:, 0], detector.bottom_.distortion[:, 1], label="bottom")
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

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
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

    for side, result in zip(["top", "bottom"], [detector.top_, detector.bottom_]):
        ax.plot(result.distortion[:, 0], result.distortion[:, 1], label=side)

    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


# ---------------------------------------------------------------------------
# Transformation overview (all strategies)
# ---------------------------------------------------------------------------


def plot_deformation_grid(
    transform: "Transformation",
    num: int = 20,
    figsize: tuple[int, int] = (6, 6),
) -> Figure:
    """
    Visualize deformation field by plotting the warped grid lines.

    Each grid line is sampled in the original image space, then
    warped using transform.deformation and re-plotted.
    """
    with rasterio.open(transform.raster_filepath) as src:
        w, h = src.width, src.height

    xs = np.linspace(0, w - 1, num, dtype=np.float32)
    ys = np.linspace(0, h - 1, num, dtype=np.float32)

    fig, ax = plt.subplots(figsize=figsize)

    # -----------------------------------
    # Horizontal lines (y fixed, x varies)
    # -----------------------------------
    for y in ys:
        line = np.stack([xs, np.full_like(xs, y)], axis=-1)
        warped_line = transform.deformation(line)

        ax.plot(
            warped_line[:, 0],
            warped_line[:, 1],
            color="gray",
            lw=0.8,
            alpha=0.7,
        )

    # -----------------------------------
    # Vertical lines (x fixed, y varies)
    # -----------------------------------
    for x in xs:
        line = np.stack([np.full_like(ys, x), ys], axis=-1)
        warped_line = transform.deformation(line)

        ax.plot(
            warped_line[:, 0],
            warped_line[:, 1],
            color="gray",
            lw=0.8,
            alpha=0.7,
        )

    ax.set_title("Warped deformation grid")
    ax.invert_yaxis()

    return fig


def plot_crop_area(transform: "Transformation", figsize: tuple[int, int] = (6, 6)) -> Figure:
    """
    Visualize crop region with a translated coordinate system:
    the crop origin (0,0) is shown at its real position in the original image,
    but axes are interpreted in crop-local coordinates.
    """
    fig, ax = plt.subplots(figsize=figsize)

    with rasterio.open(transform.raster_filepath) as src:
        w, h = src.width, src.height

    crop_x, crop_y = transform.crop_offset
    crop_w, crop_h = transform.output_size

    # Full image boundary (still in original frame)
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            w,
            h,
            fill=False,
            edgecolor="black",
            linewidth=2,
            label="Original image",
        )
    )

    # Crop area (in original coordinates)
    ax.add_patch(
        patches.Rectangle(
            (crop_x, crop_y),
            crop_w,
            crop_h,
            fill=True,
            alpha=0.3,
            color="orange",
            label="Crop region",
        )
    )

    # NEW: show crop-local origin correctly interpreted
    ax.scatter(
        crop_x,
        crop_y,
        color="red",
        marker="+",
        s=10,
        label="Crop origin (0,0 in crop space)",
    )

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("auto")
    ax.set_box_aspect(h / (w / 2))
    ax.invert_yaxis()
    ax.set_title(f"Crop visualization\ncrop_offset = ({crop_x}, {crop_y}), size = ({crop_w}, {crop_h})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    return fig


# ---------------------------------------------------------------------------
# Report figure builders
# ---------------------------------------------------------------------------

_STATUS_COLOR = {"ran": "#2ecc71", "skipped": "#95a5a6", "failed": "#e74c3c"}
_STRATEGY_DESCRIPTIONS = {
    "CollimationStrategy": (
        "Detects collimation lines (horizontal black bands) by searching for intensity peaks in each column, "
        "then fits a RANSAC polynomial through the detected points."
    ),
    "PolyStrategy": (
        "Detects film edges via intensity rupture detection (black background → image transition), "
        "then fits a RANSAC polynomial through the detected ruptures."
    ),
    "FlatStrategy": (
        "Detects film edges as flat horizontal lines via a global intensity rupture. "
        "Applies an affine transform (4 control points)."
    ),
}


def plot_pipeline_summary(
    step_results: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> Figure:
    """Pipeline step summary table with optional provenance metadata."""
    fig, ax = plt.subplots(figsize=(11, max(3.5, len(step_results) * 0.55 + 2.0)))
    ax.axis("off")
    ax.set_title("Pipeline Summary", fontsize=14, fontweight="bold", pad=16)

    if meta:
        parts = []
        if meta.get("entity_id"):
            parts.append(f"Scene: {meta['entity_id']}")
        if meta.get("hipp_version"):
            parts.append(f"hipp {meta['hipp_version']}")
        if meta.get("git_hash"):
            parts.append(f"git {meta['git_hash']}")
        if parts:
            ax.text(
                0.5,
                0.97,
                "  |  ".join(parts),
                transform=ax.transAxes,
                fontsize=9,
                ha="center",
                va="top",
                color="#666666",
            )

    headers = ["Step", "Status", "Started at", "Duration"]
    rows = []
    cell_colors: list[list[str | tuple[float, float, float, float]]] = []
    for r in step_results:
        duration = f"{r['duration']:.1f} s" if r["status"] != "skipped" else "—"
        error_suffix = f"  ✗ {r['error']}" if r["error"] else ""
        rows.append([r["name"], r["status"] + error_suffix, r["started_at"], duration])
        color = mcolors.to_rgba(_STATUS_COLOR.get(r["status"], "#ffffff"), alpha=0.25)
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


def get_figures(fitting_class: FittingClass) -> list[Figure]:
    """All QC figures for a fitted fitting_class."""
    if isinstance(fitting_class, VerticalDetector):
        return [plot_vertical_edges(fitting_class), plot_vertical_ruptures(fitting_class)]
    if isinstance(fitting_class, FlatStrategy):
        return [
            *get_figures(fitting_class.vertical_detector),
            plot_flat_edges(fitting_class),
            plot_flat_ruptures(fitting_class),
            plot_crop_area(fitting_class.get_transformation()),
        ]
    if isinstance(fitting_class, PolyStrategy):
        return [
            *get_figures(fitting_class.vertical_detector),
            plot_poly_edges(fitting_class),
            plot_poly_distortions(fitting_class),
            plot_deformation_grid(fitting_class.get_transformation()),
            plot_crop_area(fitting_class.get_transformation()),
        ]
    if isinstance(fitting_class, CollimationStrategy):
        return [
            *get_figures(fitting_class.vertical_detector),
            plot_collimation_edges(fitting_class),
            plot_collimation_distortions(fitting_class),
            plot_deformation_grid(fitting_class.get_transformation()),
            plot_crop_area(fitting_class.get_transformation()),
        ]
    if isinstance(fitting_class, MixedStrategy):
        return get_figures(fitting_class.selected_strategy_)

    return []
