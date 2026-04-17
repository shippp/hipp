from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling

from hipp.kh9pc.restitution.detectors import CollimationDetector, FlatDetector, PolyDetector, VerticalDetector
from hipp.kh9pc.utils import generate_qc_report, make_summary_figure


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


def plot_summary(detector: object) -> Figure:
    """Render the str() representation of any detector as a figure."""
    return make_summary_figure(str(detector).splitlines())


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
# FlatDetector
# ---------------------------------------------------------------------------


def plot_flat_ruptures(detector: FlatDetector) -> Figure:
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


def plot_flat_edges(detector: FlatDetector, margin_fraction: float = 0.03) -> Figure:
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
# PolyDetector
# ---------------------------------------------------------------------------


def plot_poly_edges(detector: PolyDetector) -> Figure:
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


def plot_poly_distortions(detector: PolyDetector) -> Figure:
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
# CollimationDetector
# ---------------------------------------------------------------------------


def plot_collimation_edges(detector: CollimationDetector) -> Figure:
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


def plot_collimation_distortions(detector: CollimationDetector) -> Figure:
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
# QC report
# ---------------------------------------------------------------------------


def qc_report_vertical(detector: VerticalDetector, output_path: str | Path) -> None:
    """Save a PDF QC report for a :class:`~hipp.kh9pc.restitution.detectors.VerticalDetector`."""
    generate_qc_report(
        output_path,
        [
            plot_summary(detector),
            plot_vertical_ruptures(detector),
            plot_vertical_edges(detector),
        ],
    )


def qc_report_flat(detector: FlatDetector, output_path: str | Path) -> None:
    """Save a PDF QC report for a :class:`~hipp.kh9pc.restitution.detectors.FlatDetector`."""
    generate_qc_report(
        output_path,
        [
            plot_summary(detector),
            plot_flat_edges(detector),
            plot_flat_ruptures(detector),
        ],
    )


def qc_report_poly(detector: PolyDetector, output_path: str | Path) -> None:
    """Save a PDF QC report for a :class:`~hipp.kh9pc.restitution.detectors.PolyDetector`."""
    generate_qc_report(
        output_path,
        [
            plot_summary(detector),
            plot_poly_edges(detector),
            plot_poly_distortions(detector),
        ],
    )


def qc_report_collimation(detector: CollimationDetector, output_path: str | Path) -> None:
    """Save a PDF QC report for a :class:`~hipp.kh9pc.restitution.detectors.CollimationDetector`."""
    generate_qc_report(
        output_path,
        [
            plot_summary(detector),
            plot_collimation_edges(detector),
            plot_collimation_distortions(detector),
        ],
    )
