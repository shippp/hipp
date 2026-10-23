"""
Copyright (c) 2026 HIPP developers
Description: Quality control figures for all KH-9 PC restitution strategies.
    Each ``plot_*`` function reads one detection result or transformation and writes its figure to
    ``output_file``, skipped if it already exists and ``overwrite`` is False.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.detection.detect_collimation_lines import CollimationDetectionResult
from hipp.kh9pc.detection.detect_fiducials import FiducialDetectionResult
from hipp.kh9pc.detection.detect_flat_horizontal_edges import FlatDetectionResult
from hipp.kh9pc.detection.detect_poly_horizontal_edges import PolyDetectionResult
from hipp.kh9pc.detection.detect_vertical_edges import VerticalDetectionResult
from hipp.kh9pc.transformations import Transformation
from hipp.tools import skip_existing

logger = logging.getLogger(__name__)


def _save(fig: Figure, output_file: str | Path) -> None:
    """Write *fig* to *output_file*, creating parent directories, and close it."""
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        # tight bbox so legends placed outside the axes are not clipped
        fig.savefig(output_file, bbox_inches="tight")
    finally:
        plt.close(fig)


# --- Vertical ---
def plot_vertical_ruptures(
    result: str | Path | VerticalDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Band profiles with detected edge positions for left and right sides."""
    if skip_existing(output_file, overwrite):
        return
    result = VerticalDetectionResult.load(result)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, edge in zip(axes, ["left", "right"], [result.left, result.right]):
        sub_image = edge.sub_image.read(result.raster_filepath)
        local_edge = int(sub_image.to_local_x(edge.position))
        ax.plot(sub_image.band.flatten(), color="gray")
        ax.axvline(x=local_edge, color="red", label=f"edge (local={local_edge})")
        ax.set_title(f"{side} profile \n(global col={edge.position}, ratio={edge.gradient_ratio:.2f})")
        ax.set_xlabel("local column index")
        ax.set_ylabel("intensity")
        ax.legend()

    fig.suptitle(result.raster_filepath.stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)


def plot_vertical_edges(
    result: str | Path | VerticalDetectionResult,
    output_file: str | Path,
    window_width: int = 10000,
    out_height: int = 800,
    overwrite: bool = False,
) -> None:
    """Thumbnails around the left and right edge positions."""
    if skip_existing(output_file, overwrite):
        return
    result = VerticalDetectionResult.load(result)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(result.raster_filepath) as src:
        for ax, side, edge_col in zip(axes, ["left", "right"], [result.left.position, result.right.position]):
            window = Window(edge_col - window_width // 2, 0, window_width, src.height)
            scale = out_height / window.height
            out_shape = (1, out_height, int(window.width * scale))
            sub_image = SubImage(src, window=window, out_shape=out_shape)

            ax.imshow(sub_image.band, cmap="gray", aspect="auto")

            ax.axvline(x=sub_image.to_local_x(edge_col), color="red")
            ax.set_title(f"{side} edge (col={edge_col})")
            ax.axis("off")

    fig.suptitle(result.raster_filepath.stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)


# --- Flat ---
def plot_flat_ruptures(
    result: str | Path | FlatDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Band profiles (collapsed horizontally) with detected rupture row for top and bottom."""
    if skip_existing(output_file, overwrite):
        return
    result = FlatDetectionResult.load(result)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(result.raster_filepath) as src:
        for ax, side, edge in zip(axes, ["top", "bottom"], [result.top, result.bottom]):
            sub_image = edge.sub_image.read(src)
            rupture_local = int(sub_image.to_local_y(edge.position))
            ax.plot(sub_image.band.flatten(), color="steelblue", linewidth=1)
            ax.axvline(rupture_local, color="red", linewidth=1.5, label=f"rupture={rupture_local}")
            ax.set_title(f"{side} band profile")
            ax.set_xlabel("row index (downsampled)")
            ax.set_ylabel("intensity")
            ax.legend(fontsize=8)

    fig.suptitle(result.raster_filepath.stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)


def plot_flat_edges(
    result: str | Path | FlatDetectionResult,
    output_file: str | Path,
    margin_fraction: float = 0.03,
    overwrite: bool = False,
) -> None:
    """Thumbnails around the top and bottom edge positions with detected line overlaid."""
    if skip_existing(output_file, overwrite):
        return
    result = FlatDetectionResult.load(result)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(result.raster_filepath) as src:
        margin = int(margin_fraction * src.height)

        for ax, side, edge in zip(axes, ["top", "bottom"], [result.top, result.bottom]):
            window = Window(result.left, edge.position - margin, result.right - result.left, 2 * margin)
            sub_image = SubImage(src, window, out_shape=(1, 512, 512))
            ax.imshow(sub_image.band, cmap="gray", aspect="auto")
            ax.axhline(sub_image.to_local_y(edge.position), color="yellow", linewidth=1.5)
            ax.set_title(f"{side} edge — position={edge.position} px")
            ax.axis("off")

    fig.suptitle(result.raster_filepath.stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)


# --- Poly ---
def plot_poly_edges(result: str | Path | PolyDetectionResult, output_file: str | Path, overwrite: bool = False) -> None:
    """Subimage thumbnails with RANSAC inliers/outliers and polynomial model for top and bottom edges."""
    if skip_existing(output_file, overwrite):
        return
    result = PolyDetectionResult.load(result)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    x = np.linspace(result.left, result.right, 500)

    with rasterio.open(result.raster_filepath) as src:
        for ax, side, edge in zip(axes, ["top", "bottom"], [result.top, result.bottom]):
            sub_image = edge.sub_image.read(src)
            ax.imshow(sub_image.band, cmap="gray", aspect="auto")

            inliers = np.array(edge.inlier_mask)
            pts = sub_image.to_local(np.array(edge.edge_points))
            ax.scatter(pts[~inliers, 0], pts[~inliers, 1], s=12, c="red", label="outliers")
            ax.scatter(pts[inliers, 0], pts[inliers, 1], s=12, c="green", label="inliers")

            model = sub_image.to_local(np.column_stack([x, edge.predict(x)]))
            ax.plot(model[:, 0], model[:, 1], color="blue", linewidth=1, label="model")

            ax.set_title(f"{side} edge (inliers={100 * inliers.mean():.1f}%)")
            ax.legend(loc="best", fontsize=8)
            ax.axis("off")

    fig.suptitle(result.raster_filepath.stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)


def plot_poly_distortions(
    result: str | Path | PolyDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Residual distortion curves (deviation from mean) for top and bottom polynomial fits."""
    if skip_existing(output_file, overwrite):
        return
    result = PolyDetectionResult.load(result)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.linspace(result.left, result.right, 500)

    for side, edge in zip(["top", "bottom"], [result.top, result.bottom]):
        y = edge.predict(x)
        ax.plot(x, y - y.mean(), label=side)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    _save(fig, output_file)


# --- Collimation ---
def plot_collimation_edges(
    result: str | Path | CollimationDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Subimage thumbnails with RANSAC inliers/outliers and polynomial model for top and bottom collimation lines."""
    if skip_existing(output_file, overwrite):
        return
    result = CollimationDetectionResult.load(result)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    x = np.linspace(result.left, result.right, 500)

    with rasterio.open(result.raster_filepath) as src:
        for ax, side, line in zip(axes, ["top", "bottom"], [result.top, result.bottom]):
            sub_image = line.sub_image.read(src)
            ax.imshow(sub_image.band, cmap="gray", aspect="auto")

            inliers = np.array(line.inlier_mask)
            peaks = sub_image.to_local(np.array(line.peaks))
            ax.scatter(peaks[~inliers, 0], peaks[~inliers, 1], s=12, c="red", label="outliers")
            ax.scatter(peaks[inliers, 0], peaks[inliers, 1], s=12, c="green", label="inliers")

            model = sub_image.to_local(np.column_stack([x, line.predict(x)]))
            ax.plot(model[:, 0], model[:, 1], color="blue", linewidth=1, label="model")

            ax.set_title(f"{side} collimation line (inliers={100 * line.inlier_ratio:.1f}%)")
            ax.legend(loc="best", fontsize=8)
            ax.axis("off")

    fig.suptitle(result.raster_filepath.stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)


def plot_collimation_distortions(
    result: str | Path | CollimationDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Residual distortion curves (deviation from mean) for top and bottom collimation line fits."""
    if skip_existing(output_file, overwrite):
        return
    result = CollimationDetectionResult.load(result)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.linspace(result.left, result.right, 500)

    for side, line in zip(["top", "bottom"], [result.top, result.bottom]):
        y = line.predict(x)
        ax.plot(x, y - y.mean(), label=side)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    _save(fig, output_file)


# --- Fiducial ---
_PATTERN_COLORS: dict[str, str] = {
    "regular_sparse": "red",
    "regular_mid": "orange",
    "regular_dense": "gold",
    "segmented_mid": "limegreen",
    "segmented_dense": "cyan",
    "serialized_time_word": "violet",
}


def _coord_index(centers_xy: np.ndarray) -> dict[tuple[float, float], int]:
    """Build a (cx, cy) → row-index lookup for fast pattern membership queries."""
    return {(float(cx), float(cy)): i for i, (cx, cy) in enumerate(centers_xy)}


def plot_fiducial_filtering(
    result: str | Path | FiducialDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Pattern detection diagnostics: spatial scatter and feature space for top and bottom sides.

    Each row corresponds to one side (top / bottom). The left column shows detections in
    global image space (cx vs cy) with the fitted polynomial edge overlaid. The right column
    shows the raw feature space (matching score vs residual to the edge model).

    Valid patterns are highlighted with distinct colours; unmatched detections appear in light gray.
    """
    if skip_existing(output_file, overwrite):
        return
    result = FiducialDetectionResult.load(result)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Fiducial pattern detection ({result.raster_filepath.stem})", fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("tab10")
    _noise = (0.85, 0.85, 0.85, 1.0)

    for row, (side, side_result) in enumerate(zip(("top", "bottom"), (result.top, result.bottom))):
        ax_spatial, ax_feat = axes[row]

        centers_xy = side_result.centers_xy
        features = side_result.features
        coord_idx = _coord_index(centers_xy)

        ax_spatial.scatter(centers_xy[:, 0], centers_xy[:, 1], c=[_noise], s=10, linewidths=0)
        ax_feat.scatter(features[:, 0], features[:, 1], c=[_noise], s=10, linewidths=0)

        legend_handles: list[Line2D] = []

        for i, (name, pattern) in enumerate(side_result.patterns.items()):
            if pattern.count == 0:
                continue
            color = cmap(i % 10)
            score = pattern.score
            star = " ★" if score > result.options.min_score_threshold else ""

            indices = [coord_idx[k] for pt in pattern.points if (k := (float(pt[0]), float(pt[1]))) in coord_idx]
            if indices:
                idx = np.array(indices)
                ax_spatial.scatter(centers_xy[idx, 0], centers_xy[idx, 1], c=[color], s=25, linewidths=0)
                ax_feat.scatter(features[idx, 0], features[idx, 1], c=[color], s=25, linewidths=0)

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=6,
                    label=f"{name}{star}  score={score:.3f}  n={pattern.count}",
                )
            )

        x_grid = np.linspace(0, float(centers_xy[:, 0].max()), 300)
        y_pred = np.polyval(side_result.edge_coefficients, x_grid)
        ax_spatial.plot(x_grid, y_pred, color="steelblue", linewidth=1.0, linestyle="--")

        ax_spatial.invert_yaxis()
        ax_spatial.set_title(f"{side} — spatial  ({len(centers_xy)} detections)")
        ax_spatial.set_xlabel("cx (px)")
        ax_spatial.set_ylabel("cy (px)")

        ax_feat.legend(
            handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, borderaxespad=0
        )
        ax_feat.set_title(f"{side} — feature space")
        ax_feat.set_xlabel("score")
        ax_feat.set_ylabel("residual (px)")

    _save(fig, output_file)


def plot_fiducial_distortions(
    result: str | Path | FiducialDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Fiducial center y-deviation from mean, per valid pattern, for top and bottom sides.

    In the ideal case all points lie at 0. Divergence reveals scan distortion.
    """
    if skip_existing(output_file, overwrite):
        return
    result = FiducialDetectionResult.load(result)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"Fiducial distortion — {result.raster_filepath.stem}", fontsize=12, fontweight="bold")

    for side, side_result in zip(["top", "bottom"], [result.top, result.bottom]):
        for name, pattern in side_result.patterns.items():
            if pattern.score <= result.options.min_score_threshold or pattern.count < 8:
                continue
            x, y = np.array(pattern.points).T
            ax.scatter(
                x,
                y - y.mean(),
                s=8,
                marker="x" if side == "bottom" else "o",
                label=f"{side} · {name}  (n={len(x)})",
            )

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    _save(fig, output_file)


def plot_fiducial_detected_profiles(
    result: str | Path | FiducialDetectionResult,
    output_file: str | Path,
    window_height_fraction: float = 0.08,
    overwrite: bool = False,
) -> None:
    """Detected fiducial centers overlaid on the top and bottom image strips, one scatter per valid pattern."""
    if skip_existing(output_file, overwrite):
        return
    result = FiducialDetectionResult.load(result)
    min_score = result.options.min_score_threshold
    sides_results = [result.top, result.bottom]
    n_insets = max(
        max(sum(1 for p in r.patterns.values() if p.score > min_score) for r in sides_results),
        1,
    )

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Fiducial detected profiles — {result.raster_filepath.stem}", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(2, 1 + n_insets, width_ratios=[14] + [1] * n_insets)
    main_axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
    inset_slots = [[fig.add_subplot(gs[row, col + 1]) for col in range(n_insets)] for row in range(2)]

    with rasterio.open(result.raster_filepath) as src:
        window_height = int(src.height * window_height_fraction)
        windows = [
            Window(0, 0, src.width, window_height),
            Window(0, src.height - window_height, src.width, window_height),
        ]

        def _spacing_info(cx: np.ndarray) -> str:
            if len(cx) >= 2:
                dists = np.diff(np.sort(cx.astype(np.float64)))
                return f"spacing  mean={float(dists.mean()):.1f}px  std={float(dists.std()):.1f}px"
            return "spacing  n/a"

        sides = zip(main_axes, ["top", "bottom"], windows, sides_results)
        for row, (ax, side, window, side_result) in enumerate(sides):
            sub_img = SubImage(src, window, (1, 512, 4096))
            ax.imshow(sub_img.band, cmap="gray", aspect="auto")

            ax_handles: list[Line2D] = []
            inset_data: list[tuple[str, np.ndarray]] = []

            for name, pattern in side_result.patterns.items():
                if pattern.score <= min_score:
                    continue
                color = _PATTERN_COLORS.get(name, "white")
                score = pattern.score
                centers = np.array(pattern.points, dtype=np.float64)

                if len(centers) > 0:
                    centers_local = sub_img.to_local(centers)
                    ax.scatter(centers_local[:, 0], centers_local[:, 1], c=color, s=20, zorder=3)
                    ax_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=7,
                            label=(
                                f"{side} {name}  |  score={score:.3f}  |  fiducials={pattern.count}  |  "
                                f"{_spacing_info(centers[:, 0])}"
                            ),
                        )
                    )
                    mp = mean_patch_from_centers(src, centers)
                    if mp is not None:
                        inset_data.append((color, mp))

            x_edge = np.linspace(0, src.width, 500)
            edge_local = sub_img.to_local(np.column_stack([x_edge, np.polyval(side_result.edge_coefficients, x_edge)]))
            ax.plot(edge_local[:, 0], edge_local[:, 1], color="steelblue", linewidth=1.0, linestyle="--")

            ax.legend(handles=ax_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=15, frameon=True)
            ax.axis("off")

            for col, inset_ax in enumerate(inset_slots[row]):
                if col < len(inset_data):
                    color, patch = inset_data[col]
                    inset_ax.imshow(patch, cmap="gray")
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(2)
                    inset_ax.set_xticks([])
                    inset_ax.set_yticks([])
                else:
                    inset_ax.set_visible(False)

    _save(fig, output_file)


def plot_fiducial_detected_boxes(
    result: str | Path | FiducialDetectionResult,
    output_file: str | Path,
    side: Literal["top", "bottom"],
    overwrite: bool = False,
) -> None:
    """Every detected fiducial box of one side as a cropped patch.

    Boxes are colour-coded by pattern; unmatched detections appear in gray.
    """
    if skip_existing(output_file, overwrite):
        return
    result = FiducialDetectionResult.load(result)
    side_result = result.top if side == "top" else result.bottom
    cmap = plt.get_cmap("tab10")
    noise_color = (0.85, 0.85, 0.85, 1.0)

    coord_to_pattern: dict[tuple[float, float], tuple[str, Any]] = {}
    for i, (name, pattern) in enumerate(side_result.patterns.items()):
        color = cmap(i % 10)
        for pt in pattern.points:
            coord_to_pattern[(float(pt[0]), float(pt[1]))] = (name, color)

    n = len(side_result.boxes)
    grid = max(1, int(np.ceil(np.sqrt(n))))
    fig, axes_2d = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Detected fiducial boxes — {side}  ({n} boxes)", fontsize=11, fontweight="bold")
    axes = axes_2d.flatten()

    with rasterio.open(result.raster_filepath) as src:
        for ax, box, score, (cx, cy) in zip(axes, side_result.boxes, side_result.scores, side_result.centers_xy):
            x, y, w, h = box
            ax.imshow(src.read(1, window=Window(x, y, w, h)), cmap="gray", interpolation="nearest")
            label, color = coord_to_pattern.get((float(cx), float(cy)), ("unmatched", noise_color))
            ax.set_title(f"{label}  {score:.3f}", fontsize=7, color=color)
            ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    _save(fig, output_file)


def mean_patch_from_centers(
    src: str | Path | rasterio.DatasetReader,
    centers: np.ndarray,
    half_size: int = 50,
) -> np.ndarray | None:
    """Compute the mean image patch (band 1) around a set of pixel centers.

    Uses an incremental float64 accumulator so peak memory is O(patch_size²)
    regardless of the number of centers. Out-of-bounds regions are zero-padded
    before averaging. Centers that fall entirely outside the raster are silently skipped.
    """
    if not isinstance(src, rasterio.DatasetReader):
        with rasterio.open(src) as opened:
            return mean_patch_from_centers(opened, centers, half_size)

    size = 2 * half_size
    accumulator = np.zeros((size, size), dtype=np.float64)
    count = 0

    x0s: np.ndarray = centers[:, 0].astype(np.intp) - half_size
    y0s: np.ndarray = centers[:, 1].astype(np.intp) - half_size

    for x0, y0 in zip(x0s, y0s):
        x0c = max(0, int(x0))
        y0c = max(0, int(y0))
        x1c = min(src.width, int(x0) + size)
        y1c = min(src.height, int(y0) + size)
        if x1c <= x0c or y1c <= y0c:
            continue
        patch = src.read(1, window=Window(x0c, y0c, x1c - x0c, y1c - y0c))
        accumulator[y0c - y0 : y0c - y0 + patch.shape[0], x0c - x0 : x0c - x0 + patch.shape[1]] += patch
        count += 1

    return (accumulator / count).astype(np.float32) if count > 0 else None


# --- Transform ---


def plot_deformation_grid(
    transform: str | Path | Transformation,
    output_file: str | Path,
    num: int = 20,
    figsize: tuple[int, int] = (6, 6),
    overwrite: bool = False,
) -> None:
    """Visualize the deformation field by plotting warped grid lines."""
    if skip_existing(output_file, overwrite):
        return
    transform = Transformation.load(transform)
    with rasterio.open(transform.raster_filepath) as src:
        w, h = src.width, src.height

    xs = np.linspace(0, w - 1, num, dtype=np.float32)
    ys = np.linspace(0, h - 1, num, dtype=np.float32)

    fig, ax = plt.subplots(figsize=figsize)

    for y in ys:
        line = np.stack([xs, np.full_like(xs, y)], axis=-1)
        warped_line = transform.deformation(line)
        ax.plot(warped_line[:, 0], warped_line[:, 1], color="gray", lw=0.8, alpha=0.7)

    for x in xs:
        line = np.stack([np.full_like(ys, x), ys], axis=-1)
        warped_line = transform.deformation(line)
        ax.plot(warped_line[:, 0], warped_line[:, 1], color="gray", lw=0.8, alpha=0.7)

    ax.set_title("Warped deformation grid")
    ax.invert_yaxis()

    _save(fig, output_file)


def plot_crop_area(
    transform: str | Path | Transformation,
    output_file: str | Path,
    figsize: tuple[int, int] = (6, 6),
    overwrite: bool = False,
) -> None:
    """Visualize the crop region within the original image frame."""
    if skip_existing(output_file, overwrite):
        return
    transform = Transformation.load(transform)
    fig, ax = plt.subplots(figsize=figsize)

    with rasterio.open(transform.raster_filepath) as src:
        w, h = src.width, src.height

    crop_x, crop_y = transform.crop_offset
    crop_w, crop_h = transform.output_size

    ax.add_patch(patches.Rectangle((0, 0), w, h, fill=False, edgecolor="black", linewidth=2, label="Original image"))
    ax.add_patch(
        patches.Rectangle((crop_x, crop_y), crop_w, crop_h, fill=True, alpha=0.3, color="orange", label="Crop region")
    )
    ax.scatter(crop_x, crop_y, color="red", marker="+", s=10, label="Crop origin (0,0 in crop space)")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("auto")
    ax.set_box_aspect(h / (w / 2))
    ax.invert_yaxis()
    ax.set_title(f"Crop visualization\ncrop_offset = ({crop_x}, {crop_y}), size = ({crop_w}, {crop_h})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    _save(fig, output_file)


def plot_georeferencing(
    raster_filepath: str | Path, output_file: str | Path, width: int = 1000, overwrite: bool = False
) -> None:
    """Raster warped from its GCPs over ESRI World Imagery, opaque then 50% transparent."""
    if skip_existing(output_file, overwrite):
        return
    with rasterio.open(raster_filepath) as src:
        scale = width / src.width
        band = src.read(1, out_shape=(round(src.height * scale), width), resampling=Resampling.average)
        gcps, gcp_crs = src.gcps

    # GDAL rejects GCPs with z=None
    gcps = [GroundControlPoint(row=g.row * scale, col=g.col * scale, x=g.x, y=g.y, z=0.0) for g in gcps]
    dst_transform, w, h = calculate_default_transform(gcp_crs, "EPSG:3857", band.shape[1], band.shape[0], gcps=gcps)
    warped = np.zeros((h, w), dtype=band.dtype)
    reproject(
        band,
        warped,
        gcps=gcps,
        src_crs=gcp_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        src_nodata=0,
        dst_nodata=0,
    )
    extent = plotting_extent(warped, dst_transform)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, alpha in zip(axes, (1.0, 0.5)):
        ax.imshow(np.ma.masked_equal(warped, 0), extent=extent, cmap="gray", alpha=alpha, zorder=2)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zorder=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"alpha={alpha}")
        ax.axis("off")

    fig.suptitle(Path(raster_filepath).stem, fontsize=12, fontweight="bold")
    _save(fig, output_file)
