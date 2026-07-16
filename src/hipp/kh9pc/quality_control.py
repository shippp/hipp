"""
Copyright (c) 2026 HIPP developers
Description: Quality control figures for all KH-9 PC restitution strategies.
    Each ``plot_*`` function targets one fitted object and returns a matplotlib Figure.
    ``get_figures`` dispatches to the right set of plots based on the strategy type,
    and ``save_figures`` writes them all to disk.
"""

import logging
from pathlib import Path

from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rasterio.warp import Resampling
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.restitution.base import FittingClass, Transformation
from hipp.kh9pc.restitution.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution.fiducial_strategy import FiducialStrategy
from hipp.kh9pc.restitution.flat_strategy import FlatStrategy
from hipp.kh9pc.restitution.mixed_strategy import MixedStrategy
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector

logger = logging.getLogger(__name__)


# --- Vertical ---
def plot_vertical_ruptures(detector: VerticalDetector) -> Figure:
    """Band profiles with detected rupture positions for left and right edges."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["left", "right"], [detector.left_, detector.right_]):
        ax.plot(result.profile, color="gray")
        ax.axvline(x=result.edge_local, color="red", label=f"edge (local={result.edge_local})")
        ax.set_title(f"{side} profile \n(global col={result.position}, ratio={result.gradient_ratio:.2f})")
        ax.set_xlabel("local column index")
        ax.set_ylabel("intensity")
        ax.legend()

    fig.suptitle(detector.raster_filepath_.stem, fontsize=12, fontweight="bold")
    return fig


def plot_vertical_edges(
    detector: VerticalDetector,
    window_width: int = 10000,
    out_height: int = 800,
) -> Figure:
    """Thumbnails around the left and right edge positions."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(detector.raster_filepath_) as src:
        for ax, side, edge_col in zip(axes, ["left", "right"], detector.edges_):
            window = Window(edge_col - window_width // 2, 0, window_width, src.height)
            scale = out_height / window.height
            out_shape = (1, out_height, int(window.width * scale))
            sub_image = SubImage(src, window=window, out_shape=out_shape)

            ax.imshow(sub_image.band, cmap="gray", aspect="auto")

            ax.axvline(x=sub_image.to_local_x(edge_col), color="red")
            ax.set_title(f"{side} edge (col={edge_col})")
            ax.axis("off")

    fig.suptitle(detector.raster_filepath_.stem, fontsize=12, fontweight="bold")
    return fig


# --- Flat ---
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
    left, _ = detector.vertical_detector.edges_
    roi_w = detector.vertical_detector.detected_width_

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


# --- Poly ---


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
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


# --- Collimation ---


def plot_collimation_edges(detector: CollimationStrategy) -> Figure:
    """Subimage thumbnails with RANSAC inliers/outliers and polynomial model for top and bottom collimation lines."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
        ax.imshow(result.sub_image.band, cmap="gray", aspect="auto")

        inliers = result.model.inlier_mask_
        peaks = result.peaks_local
        ax.scatter(peaks[~inliers, 0], peaks[~inliers, 1], s=12, c="red", label="outliers")
        ax.scatter(peaks[inliers, 0], peaks[inliers, 1], s=12, c="green", label="inliers")

        y_global_pred = result.model.predict(result.peaks_global[:, 0].reshape(-1, 1))
        global_pred = np.column_stack([result.peaks_global[:, 0], y_global_pred])
        local_pred = result.sub_image.to_local(global_pred)
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

    ax.invert_yaxis()
    ax.legend()
    ax.set_title("global distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


# --- Fiducial ---


_PATTERN_COLORS: dict[str, str] = {
    "regulare_sparse": "red",
    "regulare_mid": "orange",
    "regular_dense": "gold",
    "segmented_mid": "limegreen",
    "segmented_dense": "cyan",
    "serialized_time_word": "violet",
}


def _coord_index(centers_xy: np.ndarray) -> dict[tuple[float, float], int]:
    """Build a (cx, cy) → row-index lookup for fast pattern membership queries."""
    return {(float(cx), float(cy)): i for i, (cx, cy) in enumerate(centers_xy)}


def plot_fiducial_filtering(detector: FiducialStrategy) -> Figure:
    """Pattern detection diagnostics: spatial scatter and feature space for top and bottom sides.

    Each row corresponds to one side (top / bottom). The left column shows detections in
    global image space (cx vs cy) with the fitted polynomial edge overlaid. The right column
    shows the raw feature space (matching score vs residual to the edge model).

    Valid patterns are highlighted with distinct colours; unmatched detections appear in light gray.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Fiducial pattern detection ({detector.raster_filepath_.stem})", fontsize=12, fontweight="bold")

    sides = ("top", "bottom")
    results = (detector.top_, detector.bottom_)
    edge_models = [detector.poly_strategy.top_.model, detector.poly_strategy.bottom_.model]
    cmap = plt.get_cmap("tab10")
    _noise = (0.85, 0.85, 0.85, 1.0)

    for row, (side, result, edge_model) in enumerate(zip(sides, results, edge_models)):
        ax_spatial, ax_feat = axes[row]

        centers_xy = result.centers_xy
        features = result.features
        coord_idx = _coord_index(centers_xy)

        ax_spatial.scatter(centers_xy[:, 0], centers_xy[:, 1], c=[_noise], s=10, linewidths=0)
        ax_feat.scatter(features[:, 0], features[:, 1], c=[_noise], s=10, linewidths=0)

        legend_handles: list[Line2D] = []

        for i, (name, pattern) in enumerate(result.patterns.items()):
            if pattern.count == 0:
                continue
            color = cmap(i % 10)
            score = pattern.score
            star = " ★" if score > detector.min_score_threshold else ""

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
        y_pred = edge_model.predict(x_grid.reshape(-1, 1)).ravel()
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

    return fig


def plot_fiducial_distortions(detector: FiducialStrategy) -> Figure:
    """Fiducial center y-deviation from mean, per valid pattern, for top and bottom sides.

    In the ideal case all points lie at 0. Divergence reveals scan distortion.
    """
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"Fiducial distortion — {detector.raster_filepath_.stem}", fontsize=12, fontweight="bold")

    for side, result in zip(["top", "bottom"], [detector.top_, detector.bottom_]):
        for name, pattern in result.patterns.items():
            if pattern.score <= detector.min_score_threshold or pattern.count < 8:
                continue
            x = pattern.points[:, 0].astype(np.float64)
            y = pattern.points[:, 1].astype(np.float64)
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

    return fig


def plot_fiducial_detected_profiles(detector: FiducialStrategy, window_height_fraction: float = 0.08) -> Figure:
    """Detected fiducial centers overlaid on the top and bottom image strips, one scatter per valid pattern."""
    sides_results = [detector.top_, detector.bottom_]
    n_insets = max(
        max(sum(1 for p in r.patterns.values() if p.score > detector.min_score_threshold) for r in sides_results),
        1,
    )

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Fiducial detected profiles — {detector.raster_filepath_.stem}", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(2, 1 + n_insets, width_ratios=[14] + [1] * n_insets)
    main_axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
    inset_slots = [[fig.add_subplot(gs[row, col + 1]) for col in range(n_insets)] for row in range(2)]

    with rasterio.open(detector.raster_filepath_) as src:
        window_height = int(src.height * window_height_fraction)
        windows = [
            Window(0, 0, src.width, window_height),
            Window(0, src.height - window_height, src.width, window_height),
        ]
        edge_models = [detector.poly_strategy.top_.model, detector.poly_strategy.bottom_.model]

        def _spacing_info(cx: np.ndarray) -> str:
            if len(cx) >= 2:
                dists = np.diff(np.sort(cx.astype(np.float64)))
                return f"spacing  mean={float(dists.mean()):.1f}px  std={float(dists.std()):.1f}px"
            return "spacing  n/a"

        for row, (ax, side, window, result, edge_model) in enumerate(
            zip(main_axes, ["top", "bottom"], windows, sides_results, edge_models)
        ):
            sub_img = SubImage(src, window, (1, 512, 4096))
            ax.imshow(sub_img.band, cmap="gray", aspect="auto")

            ax_handles: list[Line2D] = []
            inset_data: list[tuple[str, np.ndarray]] = []

            for name, pattern in result.patterns.items():
                if pattern.score <= detector.min_score_threshold:
                    continue
                color = _PATTERN_COLORS.get(name, "white")
                score = pattern.score
                centers = pattern.points.astype(np.float64)

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
                            label=f"{side} {name}  |  score={score:.3f}  |  fiducials={pattern.count}  |  {_spacing_info(centers[:, 0])}",
                        )
                    )
                    mp = mean_patch_from_centers(src, centers)
                    if mp is not None:
                        inset_data.append((color, mp))

            x_edge = np.linspace(0, src.width, 500)
            edge_local = sub_img.to_local(np.column_stack([x_edge, edge_model.predict(x_edge.reshape(-1, 1)).ravel()]))
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

    return fig


def plot_fiducial_detected_boxes(detector: FiducialStrategy) -> tuple[Figure, Figure]:
    """One figure per side showing every detected fiducial box as a cropped patch.

    Boxes are colour-coded by pattern; unmatched detections appear in gray.
    """
    figures: list[Figure] = []
    cmap = plt.get_cmap("tab10")

    for side, side_result in zip(("top", "bottom"), (detector.top_, detector.bottom_)):
        boxes = side_result.boxes
        scores = side_result.scores
        centers_xy = side_result.centers_xy
        n = len(boxes)

        coord_to_pattern: dict[tuple[float, float], tuple[str, Any]] = {}
        for i, (name, pattern) in enumerate(side_result.patterns.items()):
            color = cmap(i % 10)
            for pt in pattern.points:
                coord_to_pattern[(float(pt[0]), float(pt[1]))] = (name, color)

        _noise_color = (0.85, 0.85, 0.85, 1.0)

        grid = max(1, int(np.ceil(np.sqrt(n))))
        fig, axes_2d = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Detected fiducial boxes — {side}  ({n} boxes)", fontsize=11, fontweight="bold")
        axes = axes_2d.flatten()

        with rasterio.open(detector.raster_filepath_) as src:
            for ax, box, score, (cx, cy) in zip(axes, boxes, scores, centers_xy):
                x, y, w, h = box
                band = src.read(1, window=Window(x, y, w, h))
                ax.imshow(band, cmap="gray", interpolation="nearest")

                match = coord_to_pattern.get((float(cx), float(cy)))
                if match is not None:
                    pattern_name, color = match
                    label_str = pattern_name
                else:
                    color = _noise_color
                    label_str = "unmatched"

                ax.set_title(f"{label_str}  {score:.3f}", fontsize=7, color=color)
                ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        figures.append(fig)

    return figures[0], figures[1]


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
    transform: Transformation,
    num: int = 20,
    figsize: tuple[int, int] = (6, 6),
) -> Figure:
    """Visualize the deformation field by plotting warped grid lines."""
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

    return fig


def plot_crop_area(transform: Transformation, figsize: tuple[int, int] = (6, 6)) -> Figure:
    """Visualize the crop region within the original image frame."""
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

    return fig


# --- Dispatch ---


def save_figures(fitting_class: FittingClass, output_dir: str | Path) -> None:
    """Save all QC figures for a fitted strategy to ``output_dir/<figure_name>/<stem>.png``.

    Each figure type gets its own sub-directory named after the plot (e.g.
    ``poly_edges/``, ``deformation_grid/``). Errors on individual figures are
    logged as warnings so a single bad plot never aborts the whole QC run.
    """
    output_dir = Path(output_dir)
    gen = get_figures(fitting_class)
    while True:
        try:
            name, fig = next(gen)
            (output_dir / name).mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / name / f"{fitting_class.raster_filepath_.stem}.png")
            plt.close(fig)
        except StopIteration:
            break
        except Exception as e:
            logger.warning("Skipping QC figure: %s", e)


def get_figures(fitting_class: FittingClass, plot_transformation: bool = True) -> Iterator[tuple[str, Figure]]:
    """Yield (name, figure) pairs for all QC plots of a fitted FittingClass instance."""
    if isinstance(fitting_class, VerticalDetector):
        yield "vertical_edges", plot_vertical_edges(fitting_class)
        yield "vertical_ruptures", plot_vertical_ruptures(fitting_class)
        return
    if isinstance(fitting_class, FlatStrategy):
        yield from get_figures(fitting_class.vertical_detector, plot_transformation=False)
        yield "flat_edges", plot_flat_edges(fitting_class)
        yield "flat_ruptures", plot_flat_ruptures(fitting_class)
        if plot_transformation:
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, PolyStrategy):
        yield from get_figures(fitting_class.vertical_detector, plot_transformation=False)
        yield "poly_edges", plot_poly_edges(fitting_class)
        yield "poly_distortions", plot_poly_distortions(fitting_class)
        if plot_transformation:
            yield "deformation_grid", plot_deformation_grid(fitting_class.transformation_)
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, CollimationStrategy):
        yield from get_figures(fitting_class.poly_strategy, plot_transformation=False)
        yield "collimation_edges", plot_collimation_edges(fitting_class)
        yield "collimation_distortions", plot_collimation_distortions(fitting_class)
        if plot_transformation:
            yield "deformation_grid", plot_deformation_grid(fitting_class.transformation_)
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, FiducialStrategy):
        yield from get_figures(fitting_class.poly_strategy, plot_transformation=False)
        yield "fiducial_filtering", plot_fiducial_filtering(fitting_class)
        yield "fiducial_distortions", plot_fiducial_distortions(fitting_class)
        yield "fiducial_detected_profiles", plot_fiducial_detected_profiles(fitting_class)
        # yield from zip(("fiducial_boxes_top", "fiducial_boxes_bottom"), plot_fiducial_detected_boxes(fitting_class))
        if plot_transformation:
            yield "deformation_grid", plot_deformation_grid(fitting_class.transformation_)
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, MixedStrategy):
        yield from get_figures(fitting_class.selected_strategy_, plot_transformation=plot_transformation)
