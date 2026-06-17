from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.restitution.fiducial import FiducialStrategy


_PATTERN_COLORS: dict[str, str] = {
    "RegularSparse": "red",
    "RegularMid": "orange",
    "RegularDense": "gold",
    "SegmentedMid": "limegreen",
    "SegmentedDense": "cyan",
    "SerializedTimeWord": "violet",
}


def _coord_index(centers_xy: np.ndarray) -> dict[tuple[float, float], int]:
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

        for i, pattern in enumerate(result.patterns):
            if pattern.count == 0:
                continue
            color = cmap(i % 10)
            score = pattern.final_score
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
                    label=f"{type(pattern).__name__}{star}  score={score:.3f}  n={pattern.count}",
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
        for pattern in result.patterns:
            if pattern.final_score <= detector.min_score_threshold or pattern.count < 8:
                continue
            x = pattern.points[:, 0].astype(np.float64)
            y = pattern.points[:, 1].astype(np.float64)
            ax.scatter(
                x,
                y - y.mean(),
                s=8,
                marker="x" if side == "bottom" else "o",
                label=f"{side} · {type(pattern).__name__}  (n={len(x)})",
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
        max(sum(1 for p in r.patterns if p.final_score > detector.min_score_threshold) for r in sides_results),
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

            for pattern in result.patterns:
                if pattern.final_score <= detector.min_score_threshold:
                    continue
                color = _PATTERN_COLORS.get(type(pattern).__name__, "white")
                score = pattern.final_score
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
                            label=f"{side} {type(pattern).__name__}  |  score={score:.3f}  |  fiducials={pattern.count}  |  {_spacing_info(centers[:, 0])}",
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
        for i, pattern in enumerate(side_result.patterns):
            color = cmap(i % 10)
            for pt in pattern.points:
                coord_to_pattern[(float(pt[0]), float(pt[1]))] = (type(pattern).__name__, color)

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
