from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.restitution.fiducial import FiducialStrategy


def plot_fiducial_filtering(detector: FiducialStrategy) -> Figure:
    """Outlier filtering diagnostics: spatial scatter and feature space for top and bottom sides.

    Each row corresponds to one side (top / bottom). The left column shows detections in
    global image space (cx vs cy) with the fitted polynomial edge and the fiducial polynomial
    overlaid. The right column shows the feature space (matching score vs residual to the edge
    model) used by DBSCAN.

    Each DBSCAN cluster gets a distinct colour; noise (label -1) is shown in light gray.
    The legend lists every cluster with its size, mean matching score, and a ★ for the
    selected inlier cluster.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Fiducial outlier filtering ({detector.raster_filepath_.stem})", fontsize=12, fontweight="bold")

    sides = ("top", "bottom")
    results = (detector.top_, detector.bottom_)
    models = (detector.poly_strategy.top_.model, detector.poly_strategy.bottom_.model)
    cmap = plt.get_cmap("tab10")

    for row, (side, result, model) in enumerate(zip(sides, results, models)):
        ax_spatial, ax_feat = axes[row]

        filtering = result.filtering
        if filtering is None:
            for ax in (ax_spatial, ax_feat):
                ax.set_visible(False)
            continue

        cx, cy = filtering.cx, filtering.cy
        labels = filtering.labels
        scores = filtering.scores_all
        residuals = filtering.residuals

        unique_labels = sorted(int(lbl) for lbl in np.unique(labels) if lbl != -1)
        label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}
        legend_labels = unique_labels[:5]
        _noise = (0.85, 0.85, 0.85, 1.0)
        colours = np.array([label_to_color[int(lbl)] if lbl != -1 else _noise for lbl in labels])

        ax_spatial.scatter(cx, cy, c=colours, s=20, linewidths=0)

        x_line = np.linspace(cx.min(), cx.max(), 300)
        ax_spatial.plot(
            x_line,
            model.predict(x_line.reshape(-1, 1)).ravel(),
            color="steelblue",
            linewidth=1.2,
            linestyle="--",
        )

        if len(result.centers) >= 2:
            ax_spatial.plot(x_line, result.poly(x_line), color="crimson", linewidth=1.5)

        inlier_mask = labels == filtering.best_cluster_label
        legend_handles: list[Line2D] = []
        for lbl in legend_labels:
            n = int((labels == lbl).sum())
            spatial_score = filtering.cluster_scores.get(lbl, float("nan"))
            star = " ★" if lbl == filtering.best_cluster_label else ""
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=label_to_color[lbl],
                    markersize=6,
                    label=f"cluster {lbl}{star}  n={n}  s={spatial_score:.3f}",
                )
            )
        noise_n = int((labels == -1).sum())
        legend_handles += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_noise, markersize=6, label=f"noise  n={noise_n}"),
            Line2D([0], [0], color="steelblue", linewidth=1.2, linestyle="--", label="edge model"),
            Line2D([0], [0], color="crimson", linewidth=1.5, label="fiducial poly"),
        ]
        ax_feat.legend(
            handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, borderaxespad=0
        )
        ax_spatial.invert_yaxis()

        n_inliers = int(inlier_mask.sum())
        ax_spatial.set_title(
            f"{side} — spatial  (eps={filtering.best_eps:.2f}, w={filtering.best_weight:.2f})\n"
            f"inliers={n_inliers} | coverage={result.width_coverage:.1%}"
        )
        ax_spatial.set_xlabel("cx (px)")
        ax_spatial.set_ylabel("cy (px)")

        ax_feat.scatter(scores, residuals, c=colours, s=20, linewidths=0)
        ax_feat.set_title(f"{side} — feature space")
        ax_feat.set_xlabel("matching score")
        ax_feat.set_ylabel("residual to edge model (px)")

    return fig


def plot_fiducial_distortions(detector: FiducialStrategy) -> Figure:
    """Residual distortion curves (deviation from mean) for top and bottom fiducial polynomial fits."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for side, result in zip(["top", "bottom"], [detector.top_, detector.bottom_]):
        ax.plot(result.distortion[:, 0], result.distortion[:, 1], label=side)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("fiducial distortion (top & bottom)")
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


def plot_fiducial_detected_profiles(detector: FiducialStrategy, window_height_fraction: float = 0.08) -> Figure:
    """Detected fiducial centers overlaid on the top and bottom image strips."""
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Fiducial detected profiles — {detector.raster_filepath_.stem}", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(2, 3, width_ratios=[14, 1, 1])
    main_axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
    inset_slots = [[fig.add_subplot(gs[row, col + 1]) for col in range(2)] for row in range(2)]

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
            zip(main_axes, ["top", "bottom"], windows, [detector.top_, detector.bottom_], edge_models)
        ):
            sub_img = SubImage(src, window, (1, 512, 4096))
            ax.imshow(sub_img.band, cmap="gray", aspect="auto")

            n = len(result.centers)
            if n > 0:
                centers_local = sub_img.to_local(result.centers.astype(np.float64))
                ax.scatter(centers_local[:, 0], centers_local[:, 1], c="red", s=20, zorder=3)

            ax_handles: list[Line2D] = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=7,
                    label=f"{side}  |  fiducials={n}  |  {_spacing_info(result.centers[:, 0]) if n > 0 else 'spacing  n/a'}",
                )
            ]
            inset_data: list[tuple[str, np.ndarray]] = []
            if n > 0:
                mp = mean_patch_from_centers(src, result.centers.astype(np.float64))
                if mp is not None:
                    inset_data.append(("red", mp))

            if side == "bottom" and result.filtering is not None:
                filtering = result.filtering
                second_candidates = sorted(
                    [(lbl, s) for lbl, s in filtering.cluster_scores.items() if lbl != filtering.best_cluster_label],
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                if second_candidates and second_candidates[0][1] > 0.9:
                    second_label, _ = second_candidates[0]
                    mask = filtering.labels == second_label
                    second_cx = filtering.cx[mask]
                    second_centers = np.column_stack([second_cx, filtering.cy[mask]])
                    second_local = sub_img.to_local(second_centers)
                    ax.scatter(second_local[:, 0], second_local[:, 1], c="orange", s=20, zorder=3)
                    ax_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor="orange",
                            markersize=7,
                            label=f"bottom 2nd cluster  |  fiducials={int(mask.sum())}  |  {_spacing_info(second_cx)}",
                        )
                    )
                    mp2 = mean_patch_from_centers(src, second_centers)
                    if mp2 is not None:
                        inset_data.append(("orange", mp2))

            x_edge = np.linspace(0, src.width, 500)
            edge_local = sub_img.to_local(np.column_stack([x_edge, edge_model.predict(x_edge.reshape(-1, 1)).ravel()]))
            ax.plot(edge_local[:, 0], edge_local[:, 1], color="steelblue", linewidth=1.0, linestyle="--")

            ax.legend(
                handles=ax_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.0),
                fontsize=15,
                frameon=True,
            )
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
    """One figure per side showing every detected fiducial box as a cropped patch."""
    figures: list[Figure] = []

    for side, side_result in zip(("top", "bottom"), (detector.top_, detector.bottom_)):
        boxes = side_result.boxes
        scores = side_result.scores
        template_ids = side_result.template_ids
        n = len(boxes)

        grid = max(1, int(np.ceil(np.sqrt(n))))
        fig, axes_2d = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Detected fiducial boxes — {side}  ({n} boxes)", fontsize=11, fontweight="bold")
        axes = axes_2d.flatten()

        with rasterio.open(detector.raster_filepath_) as src:
            for ax, box, score, tid in zip(axes, boxes, scores, template_ids):
                x, y, w, h = box
                band = src.read(1, window=Window(x, y, w, h))
                ax.imshow(band, cmap="gray", interpolation="nearest")
                ax.set_title(f"tpl={tid}  score={score:.3f}", fontsize=7)
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
