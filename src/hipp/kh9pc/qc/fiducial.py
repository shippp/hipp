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
    global image space (cx vs cy) with the fitted polynomial edge overlaid. The right column
    shows the feature space (matching score vs residual to the edge model) used by DBSCAN.

    Each DBSCAN cluster gets a distinct colour; noise (label -1) is shown in light gray.
    The legend lists every cluster with its size, pattern score, a ★ for the best-scoring
    cluster, and a ✓ for clusters above the min_score_threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Fiducial outlier filtering ({detector.raster_filepath_.stem})", fontsize=12, fontweight="bold")

    sides = ("top", "bottom")
    results = (detector.top_, detector.bottom_)
    cmap = plt.get_cmap("tab10")

    for row, (side, result) in enumerate(zip(sides, results)):
        ax_spatial, ax_feat = axes[row]

        clustering = result.clustering
        centers_xy = result.centers_xy
        labels = clustering.labels
        cluster_df = clustering.cluster_df

        unique_labels = sorted(int(lbl) for lbl in np.unique(labels) if lbl != -1)
        label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}
        legend_labels = unique_labels[:5]
        _noise = (0.85, 0.85, 0.85, 1.0)
        colours = np.array([label_to_color[int(lbl)] if lbl != -1 else _noise for lbl in labels])

        ax_spatial.scatter(centers_xy[:, 0], centers_xy[:, 1], c=colours, s=20, linewidths=0)

        score_by_label = dict(zip(cluster_df["label"].tolist(), cluster_df["score"].tolist()))
        pattern_by_label = dict(zip(cluster_df["label"].tolist(), cluster_df["pattern"].tolist()))
        good_labels_set = set(cluster_df.loc[cluster_df["is_good"], "label"].tolist())
        best_label = int(cluster_df.loc[cluster_df["score"].idxmax(), "label"]) if not cluster_df.empty else -1

        inlier_mask = np.isin(labels, list(good_labels_set))
        legend_handles: list[Line2D] = []
        for lbl in legend_labels:
            if lbl in good_labels_set:
                spatial_score = score_by_label.get(lbl, float("nan"))
                pattern = pattern_by_label.get(lbl)
                star = " ★" if lbl == best_label else ""
                label_text = f"{pattern}{star}  s={spatial_score:.3f}"
            else:
                label_text = f"cluster {lbl}"
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=label_to_color[lbl],
                    markersize=6,
                    label=label_text,
                )
            )
        noise_n = int((labels == -1).sum())
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_noise, markersize=6, label=f"noise  n={noise_n}"),
        )
        ax_feat.legend(
            handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, borderaxespad=0
        )
        ax_spatial.invert_yaxis()

        n_inliers = int(inlier_mask.sum())
        ax_spatial.set_title(
            f"{side} — spatial  (eps={clustering.eps:.2f}, w={clustering.weight:.2f})\ninliers={n_inliers}"
        )
        ax_spatial.set_xlabel("cx (px)")
        ax_spatial.set_ylabel("cy (px)")

        ax_feat.scatter(clustering.features[:, 0], clustering.features[:, 1], c=colours, s=20, linewidths=0)
        ax_feat.set_title(f"{side} — feature space")
        ax_feat.set_xlabel("score")
        ax_feat.set_ylabel("residual (px)")

    return fig


def plot_fiducial_distortions(detector: FiducialStrategy) -> Figure:
    """Degree-7 polynomial fit per cluster, normalized by its mean, for top and bottom sides.

    Each good cluster gets a degree-7 polynomial fitted to its fiducial centers, evaluated on a
    common x-grid and mean-centered (``y − mean(y)``). In the ideal case all curves coincide at 0.
    Divergence between curves reveals scan distortion that differs across the fiducial strip.
    """
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"Fiducial distortion — {detector.raster_filepath_.stem}", fontsize=12, fontweight="bold")

    for side, result in zip(["top", "bottom"], [detector.top_, detector.bottom_]):
        clustering = result.clustering
        centers_xy = result.centers_xy
        good_df = clustering.cluster_df[clustering.cluster_df["is_good"]]

        for _, row in good_df.iterrows():
            label = int(row["label"])
            pattern = str(row["pattern"]) if row["pattern"] is not None else f"cluster {label}"
            mask = clustering.labels == label
            inlier_centers = centers_xy[mask]
            if len(inlier_centers) < 8:
                continue
            x = inlier_centers[:, 0].astype(np.float64)
            y = inlier_centers[:, 1].astype(np.float64)
            ax.scatter(
                x,
                y - y.mean(),
                s=8,
                marker="x" if side == "bottom" else "o",
                label=f"{side} · {pattern}  (n={len(x)})",
            )

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


def plot_fiducial_detected_profiles(detector: FiducialStrategy, window_height_fraction: float = 0.08) -> Figure:
    """Detected fiducial centers overlaid on the top and bottom image strips, one scatter per valid pattern."""
    _PATTERN_COLORS: dict[str | None, str] = {
        "regular_sparse": "red",
        "regular_mid": "orange",
        "regular_dense": "gold",
        "segmented_mid": "limegreen",
        "segmented_dense": "cyan",
        "serialized_time_word": "violet",
    }

    sides_results = [detector.top_, detector.bottom_]
    n_insets = max(
        max(int(r.clustering.cluster_df["is_good"].sum()) for r in sides_results),
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

            clustering = result.clustering
            centers_xy = result.centers_xy

            good_df = clustering.cluster_df[clustering.cluster_df["is_good"]].reset_index(drop=True)

            ax_handles: list[Line2D] = []
            inset_data: list[tuple[str, np.ndarray]] = []

            for _, cluster_row in good_df.iterrows():
                label = int(cluster_row["label"])
                pattern = cluster_row["pattern"]
                score = float(cluster_row["score"])
                color = _PATTERN_COLORS.get(pattern, "white")

                mask = clustering.labels == label
                cluster_centers = centers_xy[mask]
                if len(cluster_centers) > 0:
                    centers_local = sub_img.to_local(cluster_centers.astype(np.float64))
                    ax.scatter(centers_local[:, 0], centers_local[:, 1], c=color, s=20, zorder=3)

                    ax_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=7,
                            label=f"{side} {pattern}  |  score={score:.3f}  |  fiducials={int(mask.sum())}  |  {_spacing_info(cluster_centers[:, 0])}",
                        )
                    )

                    mp = mean_patch_from_centers(src, cluster_centers.astype(np.float64))
                    if mp is not None:
                        inset_data.append((color, mp))

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
    """One figure per side showing every detected fiducial box as a cropped patch.

    Boxes are colour-coded by DBSCAN cluster label; noise (label -1) appears in gray.
    The subplot title shows the pattern name (if the cluster is scored) and matching score.
    """
    figures: list[Figure] = []
    cmap = plt.get_cmap("tab10")

    for side, side_result in zip(("top", "bottom"), (detector.top_, detector.bottom_)):
        boxes = side_result.boxes
        scores = side_result.scores
        clustering = side_result.clustering
        labels = clustering.labels
        cluster_df = clustering.cluster_df
        n = len(boxes)

        pattern_by_label = dict(zip(cluster_df["label"].tolist(), cluster_df["pattern"].tolist()))
        unique_labels = sorted(int(lbl) for lbl in np.unique(labels) if lbl != -1)
        label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}

        grid = max(1, int(np.ceil(np.sqrt(n))))
        fig, axes_2d = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Detected fiducial boxes — {side}  ({n} boxes)", fontsize=11, fontweight="bold")
        axes = axes_2d.flatten()

        with rasterio.open(detector.raster_filepath_) as src:
            for ax, box, score, label in zip(axes, boxes, scores, labels):
                x, y, w, h = box
                band = src.read(1, window=Window(x, y, w, h))
                ax.imshow(band, cmap="gray", interpolation="nearest")

                color = label_to_color.get(int(label), (0.85, 0.85, 0.85, 1.0))
                pattern = pattern_by_label.get(int(label), "noise" if label == -1 else None)
                label_str = pattern if pattern else f"lbl={label}"
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
