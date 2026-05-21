import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patches
from matplotlib.figure import Figure
from rasterio.warp import Resampling
from rasterio.windows import Window

from hipp.kh9pc.restitution_strategy.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution_strategy.fiducial_strategy import FiducialStrategy
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
        ax.set_title(f"{side} band profile (global col={result.position})\nGradient:{result.gradient_pct:.2%}")
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
# CollimationStrategy
# ---------------------------------------------------------------------------


def plot_fiducial_filtering(detector: FiducialStrategy) -> Figure:
    """Outlier filtering diagnostics: spatial scatter and feature space for top and bottom sides.

    Each row corresponds to one side (top / bottom). The left column shows detections in
    global image space (cx vs cy) with the fitted polynomial edge overlaid. The right column
    shows the feature space (matching score vs residual to the edge model) used by DBSCAN.

    In both columns points are coloured by their cluster label:
    - green  — inlier cluster selected as the fiducial strip
    - orange — other DBSCAN clusters (false positives kept together)
    - gray   — DBSCAN noise (label -1)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Fiducial outlier filtering ({detector.raster_filepath_.stem})", fontsize=12, fontweight="bold")

    sides = ("top", "bottom")
    results = (detector.top_, detector.bottom_)
    models = (detector.poly_strategy.top_.model, detector.poly_strategy.bottom_.model)

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

        # build a colour array: green=inlier, orange=other cluster, gray=noise
        colours = np.where(
            labels == filtering.best_cluster_label,
            "green",
            np.where(labels == -1, "lightgray", "orange"),
        )

        # --- left: spatial scatter (cx, cy) ---
        ax_spatial.scatter(cx, cy, c=colours, s=20, linewidths=0)

        # overlay the polynomial edge model
        x_range = np.linspace(cx.min(), cx.max(), 300).reshape(-1, 1)
        y_pred = model.predict(x_range).ravel()
        ax_spatial.plot(x_range, y_pred, color="steelblue", linewidth=1.2, label="poly model")

        # legend proxies
        ax_spatial.scatter([], [], c="green", s=20, label="inliers")
        ax_spatial.scatter([], [], c="orange", s=20, label="other cluster")
        ax_spatial.scatter([], [], c="lightgray", s=20, label="noise")
        ax_spatial.legend(loc="best", fontsize=7)
        ax_spatial.invert_yaxis()
        ax_spatial.set_title(f"{side} — spatial  (eps={filtering.best_eps:.2f}, w={filtering.best_weight:.2f})")
        ax_spatial.set_xlabel("cx (px)")
        ax_spatial.set_ylabel("cy (px)")

        # --- right: feature space (score vs residual) ---
        ax_feat.scatter(scores, residuals, c=colours, s=20, linewidths=0)
        ax_feat.set_title(f"{side} — feature space")
        ax_feat.set_xlabel("matching score")
        ax_feat.set_ylabel("residual to poly (px)")

    return fig


def plot_fiducal_detected_profiles(detector: FiducialStrategy) -> Figure:
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), constrained_layout=True)
    fig.suptitle("Fiducial detected profiles", fontsize=12, fontweight="bold")

    for ax, side, side_result in zip(axes, ("top", "bottom"), (detector.top_, detector.bottom_)):
        count = len(side_result.boxes)
        mean_score = float(np.mean(side_result.scores)) if count > 0 else float("nan")

        if count > 0:
            x, y = side_result.boxes[:, 0], side_result.boxes[:, 1]
            ax.scatter(x, y, marker="+")

        ax.set_title(f"{side}  —  {count} templates detected, mean score = {mean_score:.3f}")
        ax.set_xlabel("x (global px)")
        ax.set_ylabel("y (global px)")
        ax.invert_yaxis()

    return fig


def plot_fiducial_detected_boxs(detector: FiducialStrategy) -> tuple[Figure, Figure]:
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


def get_figures(fitting_class: FittingClass) -> list[Figure]:
    """All QC figures for a fitted fitting_class."""
    if isinstance(fitting_class, VerticalDetector):
        return [plot_vertical_edges(fitting_class), plot_vertical_ruptures(fitting_class)]
    if isinstance(fitting_class, FlatStrategy):
        return [
            *get_figures(fitting_class.vertical_detector),
            plot_flat_edges(fitting_class),
            plot_flat_ruptures(fitting_class),
            plot_crop_area(fitting_class.transformation_),
        ]
    if isinstance(fitting_class, PolyStrategy):
        return [
            *get_figures(fitting_class.vertical_detector),
            plot_poly_edges(fitting_class),
            plot_poly_distortions(fitting_class),
            plot_deformation_grid(fitting_class.transformation_),
            plot_crop_area(fitting_class.transformation_),
        ]
    if isinstance(fitting_class, CollimationStrategy):
        return [
            *get_figures(fitting_class.vertical_detector),
            plot_collimation_edges(fitting_class),
            plot_collimation_distortions(fitting_class),
            plot_deformation_grid(fitting_class.transformation_),
            plot_crop_area(fitting_class.transformation_),
        ]
    if isinstance(fitting_class, FiducialStrategy):
        return [
            *get_figures(fitting_class.poly_strategy),
            plot_fiducial_filtering(fitting_class),
            plot_fiducal_detected_profiles(fitting_class),
            plot_fiducial_detected_boxs(fitting_class)[0],
            plot_fiducial_detected_boxs(fitting_class)[1],
        ]
    if isinstance(fitting_class, MixedStrategy):
        return get_figures(fitting_class.selected_strategy_)

    return []
