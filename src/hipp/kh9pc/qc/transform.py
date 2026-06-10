import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patches
from matplotlib.figure import Figure

from hipp.kh9pc.restitution.base import Transformation


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
