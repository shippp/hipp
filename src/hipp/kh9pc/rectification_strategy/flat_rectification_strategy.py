from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import rasterio

from hipp.kh9pc.rectification_strategy.base import RectificationStrategy
from hipp.kh9pc.rectification_strategy.vertical_edges_estimator import VerticalEdgesEstimator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from rasterio.windows import Window
from rasterio.warp import Resampling

from hipp.kh9pc.utils import detect_ruptures, make_summary_figure


@dataclass
class FlatResult:
    position: int
    rupture_local: int
    band: NDArray[np.integer]


class FlatRectificationStrategy(RectificationStrategy):
    """Simplest strategy: a single constant row for each of the top and bottom edges.

    Vertical boundaries are detected by :class:`~hipp.kh9pc.VerticalEdgesEstimator`.
    Horizontal boundaries are single-row ruptures (flat edges).

    Parameters
    ----------
    vertical_estimator:
        Estimator used to locate the left and right vertical boundaries.
        Defaults to a new :class:`VerticalEdgesEstimator` with default parameters.
    background_threshold:
        Intensity threshold below which a pixel is considered background.
    height_fraction:
        Fraction of image height to read on each side when probing.
    stride:
        Downsampling stride applied along the row axis when reading the band.
    img_height:
        Target height of the rectified image in pixels. If *None*, estimated
        as the distance between the detected top and bottom edges.
    """

    def __init__(
        self,
        vertical_estimator: VerticalEdgesEstimator | None = None,
        background_threshold: int = 20,
        height_fraction: float = 0.15,
        stride: int = 10,
        img_height: int | None = None,
    ):
        self.vertical_estimator = vertical_estimator if vertical_estimator is not None else VerticalEdgesEstimator()
        self.background_threshold = background_threshold
        self.height_fraction = height_fraction
        self.stride = stride
        self.img_height = img_height
        self.top: FlatResult | None = None
        self.bottom: FlatResult | None = None
        self.raster_filepath_: Path | None = None
        self.vertical_edges_: tuple[int, int] | None = None

    def fit(self, raster_filepath: str | Path) -> "FlatRectificationStrategy":
        self.raster_filepath_ = Path(raster_filepath)
        self.vertical_estimator.fit(raster_filepath)
        self.vertical_edges_ = self.vertical_estimator.edges

        with rasterio.open(raster_filepath) as src:
            window_width = self.vertical_edges_[1] - self.vertical_edges_[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, 1)

            for side, window in {
                "top": Window(self.vertical_edges_[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges_[0], src.height - window_height, window_width, window_height),
            }.items():
                band = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.average)
                setattr(self, side, self._process_side(band, window, side))

        return self

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        assert self.top is not None and self.bottom is not None and self.vertical_edges_ is not None
        left, right = self.vertical_edges_
        output_width = right - left
        output_height = self.img_height if self.img_height is not None else self.bottom.position - self.top.position

        # 4-corner grid (2x2) — pure translation/crop, no distortion correction
        src_points = np.array(
            [
                [left, self.top.position],
                [left, self.bottom.position],
                [right, self.top.position],
                [right, self.bottom.position],
            ],
            dtype=float,
        )
        dst_points = np.array(
            [
                [0, 0],
                [0, output_height],
                [output_width, 0],
                [output_width, output_height],
            ],
            dtype=float,
        )
        return src_points, dst_points, (output_width, output_height)

    def generate_qc_report(self, output_path: str | Path) -> None:
        if self.raster_filepath_ is None or self.top is None or self.bottom is None or self.vertical_edges_ is None:
            raise RuntimeError("Call fit() before generate_qc_report()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        left, right = self.vertical_edges_

        summary_lines = [
            "FlatRectificationStrategy — QC Report",
            "",
            f"Image              : {self.raster_filepath_.name}",
            "",
            "Detected edges",
            f"  Vertical         : left={left} px,  right={right} px",
            f"  Horizontal       : top={self.top.position} px,  bottom={self.bottom.position} px",
            "",
            "Parameters",
            f"  background_threshold : {self.background_threshold}",
            f"  height_fraction      : {self.height_fraction}",
            f"  stride               : {self.stride}",
        ]

        with PdfPages(output_path) as pdf:
            summary_fig = make_summary_figure(summary_lines)
            pdf.savefig(summary_fig)
            plt.close(summary_fig)

            vert_edges_fig = self.vertical_estimator.plot_edges(self.raster_filepath_)
            pdf.savefig(vert_edges_fig)
            plt.close(vert_edges_fig)

            vert_rupt_fig = self.vertical_estimator.plot_ruptures()
            pdf.savefig(vert_rupt_fig)
            plt.close(vert_rupt_fig)

            horiz_fig = self._plot_horizontal_edges()
            pdf.savefig(horiz_fig)
            plt.close(horiz_fig)

    def _plot_horizontal_edges(self) -> Figure:
        assert self.raster_filepath_ is not None and self.top is not None
        assert self.bottom is not None and self.vertical_edges_ is not None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        (ax_top_img, ax_bot_img), (ax_top_prof, ax_bot_prof) = axes

        with rasterio.open(self.raster_filepath_) as src:
            img_h = src.height
            left, right = self.vertical_edges_
            roi_w = right - left
            margin = int(0.03 * img_h)

            for ax_img, ax_prof, side, result in [
                (ax_top_img, ax_top_prof, "top", self.top),
                (ax_bot_img, ax_bot_prof, "bottom", self.bottom),
            ]:
                row_off = max(0, result.position - margin)
                row_end = min(img_h, result.position + margin)
                win_h = row_end - row_off
                thumb = src.read(
                    1,
                    window=Window(left, row_off, roi_w, win_h),
                    out_shape=(512, 512),
                    resampling=Resampling.average,
                )
                line_row = (result.position - row_off) / win_h * 512
                ax_img.imshow(thumb, cmap="gray", aspect="auto")
                ax_img.axhline(line_row, color="yellow", linewidth=1.5)
                ax_img.set_title(f"{side} edge — position={result.position} px")
                ax_img.axis("off")

                profile = result.band.flatten()
                ax_prof.plot(profile, color="steelblue", linewidth=1)
                ax_prof.axvline(
                    result.rupture_local, color="red", linewidth=1.5, label=f"rupture={result.rupture_local}"
                )
                ax_prof.set_title(f"{side} band profile")
                ax_prof.set_xlabel("row index (downsampled)")
                ax_prof.set_ylabel("intensity")
                ax_prof.legend(fontsize=8)

        return fig

    def _process_side(self, band: NDArray[np.integer], window: Window, side: str) -> FlatResult:
        ruptures = detect_ruptures(band.flatten(), self.background_threshold, reverse_scan=(side == "top"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        rupture_local = int(ruptures[0])
        position = int(rupture_local * self.stride + window.row_off)
        return FlatResult(position, rupture_local, band)
