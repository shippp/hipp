from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import rasterio
from rasterio.windows import Window

from hipp.image import match_multi_templates
from hipp.kh9pc.restitution.base import RectificationStrategy
from hipp.kh9pc.utils import SubImage, create_circle_template, measure_circularity


@dataclass
class FiducialResult:
    candidates: pd.DataFrame
    # Columns: x, y, score, circularity, radius, used_radius, inlier


@dataclass
class FiducialRectificationStrategy(RectificationStrategy):
    vertical_edges: tuple[int, int]
    height_fraction: float = 0.15
    block_width: int = 512
    threshold: float = 0.7
    template_fiducial_radii: list[int] = field(default_factory=lambda: [18, 25])
    mad_window: int = 11  # number of neighbours for the sliding MAD filter
    mad_threshold: float = 3.0  # k × MAD rejection threshold

    def __post_init__(self) -> None:
        super().__init__()
        self.top_: FiducialResult | None = None
        self.bottom_: FiducialResult | None = None

    @property
    def top(self) -> FiducialResult:
        if self.top_ is None:
            raise RuntimeError("top fiducials not available — call fit() first")
        return self.top_

    @property
    def bottom(self) -> FiducialResult:
        if self.bottom_ is None:
            raise RuntimeError("bottom fiducials not available — call fit() first")
        return self.bottom_

    def _fit(self, raster_filepath: str | Path) -> Self:
        template_dict = {
            f"circle_{r}": cv2.GaussianBlur(create_circle_template(r), (5, 5), 1.5)
            for r in self.template_fiducial_radii
        }
        margin = 2 * max(self.template_fiducial_radii)

        with rasterio.open(raster_filepath) as src:
            col_start, col_end = self.vertical_edges
            window_height = int(src.height * self.height_fraction)

            for side, row_off in {"top": 0, "bottom": src.height - window_height}.items():
                blocks = []
                for x in range(col_start, col_end, self.block_width):
                    block_start = max(col_start, x - margin)
                    block_end = min(col_end, x + self.block_width + margin)
                    window = Window(block_start, row_off, block_end - block_start, window_height)
                    sub_img = SubImage(src, window)

                    df = match_multi_templates(sub_img.band, template_dict, margin, n_matches=2)

                    # add the circularity and radius (local coords, before global conversion)
                    max_r = max(self.template_fiducial_radii)
                    h, w = sub_img.band.shape
                    circularities, radii = [], []
                    for row in df.itertuples():
                        cx, cy = int(row.x), int(row.y)
                        x0, x1 = max(0, cx - max_r), min(w, cx + max_r + 1)
                        y0, y1 = max(0, cy - max_r), min(h, cy + max_r + 1)
                        circ, rad = measure_circularity(sub_img.band[y0:y1, x0:x1])
                        circularities.append(circ)
                        radii.append(rad)
                    df["circularity"] = circularities
                    df["radius"] = radii

                    # transform local coord to global image coords
                    df[["x", "y"]] = sub_img.to_global(df[["x", "y"]].values).astype(int)

                    blocks.append(df)

                all_candidates = (
                    self._nms(pd.concat(blocks, ignore_index=True), radius=margin)
                    if blocks
                    else pd.DataFrame(columns=["template", "x", "y", "score"])
                )
                setattr(self, side + "_", FiducialResult(all_candidates))

        return self

    def compute_grid(self) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        raise RuntimeError("not implemented")

    def get_qc_figures(self) -> list[Figure]:
        return [
            self._plot_fiducials(self.top, "top"),
            self._plot_fiducials(self.bottom, "bottom"),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nms(self, df: pd.DataFrame, radius: int) -> pd.DataFrame:
        """Remove duplicate detections within `radius` pixels, keeping the highest score."""
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        xy = df[["x", "y"]].values.astype(float)
        keep = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            if not keep[i]:
                continue
            dists = np.linalg.norm(xy[i + 1 :] - xy[i], axis=1)
            keep[i + 1 :][dists < radius] = False
        return df[keep].reset_index(drop=True)

    # def _compute_inter_distances(self, candidates: pd.DataFrame) -> NDArray[np.floating]:
    #     """Euclidean distances between consecutive inlier fiducials ordered by X."""
    #     inliers = candidates[candidates["inlier"]].sort_values("x")
    #     if len(inliers) < 2:
    #         return np.array([], dtype=float)
    #     diffs = np.diff(inliers[["x", "y"]].values, axis=0).astype(float)
    #     return np.linalg.norm(diffs, axis=1)

    def _plot_fiducials(self, result: FiducialResult, side: str) -> Figure:
        df = result.candidates
        n = len(df)
        half = max(self.template_fiducial_radii) * 2
        patch_size = 2 * half + 1

        ncols = max(1, int(np.ceil(np.sqrt(n))))
        nrows = max(1, int(np.ceil(n / ncols)))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
        fig.suptitle(f"Fiducials — {side} ({n} detected)")

        axes_flat = np.array(axes).reshape(-1)
        for ax in axes_flat:
            ax.axis("off")

        if n == 0:
            fig.tight_layout()
            return fig

        with rasterio.open(self.raster_filepath) as src:
            for ax, row in zip(axes_flat, df.itertuples()):
                col_off = max(0, min(int(row.x) - half, src.width - patch_size))
                row_off = max(0, min(int(row.y) - half, src.height - patch_size))
                patch = src.read(1, window=Window(col_off, row_off, patch_size, patch_size))

                ax.imshow(patch, cmap="gray", interpolation="nearest")
                ax.set_title(f"{row.score:.3f}", fontsize=9, color="red" if not row.inlier else "black")
                ax.axis("off")
                if not row.inlier:
                    ax.add_patch(
                        Rectangle(
                            (0, 0),
                            1,
                            1,
                            transform=ax.transAxes,
                            fill=False,
                            edgecolor="red",
                            linewidth=4,
                            clip_on=False,
                        )
                    )

        fig.tight_layout()
        return fig
