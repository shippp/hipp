"""
Copyright (c) 2025 HIPP developers
Description: RectificationStrategy — combined detection + control-point + transform ABC.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
import warnings

import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import Pipeline

import cv2
import pandas as pd

from hipp.image import match_multi_templates, remap_tif_blockwise_from_points
from hipp.kh9pc.restitution.output_size import OutputSize
from hipp.kh9pc.utils import (
    SubImage,
    create_circle_template,
    detect_collimation_peak,
    detect_ruptures,
    fit_ransac_poly,
    measure_circularity,
)


class RectificationStrategy(ABC):
    """ABC for horizontal-edge detection + geometric rectification.

    Subclasses implement :meth:`_fit` (detection) and :meth:`_control_points`
    (grid generation). The concrete :meth:`transform` method chains both together
    with an :class:`~hipp.kh9pc.restitution.output_size.OutputSize` to produce the
    final rectified raster.
    """

    def __init__(self) -> None:
        self.raster_filepath_: Path | None = None

    def fit(self, raster_filepath: str | Path) -> Self:
        result = self._fit(raster_filepath)
        self.raster_filepath_ = Path(raster_filepath)
        return result

    @abstractmethod
    def _fit(self, raster_filepath: str | Path) -> Self: ...

    @abstractmethod
    def _control_points(self) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        """Return ``(src_points, dst_points, detected_size)``."""
        ...

    @property
    @abstractmethod
    def transformation(self) -> str:
        """Interpolation type passed to ``remap_tif_blockwise_from_points``."""
        ...

    def transform(self, input_path: str | Path, output_path: str | Path, output_size: OutputSize) -> None:
        """Apply the full rectification transform.

        Parameters
        ----------
        input_path : str | Path
            Source mosaic raster.
        output_path : str | Path
            Destination rectified raster.
        output_size : OutputSize
            Canvas-sizing strategy applied on top of the detected content region.
        """
        src_pts, dst_pts, detected_size = self._control_points()
        src_pts, dst_pts, out_size = output_size.apply(src_pts, dst_pts, detected_size)
        remap_tif_blockwise_from_points(
            input_path=input_path,
            output_path=output_path,
            src_points=src_pts,
            dst_points=dst_pts,
            output_size=out_size,
            transformation=self.transformation,
        )

    @property
    def raster_filepath(self) -> Path:
        if self.raster_filepath_ is None:
            raise RuntimeError("need to call fit() first")
        return self.raster_filepath_

    @property
    def is_fitted(self) -> bool:
        return self.raster_filepath_ is not None


# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


@dataclass
class EdgeResult:
    ruptures_local: NDArray[np.integer]
    ruptures_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_image: SubImage

    @property
    def poly(self) -> Pipeline:
        return self.model.estimator_


@dataclass
class CollimationResult:
    peaks_local: NDArray[np.integer]
    peaks_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_img: SubImage


@dataclass
class FlatResult:
    position: int
    rupture_local: int
    sub_image: SubImage


# ---------------------------------------------------------------------------
# PolyStrategy
# ---------------------------------------------------------------------------


@dataclass
class PolyStrategy(RectificationStrategy):
    """Detect top/bottom edges via rupture detection + polynomial RANSAC fit."""

    vertical_edges: tuple[int, int]
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10
    polynomial_degree: int = 2
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)

    def __post_init__(self) -> None:
        super().__init__()
        self.top_: EdgeResult | None = None
        self.bottom_: EdgeResult | None = None

    @property
    def transformation(self) -> str:
        return "tps"

    @property
    def top(self) -> EdgeResult:
        if self.top_ is None:
            raise RuntimeError("top edge not available — call fit() first")
        return self.top_

    @property
    def bottom(self) -> EdgeResult:
        if self.bottom_ is None:
            raise RuntimeError("bottom edge not available — call fit() first")
        return self.bottom_

    def _fit(self, raster_filepath: str | Path) -> Self:
        with rasterio.open(raster_filepath) as src:
            col_off, col_end = self.vertical_edges
            window_width = col_end - col_off
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                setattr(self, side + "_", self._process_side(sub_image, side))

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> EdgeResult:
        res = []
        for i in range(sub_image.band.shape[1]):
            ruptures = detect_ruptures(sub_image.band[:, i], self.background_threshold, reverse_scan=(side == "top"))
            if len(ruptures) > 0:
                res.append((i, ruptures[0]))

        if not res:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        ruptures_local = np.array(res)
        ruptures_global = sub_image.to_global(ruptures_local)

        model = fit_ransac_poly(
            ruptures_global[:, 0],
            ruptures_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())
        if inlier_ratio < 0.5:
            warnings.warn(
                f"{side} edge: low inlier ratio ({inlier_ratio:.1%}), RANSAC fit may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        x_sample = np.linspace(
            sub_image.window.col_off, sub_image.window.col_off + sub_image.window.width, self.grid_shape[0]
        )
        y_global_pred = model.predict(x_sample.reshape(-1, 1)).ravel()
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([x_sample, y_distortion])

        return EdgeResult(
            ruptures_local=ruptures_local,
            ruptures_global=ruptures_global.astype(int),
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_image=sub_image,
        )

    def _control_points(self) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        left, right = self.vertical_edges
        detected_width = right - left

        x_src = np.linspace(left, right, self.grid_shape[0])
        y_top_src = self.top.poly.predict(x_src.reshape(-1, 1)).ravel()
        y_bottom_src = self.bottom.poly.predict(x_src.reshape(-1, 1)).ravel()
        detected_height = int(np.abs(np.mean(y_bottom_src - y_top_src)))

        x_dst = np.linspace(0, detected_width, self.grid_shape[0])

        src_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        dst_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        for i, (xi_src, xi_dst, yt, yb) in enumerate(zip(x_src, x_dst, y_top_src, y_bottom_src)):
            src_points[i, :, 0] = xi_src
            src_points[i, :, 1] = np.linspace(yt, yb, self.grid_shape[1])
            dst_points[i, :, 0] = xi_dst
            dst_points[i, :, 1] = np.linspace(0, detected_height, self.grid_shape[1])

        return src_points.reshape(-1, 2), dst_points.reshape(-1, 2), (detected_width, detected_height)


# ---------------------------------------------------------------------------
# CollimationStrategy
# ---------------------------------------------------------------------------


@dataclass
class CollimationStrategy(RectificationStrategy):
    """Detect top/bottom collimation lines via peak detection + polynomial RANSAC fit."""

    vertical_edges: tuple[int, int]
    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    stride: int = 10
    height_fraction: float = 0.15
    max_width_peak: int = 200
    collimation_line_dist: int = 21770

    def __post_init__(self) -> None:
        super().__init__()
        self.top_: CollimationResult | None = None
        self.bottom_: CollimationResult | None = None

    @property
    def transformation(self) -> str:
        return "tps"

    @property
    def top(self) -> CollimationResult:
        if self.top_ is None:
            raise RuntimeError("top collimation line not available — call fit() first")
        return self.top_

    @property
    def bottom(self) -> CollimationResult:
        if self.bottom_ is None:
            raise RuntimeError("bottom collimation line not available — call fit() first")
        return self.bottom_

    def _fit(self, raster_filepath: str | Path) -> Self:
        with rasterio.open(raster_filepath) as src:
            window_width = self.vertical_edges[1] - self.vertical_edges[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(self.vertical_edges[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges[0], src.height - window_height, window_width, window_height),
            }.items():
                sub_img = SubImage(src, window, out_shape, resampling=Resampling.average)
                setattr(self, side + "_", self._process_side(side, sub_img))

        return self

    def _process_side(self, side: str, sub_img: SubImage) -> CollimationResult:
        h, w = sub_img.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)
        for col in range(w):
            vec = sub_img.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        peaks_global = sub_img.to_global(peaks_local).astype(int)

        model = fit_ransac_poly(
            peaks_global[:, 0],
            peaks_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())
        if inlier_ratio < 0.5:
            warnings.warn(
                f"{side} collimation line: low inlier ratio ({inlier_ratio:.1%}), RANSAC fit may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        y_global_pred = model.predict(peaks_global[:, 0].reshape(-1, 1))
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([peaks_global[:, 0], y_distortion])

        return CollimationResult(
            peaks_local=peaks_local,
            peaks_global=peaks_global,
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_img=sub_img,
        )

    def _control_points(self) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        left, right = self.vertical_edges
        detected_width = right - left
        detected_height = self.collimation_line_dist

        x_src = np.linspace(left, right, self.grid_shape[0])
        y_top_src = self.top.model.predict(x_src.reshape(-1, 1)).ravel()
        y_bottom_src = self.bottom.model.predict(x_src.reshape(-1, 1)).ravel()
        x_dst = np.linspace(0, detected_width, self.grid_shape[0])

        src_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        dst_points = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=float)
        for i, (xi_src, xi_dst, yt, yb) in enumerate(zip(x_src, x_dst, y_top_src, y_bottom_src)):
            src_points[i, :, 0] = xi_src
            src_points[i, :, 1] = np.linspace(yt, yb, self.grid_shape[1])
            dst_points[i, :, 0] = xi_dst
            dst_points[i, :, 1] = np.linspace(0, detected_height, self.grid_shape[1])

        return src_points.reshape(-1, 2), dst_points.reshape(-1, 2), (detected_width, detected_height)


# ---------------------------------------------------------------------------
# FlatStrategy
# ---------------------------------------------------------------------------


@dataclass
class FlatStrategy(RectificationStrategy):
    """Detect top/bottom edges as flat horizontal lines (affine rectification)."""

    vertical_edges: tuple[int, int]
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10

    def __post_init__(self) -> None:
        super().__init__()
        self.top_: FlatResult | None = None
        self.bottom_: FlatResult | None = None

    @property
    def transformation(self) -> str:
        return "affine"

    @property
    def top(self) -> FlatResult:
        if self.top_ is None:
            raise RuntimeError("top edge not available — call fit() first")
        return self.top_

    @property
    def bottom(self) -> FlatResult:
        if self.bottom_ is None:
            raise RuntimeError("bottom edge not available — call fit() first")
        return self.bottom_

    def _fit(self, raster_filepath: str | Path) -> Self:
        with rasterio.open(raster_filepath) as src:
            window_width = self.vertical_edges[1] - self.vertical_edges[0]
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, 1)

            for side, window in {
                "top": Window(self.vertical_edges[0], 0, window_width, window_height),
                "bottom": Window(self.vertical_edges[0], src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                setattr(self, side + "_", self._process_side(sub_image, side))

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> FlatResult:
        ruptures = detect_ruptures(sub_image.band.flatten(), self.background_threshold, reverse_scan=(side == "top"))
        if len(ruptures) == 0:
            raise RuntimeError(f"No rupture detected on the {side} edge.")
        rupture_local = int(ruptures[0])
        position = int(sub_image.to_global(np.array([0.0, rupture_local]))[1])
        return FlatResult(position=position, rupture_local=rupture_local, sub_image=sub_image)

    def _control_points(self) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        left, right = self.vertical_edges
        detected_width = right - left
        detected_height = self.bottom.position - self.top.position

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
                [0, detected_height],
                [detected_width, 0],
                [detected_width, detected_height],
            ],
            dtype=float,
        )
        return src_points, dst_points, (detected_width, detected_height)


# ---------------------------------------------------------------------------
# FiducialStrategy
# ---------------------------------------------------------------------------


@dataclass
class FiducialResult:
    candidates: pd.DataFrame
    # Columns: template, x, y, score, circularity, radius


@dataclass
class FiducialStrategy(RectificationStrategy):
    """Detect fiducial marks (circular dots) along top/bottom film edges."""

    vertical_edges: tuple[int, int]
    height_fraction: float = 0.15
    block_width: int = 512
    threshold: float = 0.7
    template_fiducial_radii: list[int] = field(default_factory=lambda: [18, 25])
    mad_window: int = 11
    mad_threshold: float = 3.0

    def __post_init__(self) -> None:
        super().__init__()
        self.top_: FiducialResult | None = None
        self.bottom_: FiducialResult | None = None

    @property
    def transformation(self) -> str:
        raise NotImplementedError

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

                    df[["x", "y"]] = sub_img.to_global(df[["x", "y"]].values).astype(int)
                    blocks.append(df)

                all_candidates = (
                    self._nms(pd.concat(blocks, ignore_index=True), radius=margin)
                    if blocks
                    else pd.DataFrame(columns=["template", "x", "y", "score"])
                )
                setattr(self, side + "_", FiducialResult(all_candidates))

        return self

    def _control_points(self) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        raise NotImplementedError

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
