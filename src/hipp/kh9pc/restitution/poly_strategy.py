"""
Copyright (c) 2026 HIPP developers
Description: PolyStrategy — polynomial edge fitting for KH-9 PC restitution. Samples
    rupture points along the top and bottom film edges in a downsampled grid, fits a
    RANSAC polynomial per edge, then applies a Thin Plate Spline warp to straighten
    the curved edges.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from skimage.transform import AffineTransform
from sklearn.linear_model import RANSACRegressor
import cv2

from hipp.image import SubImage, remap_tif_blockwise
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import RestitutionStrategy, Transformation, fit_ransac_poly
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector


@dataclass
class PolyResult:
    edge_points_global: NDArray[np.integer]
    edge_points_local: NDArray[np.integer]
    model: RANSACRegressor
    sub_image: SubImage


@dataclass
class PolyStrategy(RestitutionStrategy):
    """Restitution strategy based on polynomial edge fitting.

    Detects rupture points along the top and bottom film edges in a downsampled
    horizontal grid, fits a RANSAC polynomial per edge, then warps the image with
    a Thin Plate Spline transform to map the curved edges to horizontal target lines.
    Fails if the inlier ratio on either edge falls below ``min_inliers_threshold``.
    """

    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    downsample_scale: tuple[float, float] = (0.001, 0.1)
    polynomial_degree: int = 2
    background_threshold: int = 20
    ransac_residual_threshold: float = 200.0
    ransac_max_trials: int = 1000

    max_relative_error: float = 0.05
    max_std: float = 100

    n_points: int = 10

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, PolyResult] = {}
        self.__transformation_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        """True if either edge inlier ratio is below ``min_inliers_threshold``."""
        relative_error, std_dist = self.compute_metrics()

        return (relative_error > self.max_relative_error) or (std_dist > self.max_std)

    @property
    def top_(self) -> PolyResult:
        """Fitted top edge result. Raises if ``fit()`` has not been called."""
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> PolyResult:
        """Fitted bottom edge result. Raises if ``fit()`` has not been called."""
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        """TPS Transformation from curved edges to horizontal lines (computed lazily)."""
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def _fit(self, raster_filepath: Path) -> Self:
        """Detect and fit polynomial models for the top and bottom edges."""
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_off, _ = self.vertical_detector.edges_
        window_width = self.vertical_detector.detected_width_
        expected_height = KH9ImageSpec.from_raster_filepath(raster_filepath).expected_size[1]

        with rasterio.open(raster_filepath) as src:
            window_height = src.height - expected_height
            out_shape = (1, int(window_height * self.downsample_scale[1]), int(window_width * self.downsample_scale[0]))

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                self._results[side] = self._process_side(sub_image, side)

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> PolyResult:
        """Sample column-wise ruptures, fit a RANSAC polynomial, and compute the distortion curve."""
        edge_points_local = find_edge_points(sub_image.band, self.background_threshold, reverse_scan=(side == "top"))

        if len(edge_points_local) == 0:
            raise RuntimeError(f"No edge points detected on the {side} edge.")

        edge_points_global = sub_image.to_global(edge_points_local.astype(np.float64)).astype(int)

        model = fit_ransac_poly(
            edge_points_global[:, 0],
            edge_points_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold * self.downsample_scale[1],
            max_trials=self.ransac_max_trials,
        )

        return PolyResult(
            edge_points_local=edge_points_local,
            edge_points_global=edge_points_global,
            model=model,
            sub_image=sub_image,
        )

    def _compute_transformation(self) -> Transformation:
        """Build an affine Transformation that maps the fitted curved edges to horizontal target lines."""
        left, right = self.vertical_detector.edges_
        detected_width = self.vertical_detector.detected_width_
        output_width, output_height = KH9ImageSpec.from_raster_filepath(self.raster_filepath_).expected_size

        x = np.linspace(left, right, self.n_points)

        y_top_src = self.top_.model.predict(x.reshape(-1, 1))
        y_bot_src = self.bottom_.model.predict(x.reshape(-1, 1))

        top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))
        detected_height = bot - top

        y_top_dst = np.full_like(x, top)
        y_bot_dst = np.full_like(x, bot)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important)
        deformation = AffineTransform()
        if not deformation.estimate(dst, src):
            raise RuntimeError("Affine transformation estimation failed.")

        # ---- CENTERING TO OUTPUT ----
        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2

        crop_offset = (int(left - pad_x), int(top - pad_y))

        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )

    def transform(self, output_path: str | Path) -> None:
        """Write the restituted image using the polynomial TPS warp."""
        tf = self.transformation_

        remap_tif_blockwise(
            tf.raster_filepath,
            output_path,
            tf.inverse_remap,
            tf.output_size,
            block_size=2**13,
            lowres_step=100,
        )

    def compute_metrics(self) -> tuple[float, float]:
        start, end = self.vertical_detector.edges_

        X = np.linspace(start, end, 100).reshape(-1, 1)

        y_top = self.top_.model.predict(X).ravel()
        y_bottom = self.bottom_.model.predict(X).ravel()

        dist = y_bottom - y_top

        true_height = KH9ImageSpec.from_raster_filepath(self.raster_filepath_).expected_size[1]

        relative_error = np.abs(np.mean(dist) - true_height) / true_height
        std_dist = np.std(dist)

        return relative_error, std_dist


def find_edge_on_vec(vec: NDArray[np.floating], threshold: float = 20.0, reverse_scan: bool = False) -> int | None:
    if reverse_scan:
        vec = vec[::-1]

    mask = vec < threshold

    if not mask.any():
        return None

    rupture = np.argmax(mask)

    if rupture < 2:
        return None

    grad = np.gradient(vec[:rupture])
    edge = int(np.argmin(grad))

    return len(vec) - edge - 1 if reverse_scan else edge


def find_edge_points(img: cv2.typing.MatLike, threshold: float = 20, reverse_scan: bool = False) -> NDArray[np.integer]:
    res = []
    for i in range(img.shape[1]):
        edge = find_edge_on_vec(img[:, i].astype(np.float64), threshold, reverse_scan)
        if edge is not None:
            res.append([i, edge])
    return np.array(res)
