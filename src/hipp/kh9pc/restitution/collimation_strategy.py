"""
Copyright (c) 2026 HIPP developers
Description: CollimationStrategy — refines the polynomial edge estimate using the physical
    collimation lines printed on the KH-9 film. Collimation lines are narrow bright bands
    whose peak can be located column-by-column with high accuracy, providing a stronger
    geometric reference than the film-edge rupture alone. The fixed known distance between
    collimation lines is used to set the output height precisely.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d
from skimage.transform import ThinPlateSplineTransform
from sklearn.linear_model import RANSACRegressor

from hipp.image import SubImage, remap_tif_blockwise
from hipp.kh9pc.restitution.base import fit_ransac_poly
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy


@dataclass
class CollimationResult:
    """Fitted collimation line model and diagnostics for one side (top or bottom).

    Attributes
    ----------
    peaks_local:
        (N, 2) detected collimation peak coordinates in sub-image pixel space.
    peaks_global:
        (N, 2) same peaks converted to full-raster pixel coordinates.
    distortion:
        (N, 2) array of ``[x, deviation_from_mean]`` along the fitted collimation curve.
    inlier_ratio:
        Fraction of column peaks classified as inliers by RANSAC.
    model:
        Fitted ``RANSACRegressor`` wrapping the polynomial pipeline.
    sub_image:
        Narrow strip around the collimation line used for detection (kept for QC).
    """

    peaks_local: NDArray[np.integer]
    peaks_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_image: SubImage


@dataclass
class CollimationStrategy(RestitutionStrategy):
    """Restitution strategy using physical collimation lines printed on the film.

    Builds on a fitted ``PolyStrategy`` to narrow the search to a thin strip around
    each expected edge, then locates the collimation line peak column-by-column
    with sub-pixel accuracy. The result is fitted with a higher-degree RANSAC
    polynomial, and the fixed physical distance between lines (``collimation_line_dist``)
    sets the output height precisely rather than relying on the detected edge positions.
    Fails if the inlier ratio on either side falls below ``min_inliers_threshold``.
    """

    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    stride: int = 10
    refinement_fraction: float = 0.03
    max_width_peak: int = 200
    collimation_line_dist: int = (
        21770  # known physical distance between top/bottom collimation lines at nominal scan resolution
    )
    min_inliers_threshold: float = 0.5
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, CollimationResult] = {}
        self.__transformation_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        """True if either collimation line inlier ratio is below ``min_inliers_threshold``."""
        return min(self.top_.inlier_ratio, self.bottom_.inlier_ratio) < self.min_inliers_threshold

    @property
    def top_(self) -> CollimationResult:
        """Fitted top collimation line result. Raises if ``fit()`` has not been called."""
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> CollimationResult:
        """Fitted bottom collimation line result. Raises if ``fit()`` has not been called."""
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        """TPS Transformation from collimation curves to horizontal lines (computed lazily)."""
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def _fit(self, raster_filepath: Path) -> Self:
        """Run PolyStrategy first, then detect collimation peaks in a narrow refinement strip."""
        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)

        col_off, col_end = self.poly_strategy.vertical_detector.edges_
        window_width = self.poly_strategy.vertical_detector.detected_width_
        col_center = (col_off + col_end) // 2

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.refinement_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            top_edge = int(self.poly_strategy.top_.model.predict(np.array([[col_center]])).flat[0])
            bot_edge = int(self.poly_strategy.bottom_.model.predict(np.array([[col_center]])).flat[0])

            for side, window in {
                "top": Window(col_off, top_edge, window_width, window_height),
                "bottom": Window(col_off, bot_edge - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape, resampling=Resampling.average)
                self._results[side] = self._process_side(sub_image, side)

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> CollimationResult:
        """Detect collimation peaks column-by-column, fit a RANSAC polynomial, and compute the distortion curve."""
        _, w = sub_image.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)
        for col in range(w):
            vec = sub_image.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        peaks_global = sub_image.to_global(peaks_local).astype(int)

        model = fit_ransac_poly(
            peaks_global[:, 0],
            peaks_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())

        y_global_pred = model.predict(peaks_global[:, 0].reshape(-1, 1))
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([peaks_global[:, 0], y_distortion])

        return CollimationResult(
            peaks_local=peaks_local,
            peaks_global=peaks_global,
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_image=sub_image,
        )

    def _compute_transformation(self) -> Transformation:
        """Build a TPS Transformation using the fixed physical collimation line separation."""
        left, right = self.poly_strategy.vertical_detector.edges_
        detected_width = self.poly_strategy.vertical_detector.detected_width_
        output_width = self.output_width or detected_width

        x = np.linspace(left, right, self.grid_shape[0])

        y_top_src = self.top_.model.predict(x.reshape(-1, 1))
        y_bot_src = self.bottom_.model.predict(x.reshape(-1, 1))

        top = int(np.median(y_top_src))
        bot = top + self.collimation_line_dist
        detected_height = bot - top
        output_height = self.output_height or detected_height

        y_top_dst = np.full_like(x, top)
        y_bot_dst = np.full_like(x, bot)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important)
        deformation = ThinPlateSplineTransform().from_estimate(dst, src)

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
        """Write the restituted image using the collimation TPS warp."""
        tf = self.transformation_

        remap_tif_blockwise(
            tf.raster_filepath,
            output_path,
            tf.inverse_remap,
            tf.output_size,
            block_size=2**13,
            lowres_step=100,
        )


def detect_collimation_peak(x: NDArray[np.number], max_peak_width: int, sigma: int = 2) -> int:
    """Locate the center of the collimation line in a 1D column profile.

    Smooths the profile with a Gaussian, then finds the gradient peak pair
    (rising + falling edge) whose separation is within ``max_peak_width``.
    Returns the index of the intensity maximum between the two gradient peaks,
    or the global maximum as a fallback when no compact peak is found.
    """
    smooth = gaussian_filter1d(x, sigma=sigma)

    grad = np.gradient(smooth)

    idx_max = np.argmax(grad)
    idx_min = np.argmin(grad)

    if abs(idx_max - idx_min) < max_peak_width and idx_max != idx_min:
        w_start = min(idx_max, idx_min)
        w_end = max(idx_max, idx_min)
        idx = np.argmax(smooth[w_start:w_end]) + w_start
    else:
        idx = np.argmax(smooth)  # fallback

    return int(idx)
