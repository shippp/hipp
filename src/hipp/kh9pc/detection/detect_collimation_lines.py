import logging
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import BaseModel
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d

from hipp.image import SubImage, SubImageSpec
from hipp.kh9pc.detection.base import DetectionResult, fit_ransac_poly
from hipp.kh9pc.detection.detect_poly_horizontal_edges import PolyDetectionResult
from hipp.tools import skip_existing

logger = logging.getLogger(__name__)


class CollimationOptions(BaseModel):
    """Parameters of the collimation lines detection."""

    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    n_columns: int = 100
    stride: int = 10
    refinement_fraction: float = 0.03
    max_width_peak: int = 200
    min_inliers_threshold: float = 0.5


class CollimationLine(BaseModel):
    """Polynomial fit of one collimation line, with what's needed to redraw it."""

    coefficients: list[float]  # np.polyval order, global x -> global y
    sub_image: SubImageSpec
    peaks: list[tuple[float, float]]  # global coordinates
    inlier_mask: list[bool]

    @property
    def inlier_ratio(self) -> float:
        """Fraction of column peaks kept by RANSAC."""
        return float(np.mean(self.inlier_mask))

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate the line row at global columns *x*."""
        return np.polyval(self.coefficients, x)


class CollimationDetectionResult(DetectionResult):
    """Top and bottom collimation lines of a raster, fitted between columns left and right."""

    raster_filepath: Path
    options: CollimationOptions
    left: int
    right: int
    top: CollimationLine
    bottom: CollimationLine

    @property
    def gap(self) -> NDArray[np.floating]:
        """Bottom minus top line rows, sampled at 100 columns across [left, right]."""
        x = np.linspace(self.left, self.right, 100)
        return self.bottom.predict(x) - self.top.predict(x)

    @property
    def failure_reason(self) -> str | None:
        """Why the fit is not trustworthy, or None if it is."""
        low = [
            f"{side}={100 * line.inlier_ratio:.0f}%"
            for side, line in (("top", self.top), ("bottom", self.bottom))
            if line.inlier_ratio < self.options.min_inliers_threshold
        ]
        if low:
            return f"too few inliers: {', '.join(low)} (min {100 * self.options.min_inliers_threshold:.0f}%)"
        return None


def detect_collimation_lines(
    raster_filepath: str | Path,
    output_file: str | Path,
    poly: str | Path | PolyDetectionResult,
    overwrite: bool = False,
    options: CollimationOptions | None = None,
) -> None:
    """Fit a RANSAC polynomial to the top and bottom collimation lines and write the result to *output_file* as JSON.

    Each line is searched in a thin strip just inside the top / bottom film edge fitted in *poly*, between its
    vertical film edges.
    A line keeping fewer than ``min_inliers_threshold`` of its peaks is still written, with
    ``is_failed`` set.
    Skipped if *output_file* already exists and *overwrite* is False.
    """
    raster_filepath = Path(raster_filepath)
    output_file = Path(output_file)

    if skip_existing(output_file, overwrite):
        return

    poly = PolyDetectionResult.load(poly)
    left, right = poly.left, poly.right

    options = options or CollimationOptions()

    with rasterio.open(raster_filepath) as src:
        center = (left + right) / 2
        top_edge = int(poly.top.predict(np.array(center)))
        bottom_edge = int(poly.bottom.predict(np.array(center)))
        window_height = int(src.height * options.refinement_fraction)
        # both collimation lines lie inside the film: the top strip extends down from the top edge,
        # the bottom strip extends up from the bottom edge
        top = _detect_side(src, Window(left, top_edge, right - left, window_height), options)
        bottom = _detect_side(src, Window(left, bottom_edge - window_height, right - left, window_height), options)

    result = CollimationDetectionResult(
        raster_filepath=raster_filepath, options=options, left=left, right=right, top=top, bottom=bottom
    )

    gap = result.gap
    logger.info(
        "%s - collimation lines fitted: gap=%.0f±%.0f px, inliers top=%.0f%% bottom=%.0f%%",
        raster_filepath.name,
        gap.mean(),
        gap.std(),
        100 * top.inlier_ratio,
        100 * bottom.inlier_ratio,
    )
    if result.failure_reason:
        logger.warning("%s - collimation detection failed: %s", raster_filepath.name, result.failure_reason)

    result.save(output_file)


def _detect_side(src: rasterio.DatasetReader, window: Window, options: CollimationOptions) -> CollimationLine:
    """Locate the collimation peak in every downsampled column and fit a RANSAC polynomial through them."""
    out_shape = (1, int(window.height) // options.stride, options.n_columns)
    sub_image = SubImage(src, window, out_shape)

    max_peak_width = options.max_width_peak // options.stride
    peaks_local = np.array(
        [
            [col, detect_collimation_peak(sub_image.band[:, col], max_peak_width)]
            for col in range(sub_image.band.shape[1])
        ],
        dtype=np.float64,
    )
    peaks = sub_image.to_global(peaks_local)
    x, y = peaks[:, 0], peaks[:, 1]

    ransac = fit_ransac_poly(
        x,
        y,
        degree=options.polynomial_degree,
        residual_threshold=options.ransac_residual_threshold,
        max_trials=options.ransac_max_trials,
    )
    inliers = ransac.inlier_mask_

    # refit on the inliers: the RANSAC pipeline standardizes features, so its own coefficients aren't polyval ones
    coefficients = np.polyfit(x[inliers], y[inliers], options.polynomial_degree)

    return CollimationLine(
        coefficients=coefficients.tolist(),
        sub_image=SubImageSpec.from_sub_image(sub_image),
        peaks=[(float(px), float(py)) for px, py in peaks],
        inlier_mask=inliers.tolist(),
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
