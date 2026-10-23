import logging
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import BaseModel
from rasterio.windows import Window

from hipp.image import SubImage, SubImageSpec
from hipp.kh9pc.detection.base import DetectionError, DetectionResult, fit_ransac_poly
from hipp.kh9pc.detection.detect_vertical_edges import VerticalDetectionResult
from hipp.kh9pc.kh9_image_spec import IMAGE_HEIGHT_PX
from hipp.tools import skip_existing

logger = logging.getLogger(__name__)


class PolyOptions(BaseModel):
    """Parameters of the poly horizontal detection."""

    downsample_scale: tuple[float, float] = (0.001, 0.1)
    polynomial_degree: int = 2
    background_threshold: int = 20
    ransac_residual_threshold: float = 800.0
    ransac_max_trials: int = 1000
    max_relative_error: float = 0.05
    max_std: float = 100


class PolyEdge(BaseModel):
    """Polynomial fit of one horizontal edge, with what's needed to redraw it."""

    coefficients: list[float]  # np.polyval order, global x -> global y
    sub_image: SubImageSpec
    edge_points: list[tuple[float, float]]  # global coordinates
    inlier_mask: list[bool]

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate the edge row at global columns *x*."""
        return np.polyval(self.coefficients, x)


class PolyDetectionResult(DetectionResult):
    """Top and bottom polynomial edges of a raster, fitted between columns left and right."""

    raster_filepath: Path
    options: PolyOptions
    left: int
    right: int
    top: PolyEdge
    bottom: PolyEdge

    @property
    def gap(self) -> NDArray[np.floating]:
        """Bottom minus top edge rows, sampled at 100 columns across [left, right]."""
        x = np.linspace(self.left, self.right, 100)
        return self.bottom.predict(x) - self.top.predict(x)

    @property
    def failure_reason(self) -> str | None:
        """Why the fit is not trustworthy, or None if it is."""
        gap = self.gap
        relative_error = abs(gap.mean() - IMAGE_HEIGHT_PX) / IMAGE_HEIGHT_PX
        if relative_error > self.options.max_relative_error or gap.std() > self.options.max_std:
            return (
                f"top/bottom gap is off the expected height (relative error={relative_error:.3f}, max "
                f"{self.options.max_relative_error}; std={gap.std():.0f} px, max {self.options.max_std})"
            )
        return None


def detect_poly_horizontal_edges(
    raster_filepath: str | Path,
    output_file: str | Path,
    vertical: str | Path | VerticalDetectionResult,
    overwrite: bool = False,
    options: PolyOptions | None = None,
) -> None:
    """Fit a RANSAC polynomial to the top and bottom film edges and write the result to *output_file* as JSON.

    The search is limited to the columns between the film edges of *vertical*.
    Raises DetectionError if no edge point is found on a side. A fit whose top/bottom gap deviates from the expected
    height is still written, with ``is_failed`` set.
    Skipped if *output_file* already exists and *overwrite* is False.
    """
    raster_filepath = Path(raster_filepath)
    output_file = Path(output_file)

    if skip_existing(output_file, overwrite):
        return

    vertical = VerticalDetectionResult.load(vertical)
    left, right = vertical.left.position, vertical.right.position

    options = options or PolyOptions()

    with rasterio.open(raster_filepath) as src:
        window_height = src.height - IMAGE_HEIGHT_PX
        top = _detect_side(src, Window(left, 0, right - left, window_height), "top", options)
        bottom_window = Window(left, src.height - window_height, right - left, window_height)
        bottom = _detect_side(src, bottom_window, "bottom", options)

    result = PolyDetectionResult(
        raster_filepath=raster_filepath, options=options, left=left, right=right, top=top, bottom=bottom
    )

    gap = result.gap
    logger.info(
        "%s - horizontal edges fitted: height=%.0f±%.0f px (expected=%d, diff=%+.0f px)",
        raster_filepath.name,
        gap.mean(),
        gap.std(),
        IMAGE_HEIGHT_PX,
        gap.mean() - IMAGE_HEIGHT_PX,
    )
    if result.failure_reason:
        logger.warning("%s - poly detection failed: %s", raster_filepath.name, result.failure_reason)

    result.save(output_file)


def _detect_side(
    src: rasterio.DatasetReader, window: Window, side: Literal["top", "bottom"], options: PolyOptions
) -> PolyEdge:
    """Sample one edge point per downsampled column and fit a RANSAC polynomial through them."""
    scale_x, scale_y = options.downsample_scale
    out_shape = (1, int(window.height * scale_y), int(window.width * scale_x))
    sub_image = SubImage(src, window, out_shape)

    # scan from the outer film border inwards
    points_local = find_edge_points(sub_image.band, options.background_threshold, reverse_scan=(side == "top"))
    if len(points_local) == 0:
        raise DetectionError(f"No {side} edge points detected")

    points = sub_image.to_global(points_local.astype(np.float64))
    x, y = points[:, 0], points[:, 1]
    ransac = fit_ransac_poly(
        x,
        y,
        degree=options.polynomial_degree,
        residual_threshold=options.ransac_residual_threshold * scale_y,
        max_trials=options.ransac_max_trials,
    )
    inliers = ransac.inlier_mask_

    # refit on the inliers: the RANSAC pipeline standardizes features, so its own coefficients aren't polyval ones
    coefficients = np.polyfit(x[inliers], y[inliers], options.polynomial_degree)

    return PolyEdge(
        coefficients=coefficients.tolist(),
        sub_image=SubImageSpec.from_sub_image(sub_image),
        edge_points=[(float(px), float(py)) for px, py in points],
        inlier_mask=inliers.tolist(),
    )


def find_edge_on_vec(vec: NDArray[np.floating], threshold: float = 20.0, reverse_scan: bool = False) -> int | None:
    """Locate the film edge in a 1D column profile as the background threshold crossing.

    Returns None if the profile never drops below *threshold*.
    """
    if reverse_scan:
        vec = vec[::-1]

    mask = vec < threshold

    if not mask.any():
        return None

    rupture = int(np.argmax(mask))

    if rupture < 2:
        return None

    return len(vec) - rupture - 1 if reverse_scan else rupture


def find_edge_points(
    img: NDArray[np.generic], threshold: float = 20, reverse_scan: bool = False
) -> NDArray[np.integer]:
    """Run ``find_edge_on_vec`` on every column of *img*, returning the ``(col, row)`` of each detected edge."""
    res = []
    for i in range(img.shape[1]):
        edge = find_edge_on_vec(img[:, i].astype(np.float64), threshold, reverse_scan)
        if edge is not None:
            res.append([i, edge])
    return np.array(res)
