"""
Copyright (c) 2026 HIPP developers
Description: Abstract base classes and shared utilities for KH-9 PC restitution strategies.
    Defines the FittingClass / RestitutionStrategy protocol, the Transformation dataclass,
    and the low-level helpers (RANSAC polynomial fitting, rupture detection) shared across
    all concrete strategy implementations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from hipp.kh9pc.kh9_image_spec import KH9ImageSpec

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_HEIGHT: int = 22064
"""Standard output height in pixels for restituted KH-9 PC images (22064 px at nominal scan resolution)."""


class DetectionError(Exception):
    """Raised when no valid detections are found during fitting."""


class FittingClass(ABC):
    """Base class for objects that are fitted against a raster file.

    Subclasses implement ``_fit`` to perform the actual work and expose
    ``is_failed`` to signal whether the result is usable. The public ``fit``
    method wraps ``_fit`` with logging and records the source raster path so
    QC code can always trace results back to their input.
    """

    def __init__(self) -> None:
        self.__raster_filepath_: Path | None = None
        self.__spec_: KH9ImageSpec | None = None

    @property
    def raster_filepath_(self) -> Path:
        """Path of the raster this instance was fitted on. Raises if ``fit()`` has not been called."""
        if self.__raster_filepath_ is None:
            raise RuntimeError("Call fit() before.")
        return self.__raster_filepath_

    @property
    def spec_(self) -> KH9ImageSpec:
        """KH9ImageSpec derived from ``raster_filepath_``, computed once per ``fit()`` call."""
        if self.__spec_ is None:
            self.__spec_ = KH9ImageSpec.from_raster_filepath(self.raster_filepath_)
        return self.__spec_

    @property
    def is_fitted(self) -> bool:
        """True after ``fit()`` has been called at least once."""
        return self.__raster_filepath_ is not None

    @property
    def logging_prefix(self) -> str:
        """Standard log prefix ``"[ClassName] raster.tif"`` for this instance."""
        return f"[{self.__class__.__name__}] {self.raster_filepath_.name}"

    @property
    @abstractmethod
    def is_failed(self) -> bool:
        """True if the last ``fit()`` produced an unusable result."""
        ...

    def fit(self, raster_filepath: str | Path) -> Self:
        """Run ``_fit`` on *raster_filepath*, log start/end, and record the source path."""
        self.__raster_filepath_ = Path(raster_filepath)
        self.__spec_ = None
        logger.info("%s - start fit...", self.logging_prefix)
        fit_res = self._fit(self.__raster_filepath_)
        logger.info("%s - finish fit : [%s]", self.logging_prefix, "FAILED" if self.is_failed else "SUCCESS")
        return fit_res

    @abstractmethod
    def _fit(self, raster_filepath: Path) -> Self:
        """Perform the actual fitting work; called by ``fit()``."""
        ...


class RestitutionStrategy(FittingClass):
    """A ``FittingClass`` that can also produce a restituted output image.

    Concrete strategies (Flat, Poly, Collimation, Fiducial) fit edge/fiducial
    detections and then expose ``transformation_`` (the geometric model) and
    ``transform`` (the method that applies it to write the output GeoTIFF).
    """

    @abstractmethod
    def transform(self, output_path: str | Path) -> None:
        """Apply the fitted transformation and write the restituted image to *output_path*."""
        ...

    @property
    @abstractmethod
    def transformation_(self) -> "Transformation":
        """The fitted ``Transformation`` object describing the geometric correction."""
        ...


@dataclass
class Transformation:
    """Geometric transformation from output pixel space back to input raster space.

    Used with ``remap_tif_blockwise``: for every output pixel coordinate the
    inverse remap first un-crops (adds ``crop_offset``) then applies the
    ``deformation`` callable to obtain the corresponding source coordinate.

    Attributes
    ----------
    raster_filepath:
        Source raster to read pixel values from.
    deformation:
        Callable mapping (N, 2) output coords → (N, 2) source coords (inverse warp).
    crop_offset:
        ``(x, y)`` translation added before deformation to go from the cropped
        output space back to the full raster coordinate system.
    output_size:
        ``(width, height)`` of the restituted output image in pixels.
    """

    raster_filepath: Path
    deformation: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    crop_offset: tuple[float, float] = (0, 0)
    output_size: tuple[int, int] = (0, 0)

    def inverse_remap(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
        """Translate coords by ``crop_offset`` then apply ``deformation``."""
        coords = coords + np.array([self.crop_offset[0], self.crop_offset[1]], dtype=coords.dtype)
        return self.deformation(coords)


def fit_ransac_poly(
    x: NDArray[np.generic],
    y: NDArray[np.generic],
    degree: int = 3,
    residual_threshold: float = 100,
    max_trials: int = 100,
) -> RANSACRegressor:
    """Fit a polynomial regression with RANSAC on 1D data. Returns the fitted RANSACRegressor."""
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        LinearRegression(),
    )

    min_samples = min(degree * 3, len(x))
    ransac = RANSACRegressor(
        poly_model, residual_threshold=residual_threshold, min_samples=min_samples, max_trials=max_trials
    )
    ransac.fit(x.reshape(-1, 1), y)
    return ransac


def detect_ruptures(vec: NDArray[np.number], threshold: float, reverse_scan: bool = False) -> NDArray[np.integer]:
    """Detect indices where the signal drops below a threshold (falling edges).

    If reverse_scan is True, scan from the end and return indices in original coordinates.
    """
    if reverse_scan:
        vec = vec[::-1]

    idx = np.where((vec[1:] <= threshold) & (vec[:-1] > threshold))[0] + 1

    if reverse_scan:
        idx = len(vec) - 1 - idx

    return idx
