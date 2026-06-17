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

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_HEIGHT: int = 22064
"""Standard output height in pixels for restituted KH-9 PC images (22064 px at nominal scan resolution)."""


class DetectionError(Exception):
    """Raised when no valid detections are found during fitting."""


class FittingClass(ABC):
    def __init__(self) -> None:
        self.__raster_filepath_: Path | None = None

    @property
    def raster_filepath_(self) -> Path:
        if self.__raster_filepath_ is None:
            raise RuntimeError("Call fit() before.")
        return self.__raster_filepath_

    @property
    def is_fitted(self) -> bool:
        return self.__raster_filepath_ is not None

    @property
    @abstractmethod
    def is_failed(self) -> bool: ...

    def fit(self, raster_filepath: str | Path) -> Self:
        raster_filepath = Path(raster_filepath)
        logger.info("[%s] %s - start fit...", self.__class__.__name__, raster_filepath.name)
        fit_res = self._fit(raster_filepath)
        self.__raster_filepath_ = raster_filepath
        logger.info(
            "[%s] %s - finish fit : [%s]",
            self.__class__.__name__,
            raster_filepath.name,
            "FAILED" if self.is_failed else "SUCCESS",
        )
        return fit_res

    @abstractmethod
    def _fit(self, raster_filepath: Path) -> Self: ...


class RestitutionStrategy(FittingClass):
    @abstractmethod
    def transform(self, output_path: str | Path) -> None: ...

    @property
    @abstractmethod
    def transformation_(self) -> "Transformation": ...


@dataclass
class Transformation:
    raster_filepath: Path
    deformation: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    crop_offset: tuple[float, float] = (0, 0)
    output_size: tuple[int, int] = (0, 0)

    def inverse_remap(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
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
