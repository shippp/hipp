"""
Copyright (c) 2026 HIPP developers
Description: Abstract base classes and shared utilities for KH-9 PC detection steps.
    Defines the FittingClass / RestitutionStrategy protocol, the Transformation dataclass,
    and the low-level helpers (RANSAC polynomial fitting, rupture detection) shared across
    all concrete strategy implementations.
"""

import logging
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from hipp.tools import atomic_output

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Raised when no valid detections are found during fitting."""


class JsonModel(BaseModel):
    """Pydantic model with a JSON round-trip."""

    @classmethod
    def load(cls, source: str | Path | Self) -> Self:
        """Return *source* as is if already an instance, otherwise read it from the JSON file."""
        if isinstance(source, (str, Path)):
            return cls.model_validate_json(Path(source).read_text())
        return source

    def save(self, output_file: str | Path) -> None:
        """Write the model as indented JSON, creating parent directories."""
        # atomic so an interrupted write never leaves a truncated file that skip_existing would take as done
        with atomic_output(output_file) as tmp:
            tmp.write_text(self.model_dump_json(indent=2))


class DetectionResult(JsonModel):
    """Base of every detection result: a JsonModel with a confidence verdict."""

    @property
    def failure_reason(self) -> str | None:
        """Why the result is not trustworthy, or None if it is."""
        return None

    @property
    def is_failed(self) -> bool:
        """True if the result exists but is not trustworthy."""
        return self.failure_reason is not None


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
