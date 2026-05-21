import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RANSACRegressor

from hipp.kh9pc.utils import SubImage

logger = logging.getLogger(__name__)

########################################################################
#                           ABSTRACT CLASS
########################################################################


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


########################################################################
#                           DATA CLASS
########################################################################
@dataclass
class Transformation:
    raster_filepath: Path
    deformation: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    crop_offset: tuple[float, float] = (0, 0)
    output_size: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def inverse_remap(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
        coords = coords + np.array([self.crop_offset[0], self.crop_offset[1]], dtype=coords.dtype)
        return self.deformation(coords)


@dataclass
class VerticalEdgeResult:
    position: int
    rupture_local: int
    sub_image: SubImage
    profile: NDArray[np.integer]
    gradient_pct: float


@dataclass
class PolyResult:
    ruptures_local: NDArray[np.integer]
    ruptures_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_image: SubImage


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


@dataclass
class FiducialFilteringResult:
    boxes_all: NDArray[np.int_]  # shape (N, 4) — (x, y, w, h) in global coordinates
    scores_all: NDArray[np.float64]
    template_ids_all: NDArray[np.int_]
    cx: NDArray[np.floating]
    cy: NDArray[np.floating]
    residuals: NDArray[np.floating]
    labels: NDArray[np.integer]
    best_cluster_label: int
    best_eps: float
    best_weight: float
    cluster_scores: dict[int, float] = field(default_factory=dict)  # spatial score per cluster at best params


@dataclass
class FiducialResult:
    centers: NDArray[np.int_] # shape (N, 2) — (x, y) in global coordinates
    poly: np.polynomial.Polynomial
    distortion: NDArray[np.floating]
    boxes: NDArray[np.int_]  # shape (N, 4) — (x, y, w, h) in global coordinates
    scores: NDArray[np.float64]
    template_ids: NDArray[np.int_]
    width_coverage: float = 0.0  # fraction of the detected image width covered by fiducials
    filtering: FiducialFilteringResult | None = None


@dataclass
class ImageAlignment:
    """Alignment result for a single image in a sequential alignment chain.

    Attributes
    ----------
    image_path : Path
        Path to the image file.
    relative_transform : np.ndarray
        3x3 homogeneous transformation matrix relative to the previous image
        (identity for the first/reference image).
    absolute_transform : np.ndarray
        3x3 homogeneous transformation matrix in the global/mosaic coordinate system,
        accumulated from the reference image.
    n_matches : int
        Total number of ORB keypoint matches found before RANSAC filtering
        (0 for the reference image).
    n_inliers : int
        Number of inlier matches kept after RANSAC filtering
        (0 for the reference image).
    """

    image_path: Path
    relative_transform: np.ndarray
    absolute_transform: np.ndarray
    n_matches: int
    n_inliers: int
