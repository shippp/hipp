import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Self

import numpy as np
from numpy.typing import NDArray

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
