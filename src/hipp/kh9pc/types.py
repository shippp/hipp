import functools
import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Self

import numpy as np
import pandas as pd
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

    @abstractmethod
    def get_transformation(
        self, output_width: int | None = None, output_height: int | None = 22064
    ) -> "Transformation": ...


@dataclass
class TaskResult:
    func_name: str
    args: dict[str, Any]
    status: str  # "ran" | "skipped" | "error"
    started_at: datetime
    duration: float  # seconds
    error: str | None = None


@dataclass
class StepResult:
    name: str
    status: str  # "ran" | "skipped" | "failed"
    started_at: datetime
    duration: float  # seconds
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class Task:
    registry: list[TaskResult] = []

    def __init__(self, input: list[str] | str | None = None, output: list[str] | str | None = None) -> None:
        self.inputs = Task._normalize_io(input)
        self.outputs = Task._normalize_io(output)

    @classmethod
    def clear_registry(cls) -> None:
        cls.registry.clear()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        for arg in self.inputs:
            if arg not in param_names:
                raise ValueError(f"[{func.__name__}] input '{arg}' not in function signature")
        for arg in self.outputs:
            if arg not in param_names:
                raise ValueError(f"[{func.__name__}] output '{arg}' not in function signature")

        @functools.wraps(func)
        def wrapper(*args: Any, overwrite: bool = False, **kwargs: Any) -> Any:
            started_at = datetime.now()
            t0 = time.perf_counter()

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_args = dict(bound.arguments)

            outputs_exist = bool(self.outputs) and all(Path(bound_args[arg]).exists() for arg in self.outputs)

            if outputs_exist and not overwrite:
                logger.info("%s: already done, skipping", func.__name__)
                Task.registry.append(
                    TaskResult(
                        func_name=func.__name__,
                        args={**bound_args, "overwrite": overwrite},
                        status="skipped",
                        started_at=started_at,
                        duration=0.0,
                    )
                )
                return None

            missing = [str(bound_args[arg]) for arg in self.inputs if not Path(bound_args[arg]).exists()]
            if missing:
                raise FileNotFoundError(f"[{func.__name__}] missing inputs: {missing}")

            for arg in self.outputs:
                Path(bound_args[arg]).parent.mkdir(parents=True, exist_ok=True)

            try:
                result = func(*args, **kwargs)
                Task.registry.append(
                    TaskResult(
                        func_name=func.__name__,
                        args={**bound_args, "overwrite": overwrite},
                        status="ran",
                        started_at=started_at,
                        duration=time.perf_counter() - t0,
                    )
                )
                return result
            except Exception as exc:
                Task.registry.append(
                    TaskResult(
                        func_name=func.__name__,
                        args={**bound_args, "overwrite": overwrite},
                        status="error",
                        started_at=started_at,
                        duration=time.perf_counter() - t0,
                        error=f"{exc.__class__.__name__} : {str(exc)}",
                    )
                )
                raise

        return wrapper

    @staticmethod
    def _normalize_io(value: list[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


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
class FiducialResult:
    candidates: pd.DataFrame



