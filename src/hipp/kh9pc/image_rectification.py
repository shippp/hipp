from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from skimage.transform import AffineTransform, ThinPlateSplineTransform

from hipp.image import remap_tif_blockwise
from hipp.kh9pc.rectification_strategy import PolyRectificationStrategy, RectificationStrategy


@dataclass
class TransformStrategy(ABC):
    lowres_step: int | None
    block_size: int

    @abstractmethod
    def fit(self, src_points: NDArray[np.float32], dst_points: NDArray[np.float32]) -> None: ...

    @property
    @abstractmethod
    def inverse_remap_function(self) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]: ...


class TransformTPS(TransformStrategy):
    def __init__(self, lowres_step: int = 100, block_size: int = 2**13):
        super().__init__(lowres_step=lowres_step, block_size=block_size)
        self._transform: ThinPlateSplineTransform | None = None

    def fit(self, src_points: NDArray[np.float32], dst_points: NDArray[np.float32]) -> None:
        result = ThinPlateSplineTransform().from_estimate(dst_points, src_points)

        if isinstance(result, ThinPlateSplineTransform):
            self._transform = result
        else:
            raise RuntimeError("TPS estimation failed")

    @property
    def inverse_remap_function(self) -> ThinPlateSplineTransform:
        if self._transform is None:
            raise RuntimeError("Call fit() before accessing inverse_remap_function")
        return self._transform


class TransformAffine(TransformStrategy):
    def __init__(self, lowres_step: int | None = None, block_size: int = 256):
        super().__init__(lowres_step=lowres_step, block_size=block_size)
        self._transform: AffineTransform | None = None

    def fit(self, src_points: NDArray[np.float32], dst_points: NDArray[np.float32]) -> None:
        self._transform = AffineTransform().from_estimate(dst_points, src_points)

    @property
    def inverse_remap_function(self) -> AffineTransform:
        if self._transform is None:
            raise RuntimeError("Call fit() before accessing inverse_remap_function")
        return self._transform


class ImageRectification:
    def __init__(
        self,
        strategy: RectificationStrategy | None = None,
        transformation: TransformStrategy | None = None,
    ):
        self.strategy = strategy if strategy is not None else PolyRectificationStrategy()
        self.transformation = transformation if transformation is not None else TransformTPS()
        self._raster_path: Path | None = None
        self._src_points: NDArray[np.float32] | None = None
        self._dst_points: NDArray[np.float32] | None = None
        self._output_size: tuple[int, int] | None = None

    def fit(self, raster: str | Path) -> "ImageRectification":
        self._raster_path = Path(raster)

        self.strategy.fit(self._raster_path)

        src_grid, dst_grid, self._output_size = self.strategy.compute_grid()
        self._src_points = src_grid.reshape(-1, 2).astype(np.float32)
        self._dst_points = dst_grid.reshape(-1, 2).astype(np.float32)

        self.transformation.fit(self._src_points, self._dst_points)

        return self

    def generate_qc_report(self, output_file: str | Path) -> None:
        if self._raster_path is None:
            raise RuntimeError("Call fit() before generate_qc_report()")

        self.strategy.generate_qc_report(output_file)

    def transform(self, output_file: str | Path) -> None:
        if self._raster_path is None or self._output_size is None:
            raise RuntimeError("Call fit() before transform()")

        remap_tif_blockwise(
            self._raster_path,
            Path(output_file),
            self.transformation.inverse_remap_function,
            self._output_size,
            block_size=self.transformation.block_size,
            lowres_step=self.transformation.lowres_step,
        )
