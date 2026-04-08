from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from skimage.transform import AffineTransform, ThinPlateSplineTransform

from hipp.image import remap_tif_blockwise


class ImageTransformer(ABC):
    """Base class for geometric transformations applied to rasters via blockwise remapping."""

    @abstractmethod
    def fit(
        self,
        src_points: NDArray[np.float32],
        dst_points: NDArray[np.float32],
        output_size: tuple[int, int],
    ) -> "ImageTransformer": ...

    def transform(self, input_file: str | Path, output_file: str | Path) -> None:
        """Apply the fitted transformation to a raster file.

        Parameters
        ----------
        input_file : str or Path
            Path to the input GeoTIFF.
        output_file : str or Path
            Path where the rectified GeoTIFF will be written.
        """
        if self._output_size is None:
            raise RuntimeError("Call fit() before transform()")

        remap_tif_blockwise(
            Path(input_file),
            Path(output_file),
            self._inverse_remap_function,
            self._output_size,
            block_size=self._block_size,
            lowres_step=self._lowres_step,
        )

    @property
    @abstractmethod
    def _inverse_remap_function(self) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]: ...

    @property
    @abstractmethod
    def _block_size(self) -> int: ...

    @property
    @abstractmethod
    def _lowres_step(self) -> int | None: ...

    @property
    @abstractmethod
    def _output_size(self) -> tuple[int, int] | None: ...


class ImageTransformerTps(ImageTransformer):
    """TPS (Thin Plate Spline) transformer for non-rigid geometric rectification.

    Parameters
    ----------
    lowres_step : int, optional
        Downsampling step used during blockwise remap. Default is 100.
    block_size : int, optional
        Block size for blockwise processing. Default is 2**13.
    """

    def __init__(self, lowres_step: int = 100, block_size: int = 2**13):
        self.__lowres_step = lowres_step
        self.__block_size = block_size
        self.__transform: ThinPlateSplineTransform | None = None
        self.__output_size: tuple[int, int] | None = None

    def fit(
        self,
        src_points: NDArray[np.float32],
        dst_points: NDArray[np.float32],
        output_size: tuple[int, int],
    ) -> "ImageTransformerTps":
        """Fit the TPS transform from source to destination control points.

        Parameters
        ----------
        src_points : NDArray[np.float32], shape (N, 2)
            Control points in the source (input) image.
        dst_points : NDArray[np.float32], shape (N, 2)
            Corresponding control points in the destination (output) image.
        output_size : tuple[int, int]
            Size of the output raster as (height, width).
        """
        result = ThinPlateSplineTransform().from_estimate(dst_points, src_points)
        if not isinstance(result, ThinPlateSplineTransform):
            raise RuntimeError("TPS estimation failed")
        self.__transform = result
        self.__output_size = output_size
        return self

    @property
    def _inverse_remap_function(self) -> ThinPlateSplineTransform:
        if self.__transform is None:
            raise RuntimeError("Call fit() before transform()")
        return self.__transform

    @property
    def _block_size(self) -> int:
        return self.__block_size

    @property
    def _lowres_step(self) -> int | None:
        return self.__lowres_step

    @property
    def _output_size(self) -> tuple[int, int] | None:
        return self.__output_size


class ImageTransformerAffine(ImageTransformer):
    """Affine transformer for rigid/linear geometric rectification.

    Parameters
    ----------
    block_size : int, optional
        Block size for blockwise processing. Default is 256.
    """

    def __init__(self, block_size: int = 256):
        self.__block_size = block_size
        self.__transform: AffineTransform | None = None
        self.__output_size: tuple[int, int] | None = None

    def fit(
        self,
        src_points: NDArray[np.float32],
        dst_points: NDArray[np.float32],
        output_size: tuple[int, int],
    ) -> "ImageTransformerAffine":
        """Fit the affine transform from source to destination control points.

        Parameters
        ----------
        src_points : NDArray[np.float32], shape (N, 2)
            Control points in the source (input) image.
        dst_points : NDArray[np.float32], shape (N, 2)
            Corresponding control points in the destination (output) image.
        output_size : tuple[int, int]
            Size of the output raster as (height, width).
        """
        self.__transform = AffineTransform().from_estimate(dst_points, src_points)
        self.__output_size = output_size
        return self

    @property
    def _inverse_remap_function(self) -> AffineTransform:
        if self.__transform is None:
            raise RuntimeError("Call fit() before transform()")
        return self.__transform

    @property
    def _block_size(self) -> int:
        return self.__block_size

    @property
    def _lowres_step(self) -> int | None:
        return None

    @property
    def _output_size(self) -> tuple[int, int] | None:
        return self.__output_size
