from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class RectificationStrategy(ABC):
    """Abstract base class for image rectification strategies.

    Each strategy is responsible for the complete ROI detection pipeline:
    vertical edge detection, horizontal edge detection, and quality control.
    Implementers receive the raster path directly in :meth:`fit` and decide
    internally how to detect vertical and horizontal boundaries.
    """

    @abstractmethod
    def fit(self, raster_filepath: str | Path) -> "RectificationStrategy":
        """Detect the image boundaries (vertical and horizontal).

        Parameters
        ----------
        raster_filepath:
            Path to the input raster file.

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        """Return the control point grids for TPS rectification.

        Must be called after :meth:`fit`.

        Returns
        -------
        src_points : np.ndarray
            Distorted source coordinates, shape ``(N, 2)`` or ``(grid_w, grid_h, 2)``.
        dst_points : np.ndarray
            Regular destination coordinates, same shape as *src_points*.
        output_size : tuple[int, int]
            Expected ``(width, height)`` of the rectified raster.
        """
        ...

    @abstractmethod
    def generate_qc_report(self, output_path: str | Path) -> None:
        """Save a PDF QC report for this strategy.

        Parameters
        ----------
        output_path:
            Destination path for the PDF file. Parent directories are created
            if they do not exist.
        """
        ...
