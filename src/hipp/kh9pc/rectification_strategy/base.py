from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import time
from typing import Self

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from hipp.kh9pc.utils import make_summary_figure


class FittedEstimator(ABC):
    """Abstract base class for estimators that track wall-clock fitting time.

    Subclasses implement :meth:`_fit` (the actual fitting logic). The public
    :meth:`fit` method wraps :meth:`_fit` with timing so that
    :attr:`fitting_time_` is always populated after a successful fit.
    """

    fitting_time_: float | None = None
    fitted_at_: datetime | None = None

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the estimator has been fitted."""
        ...

    @abstractmethod
    def _fit(self, raster_filepath: str | Path) -> Self:
        """Internal fitting logic to implement in subclasses."""
        ...

    def fit(self, raster_filepath: str | Path) -> Self:
        """Fit the estimator and record the wall-clock duration.

        Parameters
        ----------
        raster_filepath:
            Path to the input raster file.

        Returns
        -------
        self
        """
        t0 = time.perf_counter()
        self.fitted_at_ = datetime.now()
        result = self._fit(raster_filepath)
        self.fitting_time_ = time.perf_counter() - t0
        return result  # type: ignore[return-value]

    def _fitting_time_str(self) -> str:
        if self.fitting_time_ is None:
            return ""
        return f"Fitting time             : {self.fitting_time_:.2f} s"

    def _fitted_at_str(self) -> str:
        if self.fitted_at_ is None:
            return ""
        return f"Fitted at                : {self.fitted_at_.strftime('%Y-%m-%d %H:%M:%S')}"


class RectificationStrategy(FittedEstimator):
    """Abstract base class for image rectification strategies.

    Each strategy is responsible for the complete ROI detection pipeline:
    vertical edge detection, horizontal edge detection, and quality control.
    Implementers receive the raster path directly in :meth:`_fit` and decide
    internally how to detect vertical and horizontal boundaries.
    """

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
    def __str__(self) -> str:
        """Return a human-readable summary of parameters and fit results."""
        ...

    @abstractmethod
    def get_qc_figures(self) -> list[Figure]:
        """Return the list of quality-control figures for this strategy.

        Must be called after :meth:`fit`.
        """
        ...

    def generate_qc_report(self, output_path: str | Path) -> None:
        """Save a PDF QC report for this strategy.

        Parameters
        ----------
        output_path:
            Destination path for the PDF file. Parent directories are created
            if they do not exist.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before generate_qc_report()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(output_path) as pdf:
            summary_fig = make_summary_figure(str(self).splitlines())
            pdf.savefig(summary_fig)
            plt.close(summary_fig)

            for fig in self.get_qc_figures():
                pdf.savefig(fig)
                plt.close(fig)
