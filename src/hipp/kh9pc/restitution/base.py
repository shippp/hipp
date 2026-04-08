from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Self

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray


class BaseEstimator(ABC):
    def __init__(self) -> None:
        self.raster_filepath_: Path | None = None
        self.fitting_time_: float | None = None
        self.fitted_at_: datetime | None = None

    def fit(self, raster_filepath: str | Path) -> Self:
        t0 = time.perf_counter()
        self.fitted_at_ = datetime.now()
        result = self._fit(raster_filepath)

        self.raster_filepath_ = Path(raster_filepath)
        self.fitting_time_ = time.perf_counter() - t0

        return result

    @abstractmethod
    def _fit(self, raster_filepath: str | Path) -> Self: ...

    def _get_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not (k.startswith("_") or k.endswith("_"))}

    def __str__(self) -> str:
        if self.raster_filepath_ is None or self.fitting_time_ is None or self.fitted_at_ is None:
            return f"{self.__class__.__name__} (not fitted)"

        return "\n".join(
            [
                self.__class__.__name__,
                "",
                f"Image : {self.raster_filepath_.name}",
                f"Fitted at : {self.fitted_at_.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Fitting time : {self.fitting_time_:.2f} s",
                "",
                "Parameters",
                *[f"  {k:25}: {v}" for k, v in self._get_params().items()],
            ]
        )

    @property
    def raster_filepath(self) -> Path:
        if self.raster_filepath_ is None:
            raise RuntimeError("need to call the fit() method before")
        return self.raster_filepath_

    @property
    def is_fitted(self) -> bool:
        return self.raster_filepath_ is not None


class QCMixin(ABC):
    """Mixin that adds QC figure generation and PDF report export.

    Subclasses must implement :meth:`get_qc_figures`. The concrete
    :meth:`generate_qc_report` method is provided here and relies on
    :meth:`__str__` and :attr:`is_fitted` being available on the instance.
    """

    @abstractmethod
    def get_qc_figures(self) -> list[Figure]:
        """Return the list of quality-control figures.

        Must be called after fitting.
        """
        ...

    def generate_qc_report(self, output_path: str | Path) -> None:
        """Save a PDF QC report.

        Parameters
        ----------
        output_path:
            Destination path for the PDF file. Parent directories are created
            if they do not exist.
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(output_path) as pdf:
            for fig in self.get_qc_figures():
                pdf.savefig(fig)
                plt.close(fig)


class RectificationStrategy(BaseEstimator, QCMixin):
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

        Returns the *detected* content dimensions — i.e. the natural size of the
        rectified region as determined by the strategy. To place this content on a
        differently-sized canvas (with margins, fixed dimensions, etc.) pass the
        returned values to an :class:`~hipp.kh9pc.restitution.output_size.OutputSize`
        instance::

            src, dst, detected = strategy.compute_grid()
            src, dst, final    = MarginSize(top=200, bottom=200).apply(src, dst, detected)
            transformer.fit(src, dst, final)

        Returns
        -------
        src_points : np.ndarray, shape (N, 2)
            Distorted source coordinates.
        dst_points : np.ndarray, shape (N, 2)
            Regular destination coordinates normalised to ``[0, width] × [0, height]``.
        detected_size : tuple[int, int]
            ``(width, height)`` of the detected content region.
        """
        ...
