from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from scipy.ndimage import gaussian_filter1d

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from rasterio.windows import Window
import rasterio
from rasterio.warp import Resampling


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


def detect_collimation_peak(x: NDArray[np.number], max_peak_width: int, sigma: int = 2) -> int:
    smooth = gaussian_filter1d(x, sigma=sigma)

    grad = np.gradient(smooth)

    idx_max = np.argmax(grad)
    idx_min = np.argmin(grad)

    if abs(idx_max - idx_min) < max_peak_width and idx_max != idx_min:
        w_start = min(idx_max, idx_min)
        w_end = max(idx_max, idx_min)
        idx = np.argmax(smooth[w_start:w_end]) + w_start
    else:
        idx = np.argmax(smooth)  # fallback

    return int(idx)


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
    ransac = RANSACRegressor(
        poly_model, residual_threshold=residual_threshold, min_samples=degree * 3, max_trials=max_trials
    )
    ransac.fit(x.reshape(-1, 1), y)
    return ransac


def generate_qc_report(output_path: str | Path, figures: list[Figure]) -> None:
    """Save a list of matplotlib figures to a PDF QC report.

    Parameters
    ----------
    output_path : str or Path
        Destination path for the PDF file. Parent directories are created if they do not exist.
    figures : list[Figure]
        Figures to include in the report. Each figure becomes one page. All figures are closed after saving.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)


def make_summary_figure(lines: list[str]) -> Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    first = True
    for line in lines:
        if first:
            fig.text(0.5, y, line, ha="center", va="top", fontsize=12, fontweight="bold")
            first = False
            y -= 0.04
        elif line == "":
            y -= 0.01
        else:
            fig.text(0.1, y, line, ha="left", va="top", fontsize=8, family="monospace")
            y -= 0.025
    return fig


class SubImage:
    def __init__(
        self,
        raster: str | Path | rasterio.DatasetReader,
        window: Window,
        out_shape: tuple[int, int, int],
        resampling: Resampling = Resampling.average,
    ):
        self.window = window
        self.out_shape = out_shape

        if isinstance(raster, rasterio.DatasetReader):
            self.band = raster.read(1, window=window, out_shape=out_shape, resampling=resampling)
        else:
            with rasterio.open(raster) as src:
                self.band = src.read(1, window=window, out_shape=out_shape, resampling=resampling)

        self._scale = np.array([window.width / out_shape[2], window.height / out_shape[1]], dtype=np.float64)
        self._offset = np.array([window.col_off, window.row_off], dtype=np.float64)

    def to_global(self, pts: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert local sub-image pixel coordinates to global raster coordinates.

        Parameters
        ----------
        pts : ndarray of shape (2,) or (n, 2)
            Point(s) in local coordinates as [x, y] (column, row).

        Returns
        -------
        ndarray of same shape
            Corresponding [x, y] coordinates in the full raster.
        """
        return pts * self._scale + self._offset

    def to_local(self, pts: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert global raster pixel coordinates to local sub-image coordinates.

        Parameters
        ----------
        pts : ndarray of shape (2,) or (n, 2)
            Point(s) in global coordinates as [x, y] (column, row).

        Returns
        -------
        ndarray of same shape
            Corresponding [x, y] coordinates in the sub-image.
        """
        return (pts - self._offset) / self._scale
