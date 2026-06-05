from pathlib import Path

import cv2
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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


def compute_gradient_pcts(
    profile: NDArray[np.number],
    ruptures: NDArray[np.integer],
    window_size: int,
    use_max: bool,
) -> list[float]:
    """Score each rupture by its local gradient relative to the global gradient extremum.

    For a rising edge (use_max=True)  : score = max(window_gradient) / max(profile_gradient)
    For a falling edge (use_max=False): score = min(window_gradient) / min(profile_gradient)
    """
    gradient = np.diff(profile.astype(np.float32))
    global_stat = float(np.max(gradient)) if use_max else float(np.min(gradient))
    if global_stat == 0:
        return [0.0] * len(ruptures)

    pcts: list[float] = []
    for r in ruptures:
        w = np.diff(profile[max(0, r - window_size) : r + window_size].astype(np.float32))
        local_stat = float(np.max(w)) if use_max else float(np.min(w))
        pcts.append(local_stat / global_stat)
    return pcts


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

    min_samples = min(degree * 3, len(x))
    ransac = RANSACRegressor(
        poly_model, residual_threshold=residual_threshold, min_samples=min_samples, max_trials=max_trials
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


def measure_circularity(image: NDArray[np.uint8]) -> tuple[float, float]:
    """Measure how circular the main shape in a binary or grayscale image is.

    Thresholds the image (Otsu), finds the largest external contour, and
    returns its circularity score and equivalent radius.

    Parameters
    ----------
    image : NDArray[np.uint8], shape (H, W)
        Grayscale image. Does not need to be pre-thresholded.

    Returns
    -------
    circularity : float
        Score in ``[0, 1]``, where ``1.0`` is a perfect circle.
        Returns ``0.0`` if no contour is found or the perimeter is zero.
    radius : float
        Equivalent radius estimated from contour area: ``sqrt(area / pi)``.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return 0.0, 0.0

    circularity = float(4 * np.pi * area / (perimeter**2))
    radius = float(np.sqrt(area / np.pi))
    return circularity, radius


def create_circle_template(radius: int, canvas_size: int | None = None) -> cv2.typing.MatLike:
    if canvas_size is None:
        canvas_size = 2 * radius + 1
    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cx = cy = canvas_size // 2
    y, x = np.ogrid[:canvas_size, :canvas_size]
    img[(x - cx) ** 2 + (y - cy) ** 2 <= radius**2] = 255
    return img


def compute_spatial_regularization_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a spatial regularity score from consecutive 2D point spacing.

    The score evaluates how regularly points are distributed along a
    2D trajectory. Points are first ordered along the x-axis, then
    Euclidean distances between consecutive points are computed.

    The metric is based on the coefficient of variation (CV) of the
    inter-point distances:

        CV = std(distances) / mean(distances)

    The final score is normalized into the range [0, 1]:

        score = 1 / (1 + CV)

    Interpretation
    --------------
    - score ≈ 1:
        Highly regular spacing between points.
    - score ≈ 0:
        Highly irregular spacing.

    Parameters
    ----------
    x : np.ndarray
        1D array containing x coordinates.

    y : np.ndarray
        1D array containing y coordinates.

    Returns
    -------
    float
        Spatial regularity score in the range [0, 1].

    Raises
    ------
    ValueError
        If input arrays have different lengths or contain fewer than
        two points.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    if x.shape[0] < 2:
        raise ValueError("At least two points are required.")

    order = np.argsort(x)

    x_sorted = x[order]
    y_sorted = y[order]

    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)

    inter_point_distances = np.hypot(dx, dy)

    mean_distance = np.mean(inter_point_distances)

    if mean_distance == 0:
        return 0.0

    coefficient_of_variation = np.std(inter_point_distances) / mean_distance

    score = 1.0 / (1.0 + coefficient_of_variation)

    return float(score)


def mean_patch_from_centers(
    src: str | Path | rasterio.DatasetReader,
    centers: np.ndarray,
    half_size: int = 50,
) -> np.ndarray | None:
    """Compute the mean image patch (band 1) around a set of pixel centers.

    Uses an incremental float64 accumulator so peak memory is O(patch_size²)
    regardless of the number of centers. Out-of-bounds regions are zero-padded
    before averaging (same semantics as storing full zero-padded patches).

    Centers that fall entirely outside the raster are silently skipped.

    Parameters
    ----------
    src:
        Rasterio dataset, or path to a raster file, to read from.
    centers:
        Pixel coordinates (x, y) of patch centers, shape (N, 2).
    half_size:
        Half-side of the square patch in pixels; each patch is
        ``(2*half_size) × (2*half_size)``.

    Returns
    -------
    Float32 array of shape ``(2*half_size, 2*half_size)``, or ``None`` if no
    valid patch was found.
    """
    if not isinstance(src, rasterio.DatasetReader):
        with rasterio.open(src) as opened:
            return mean_patch_from_centers(opened, centers, half_size)

    size = 2 * half_size
    accumulator = np.zeros((size, size), dtype=np.float64)
    count = 0

    x0s: np.ndarray = centers[:, 0].astype(np.intp) - half_size
    y0s: np.ndarray = centers[:, 1].astype(np.intp) - half_size

    for x0, y0 in zip(x0s, y0s):
        x0c = max(0, int(x0))
        y0c = max(0, int(y0))
        x1c = min(src.width, int(x0) + size)
        y1c = min(src.height, int(y0) + size)
        if x1c <= x0c or y1c <= y0c:
            continue
        patch = src.read(1, window=Window(x0c, y0c, x1c - x0c, y1c - y0c))
        accumulator[y0c - y0 : y0c - y0 + patch.shape[0], x0c - x0 : x0c - x0 + patch.shape[1]] += patch
        count += 1

    return (accumulator / count).astype(np.float32) if count > 0 else None


class SubImage:
    def __init__(
        self,
        raster: str | Path | rasterio.DatasetReader,
        window: Window | None,
        out_shape: tuple[int, int, int] | None = None,
        resampling: Resampling = Resampling.average,
    ):
        if isinstance(raster, rasterio.DatasetReader):
            self._setup(raster, window, out_shape, resampling)
        else:
            with rasterio.open(raster) as src:
                self._setup(src, window, out_shape, resampling)

    def _setup(
        self,
        src: rasterio.DatasetReader,
        window: Window | None,
        out_shape: tuple[int, int, int] | None,
        resampling: Resampling,
    ) -> None:
        self.window = window or Window(0, 0, src.width, src.height)
        self.band = src.read(1, window=self.window, out_shape=out_shape, resampling=resampling)

        actual_shape = self.band.shape  # (height, width) after read
        self.out_shape = (1, actual_shape[0], actual_shape[1])
        self._scale = np.array(
            [self.window.width / actual_shape[1], self.window.height / actual_shape[0]], dtype=np.float64
        )
        self._offset = np.array([self.window.col_off, self.window.row_off], dtype=np.float64)

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
