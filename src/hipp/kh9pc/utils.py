from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
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


def fit_ransac_poly(
    x: NDArray[np.generic],
    y: NDArray[np.generic],
    degree: int = 3,
    residual_threshold: float = 100,
    max_trials: int = 100,
) -> RANSACRegressor:
    """Fit a polynomial regression with RANSAC on 1D data. Returns the fitted RANSACRegressor."""
    poly_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree),
        LinearRegression(),
    )
    ransac = RANSACRegressor(
        poly_model, residual_threshold=residual_threshold, min_samples=degree * 3, max_trials=max_trials
    )
    ransac.fit(x.reshape(-1, 1), y)
    return ransac


def make_summary_figure(lines: list[str]) -> Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.95
    first = True
    for line in lines:
        if first:
            fig.text(0.5, y, line, ha="center", va="top", fontsize=16, fontweight="bold")
            first = False
            y -= 0.06
        elif line == "":
            y -= 0.02
        else:
            fig.text(0.1, y, line, ha="left", va="top", fontsize=10, family="monospace")
            y -= 0.04
    return fig
