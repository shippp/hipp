from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hipp.kh9pc.rectification_strategy.base import RectificationStrategy


class CollimationRectificationStrategy(RectificationStrategy):
    """Collimation-line strategy: uses collimation lines as top and bottom boundaries.

    Instead of detecting intensity ruptures, polynomial models are fitted
    directly to the collimation lines (the physical reference marks printed on
    KH-9 film) to derive the top and bottom boundaries.

    Parameters
    ----------
    polynomial_degree:
        Degree of the polynomial fitted to the collimation line points.
    ransac_residual_threshold:
        Maximum residual (in pixels) for a point to be considered an inlier.
    ransac_max_trials:
        Maximum number of RANSAC iterations.
    img_height:
        Target height of the rectified image in pixels. If *None*, estimated
        from the collimation line separation.
    grid_shape:
        Number of control points along ``(width, height)``.
    """

    def __init__(
        self,
        polynomial_degree: int = 5,
        ransac_residual_threshold: float = 80.0,
        ransac_max_trials: int = 1000,
        img_height: int | None = None,
        grid_shape: tuple[int, int] = (100, 50),
    ):
        self.polynomial_degree = polynomial_degree
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials = ransac_max_trials
        self.img_height = img_height
        self.grid_shape = grid_shape
        self.top: object | None = None
        self.bottom: object | None = None
        self.raster_filepath_: Path | None = None

    def fit(self, raster_filepath: str | Path) -> "CollimationRectificationStrategy":
        raise NotImplementedError

    def compute_grid(self) -> tuple[NDArray[np.generic], NDArray[np.generic], tuple[int, int]]:
        raise NotImplementedError

    def generate_qc_report(self, output_path: str | Path) -> None:
        raise NotImplementedError
