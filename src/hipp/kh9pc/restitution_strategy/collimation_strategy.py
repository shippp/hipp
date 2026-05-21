from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.windows import Window
from skimage.transform import ThinPlateSplineTransform

from hipp.image import remap_tif_blockwise
from hipp.kh9pc.restitution_strategy.poly_strategy import PolyStrategy
from hipp.kh9pc.types import DEFAULT_OUTPUT_HEIGHT, CollimationResult, RestitutionStrategy, Transformation
from hipp.kh9pc.utils import SubImage, detect_collimation_peak, fit_ransac_poly


@dataclass
class CollimationStrategy(RestitutionStrategy):
    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    stride: int = 10
    refinement_fraction: float = 0.03
    max_width_peak: int = 200
    collimation_line_dist: int = 21770  # known physical distance between top/bottom collimation lines at nominal scan resolution
    min_inliers_threshold: float = 0.5
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, CollimationResult] = {}
        self.__transformation_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        return min(self.top_.inlier_ratio, self.bottom_.inlier_ratio) < self.min_inliers_threshold

    @property
    def top_(self) -> CollimationResult:
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> CollimationResult:
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)

        col_off, col_end = self.poly_strategy.vertical_detector.edges_
        window_width = col_end - col_off
        col_center = (col_off + col_end) // 2

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.refinement_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            top_edge = int(self.poly_strategy.top_.model.predict(np.array([[col_center]])).flat[0])
            bot_edge = int(self.poly_strategy.bottom_.model.predict(np.array([[col_center]])).flat[0])

            for side, window in {
                "top": Window(col_off, top_edge, window_width, window_height),
                "bottom": Window(col_off, bot_edge - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape, resampling=Resampling.average)
                self._results[side] = self._process_side(sub_image, side)

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> CollimationResult:
        _, w = sub_image.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)
        for col in range(w):
            vec = sub_image.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        peaks_global = sub_image.to_global(peaks_local).astype(int)

        model = fit_ransac_poly(
            peaks_global[:, 0],
            peaks_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())

        y_global_pred = model.predict(peaks_global[:, 0].reshape(-1, 1))
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([peaks_global[:, 0], y_distortion])

        return CollimationResult(
            peaks_local=peaks_local,
            peaks_global=peaks_global,
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_image=sub_image,
        )

    def _compute_transformation(self) -> Transformation:
        left, right = self.poly_strategy.vertical_detector.edges_
        detected_width = right - left
        output_width = self.output_width or detected_width

        x = np.linspace(left, right, self.grid_shape[0])

        y_top_src = self.top_.model.predict(x.reshape(-1, 1))
        y_bot_src = self.bottom_.model.predict(x.reshape(-1, 1))

        top = int(np.median(y_top_src))
        bot = top + self.collimation_line_dist
        detected_height = bot - top
        output_height = self.output_height or detected_height

        y_top_dst = np.full_like(x, top)
        y_bot_dst = np.full_like(x, bot)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important)
        deformation = ThinPlateSplineTransform().from_estimate(dst, src)

        # ---- CENTERING TO OUTPUT ----
        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2

        crop_offset = (int(left - pad_x), int(top - pad_y))

        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )

    def transform(self, output_path: str | Path) -> None:
        tf = self.transformation_

        remap_tif_blockwise(
            tf.raster_filepath,
            output_path,
            tf.inverse_remap,
            tf.output_size,
            block_size=2**13,
            lowres_step=100,
        )
