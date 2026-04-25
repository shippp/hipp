from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.windows import Window

from hipp.image import remap_tif_blockwise
from hipp.kh9pc.types import CollimationResult, RestitutionStrategy, Transformation
from hipp.kh9pc.utils import SubImage, build_inverse_map, detect_collimation_peak, fit_ransac_poly, wrap_ransac_model_1d
from hipp.kh9pc.vertical_detector import VerticalDetector


@dataclass
class CollimationStrategy(RestitutionStrategy):
    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    stride: int = 10
    height_fraction: float = 0.15
    max_width_peak: int = 200
    collimation_line_dist: int = 21770
    min_inliers_treshold: float = 0.5

    def __post_init__(self) -> None:
        super().__init__()
        self.__top_: CollimationResult | None = None
        self.__bottom_: CollimationResult | None = None

    @property
    def is_failed(self):
        return min(self.top_.inlier_ratio, self.bottom_.inlier_ratio) < self.min_inliers_treshold

    @property
    def top_(self) -> CollimationResult:
        if self.__top_ is None:
            raise RuntimeError("Call fit() before")
        return self.__top_

    @property
    def bottom_(self) -> CollimationResult:
        if self.__bottom_ is None:
            raise RuntimeError("Call fit() before")
        return self.__bottom_

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_off, col_end = self.vertical_detector.edges_
        window_width = col_end - col_off

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_img = SubImage(src, window, out_shape, resampling=Resampling.average)
                setattr(self, f"_CollimationStrategy__{side}_", self._process_side(side, sub_img))

        return self

    def _process_side(self, side: str, sub_img: SubImage) -> CollimationResult:
        _, w = sub_img.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)
        for col in range(w):
            vec = sub_img.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        peaks_global = sub_img.to_global(peaks_local).astype(int)

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
            sub_img=sub_img,
        )

    def get_transformation(self, output_width: int | None = None, output_height: int | None = 22064) -> Transformation:
        left, right = self.vertical_detector.edges_
        detected_width = right - left
        output_width = output_width or detected_width

        x = np.linspace(left, right, self.grid_shape[0])

        f_top_src = wrap_ransac_model_1d(self.top_.model)
        f_bot_src = wrap_ransac_model_1d(self.bottom_.model)

        top, bot = int(np.median(f_top_src(x))), int(np.median(f_bot_src(x)))
        detected_height = bot - top
        output_height = output_height or detected_height

        def f_top_ref(x):
            return np.full_like(x, top, dtype=np.float32)

        def f_bot_ref(x):
            return np.full_like(x, bot, dtype=np.float32)

        deformation = build_inverse_map(f_top_src, f_bot_src, f_top_ref, f_bot_ref)

        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2

        crop_offset = (
            int(left - pad_x),
            int(top - pad_y),
        )

        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )

    def transform(self, output_path: str | Path) -> None:
        tf = self.get_transformation()
        remap_tif_blockwise(tf.raster_filepath, output_path, tf.inverse_remap, tf.output_size)
