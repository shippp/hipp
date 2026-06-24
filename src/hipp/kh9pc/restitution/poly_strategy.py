from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from skimage.transform import ThinPlateSplineTransform
from sklearn.linear_model import RANSACRegressor

from hipp.image import SubImage, remap_tif_blockwise
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import detect_ruptures, fit_ransac_poly
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector


@dataclass
class PolyResult:
    ruptures_local: NDArray[np.integer]
    ruptures_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_image: SubImage


@dataclass
class PolyStrategy(RestitutionStrategy):
    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10
    polynomial_degree: int = 2
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    min_inliers_threshold: float = 0.5
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, PolyResult] = {}
        self.__transformation_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        return min(self.top_.inlier_ratio, self.bottom_.inlier_ratio) < self.min_inliers_threshold

    @property
    def top_(self) -> PolyResult:
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> PolyResult:
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    @property
    def metrics_(self) -> dict[str, Any]:
        """Detection metrics: expected vs. detected height, merged with vertical detector metrics."""
        expected_height = KH9ImageSpec.from_raster_filepath(self.raster_filepath_).expected_size[1]
        x = np.linspace(*self.vertical_detector.edges_, self.grid_shape[0]).reshape(-1, 1)
        heights = self.bottom_.model.predict(x) - self.top_.model.predict(x)
        return {
            **self.vertical_detector.metrics_,
            "expected_height": expected_height,
            "detected_height": float(np.mean(heights)),
            "detected_height_std": float(np.std(heights)),
        }

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_off, _ = self.vertical_detector.edges_
        window_width = self.vertical_detector.detected_width_

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                self._results[side] = self._process_side(sub_image, side)

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> PolyResult:
        res = []
        for i in range(sub_image.band.shape[1]):
            ruptures = detect_ruptures(sub_image.band[:, i], self.background_threshold, reverse_scan=(side == "top"))
            if len(ruptures) > 0:
                res.append((i, ruptures[0]))

        if not res:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        ruptures_local = np.array(res)
        ruptures_global = sub_image.to_global(ruptures_local)

        model = fit_ransac_poly(
            ruptures_global[:, 0],
            ruptures_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())

        x_sample = np.linspace(
            sub_image.window.col_off, sub_image.window.col_off + sub_image.window.width, self.grid_shape[0]
        )
        y_global_pred = model.predict(x_sample.reshape(-1, 1)).ravel()
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([x_sample, y_distortion])

        return PolyResult(
            ruptures_local=ruptures_local,
            ruptures_global=ruptures_global.astype(int),
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_image=sub_image,
        )

    def _compute_transformation(self) -> Transformation:
        left, right = self.vertical_detector.edges_
        detected_width = self.vertical_detector.detected_width_
        output_width = self.output_width or detected_width

        x = np.linspace(left, right, self.grid_shape[0])

        y_top_src = self.top_.model.predict(x.reshape(-1, 1))
        y_bot_src = self.bottom_.model.predict(x.reshape(-1, 1))

        top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))
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
