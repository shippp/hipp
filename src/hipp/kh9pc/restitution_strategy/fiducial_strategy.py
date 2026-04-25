from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from hipp.image import match_multi_templates
from hipp.kh9pc.types import FiducialResult, RestitutionStrategy
from hipp.kh9pc.utils import SubImage, create_circle_template, measure_circularity
from hipp.kh9pc.vertical_detector import VerticalDetector


@dataclass
class FiducialStrategy(RestitutionStrategy):
    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    height_fraction: float = 0.15
    block_width: int = 512
    threshold: float = 0.7
    template_fiducial_radii: list[int] = field(default_factory=lambda: [18, 25])
    mad_window: int = 11
    mad_threshold: float = 3.0

    def __post_init__(self) -> None:
        super().__init__()
        self.__top_: FiducialResult | None = None
        self.__bottom_: FiducialResult | None = None

    @property
    def top_(self) -> FiducialResult:
        if self.__top_ is None:
            raise RuntimeError("Call fit() before")
        return self.__top_

    @property
    def bottom_(self) -> FiducialResult:
        if self.__bottom_ is None:
            raise RuntimeError("Call fit() before")
        return self.__bottom_

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_start, col_end = self.vertical_detector.edges_
        template_dict = {
            f"circle_{r}": cv2.GaussianBlur(create_circle_template(r), (5, 5), 1.5)
            for r in self.template_fiducial_radii
        }
        margin = 2 * max(self.template_fiducial_radii)

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.height_fraction)

            for side, row_off in {"top": 0, "bottom": src.height - window_height}.items():
                blocks = []
                for x in range(col_start, col_end, self.block_width):
                    block_start = max(col_start, x - margin)
                    block_end = min(col_end, x + self.block_width + margin)
                    window = Window(block_start, row_off, block_end - block_start, window_height)
                    sub_img = SubImage(src, window)

                    df = match_multi_templates(sub_img.band, template_dict, margin, n_matches=2)

                    max_r = max(self.template_fiducial_radii)
                    h, w = sub_img.band.shape
                    circularities, radii = [], []
                    for row in df.itertuples():
                        cx, cy = int(row.x), int(row.y)
                        x0, x1 = max(0, cx - max_r), min(w, cx + max_r + 1)
                        y0, y1 = max(0, cy - max_r), min(h, cy + max_r + 1)
                        circ, rad = measure_circularity(sub_img.band[y0:y1, x0:x1])
                        circularities.append(circ)
                        radii.append(rad)
                    df["circularity"] = circularities
                    df["radius"] = radii

                    df[["x", "y"]] = sub_img.to_global(df[["x", "y"]].values).astype(int)
                    blocks.append(df)

                all_candidates = (
                    self._nms(pd.concat(blocks, ignore_index=True), radius=margin)
                    if blocks
                    else pd.DataFrame(columns=["template", "x", "y", "score"])
                )
                setattr(self, f"_FiducialStrategy__{side}_", FiducialResult(all_candidates))

        return self

    def _nms(self, df: pd.DataFrame, radius: int) -> pd.DataFrame:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        xy = df[["x", "y"]].values.astype(float)
        keep = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            if not keep[i]:
                continue
            dists = np.linalg.norm(xy[i + 1 :] - xy[i], axis=1)
            keep[i + 1 :][dists < radius] = False
        return df[keep].reset_index(drop=True)

    def transform(self, output_path: str | Path) -> None:
        raise NotImplementedError
