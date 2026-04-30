from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Sequence

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from hipp.image import match_multiple_templates
from hipp.kh9pc.restitution_strategy.poly_strategy import PolyStrategy
from hipp.kh9pc.types import FiducialResult, RestitutionStrategy, Transformation
from hipp.kh9pc.utils import SubImage

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class FiducialStrategy(RestitutionStrategy):
    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    template_paths: Sequence[str | Path] = field(default_factory=lambda: list(_TEMPLATE_DIR.glob("*.png")))
    block_width: int = 512
    threshold: float = 0.7
    nms_threshold: float = 0.1

    def __post_init__(self) -> None:
        super().__init__()
        self.__templates = [img for p in self.template_paths if (img := cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) is not None]
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

    @property
    def is_failed(self) -> bool:
        return False

    @property
    def transformation_(self) -> Transformation:
        raise NotImplementedError

    def transform(self, output_path: str | Path) -> None:
        raise NotImplementedError

    def _fit(self, raster_filepath: Path) -> Self:
        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)

        col_start, col_end = self.poly_strategy.vertical_detector.edges_
        margin_fraction = 0.1

        side_data: dict[str, tuple[list[list[int]], list[float], list[int]]] = {
            "top": ([], [], []),
            "bottom": ([], [], []),
        }

        with rasterio.open(raster_filepath) as src:
            for cursor in range(col_start, col_end, self.block_width):
                w_start = max(col_start, int(cursor - self.block_width * margin_fraction))
                w_end = min(col_end, int(cursor + self.block_width * (1 + margin_fraction)))
                w_width = w_end - w_start
                w_center = w_start + w_width // 2

                top_row = int(self.poly_strategy.top_.model.predict(np.array([[w_center]])).flat[0])
                bot_row = int(self.poly_strategy.bottom_.model.predict(np.array([[w_center]])).flat[0])

                for side, window in {
                    "top": Window(w_start, 0, w_width, top_row),
                    "bottom": Window(w_start, bot_row, w_width, src.height - bot_row),
                }.items():
                    if window.height <= 0 or window.width <= 0:
                        continue
                    sub_image = SubImage(src, window)

                    _boxes, _scores, _template_ids = match_multiple_templates(
                        image=sub_image.band,
                        templates=self.__templates,
                        threshold=self.threshold,
                        nms_threshold=self.nms_threshold,
                    )

                    acc_boxes, acc_scores, acc_ids = side_data[side]
                    for box in _boxes:
                        x, y, w, h = box
                        gx, gy = sub_image.to_global(np.array([x, y], dtype=np.float64))
                        acc_boxes.append([int(gx), int(gy), w, h])
                    acc_scores.extend(_scores)
                    acc_ids.extend(_template_ids)

        for side in ("top", "bottom"):
            boxes, scores, template_ids = side_data[side]

            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes, scores, score_threshold=self.threshold, nms_threshold=self.nms_threshold
                )
                indices_np = np.array(indices).reshape(-1)
                boxes = [boxes[i] for i in indices_np]
                scores = [scores[i] for i in indices_np]
                template_ids = [template_ids[i] for i in indices_np]

            setattr(self, f"_FiducialStrategy__{side}_", FiducialResult(boxes, scores, template_ids))

        return self
