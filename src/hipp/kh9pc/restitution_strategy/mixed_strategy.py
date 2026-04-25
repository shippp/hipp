from dataclasses import dataclass, field

from hipp.kh9pc.restitution_strategy.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution_strategy.flat_strategy import FlatStrategy
from hipp.kh9pc.restitution_strategy.poly_strategy import PolyStrategy
from hipp.kh9pc.types import RestitutionStrategy
from hipp.kh9pc.vertical_detector import VerticalDetector


@dataclass
class MixedStrategy(RestitutionStrategy):
    strategies: list[RestitutionStrategy] = field(
        default_factory=lambda: [CollimationStrategy(), PolyStrategy(), FlatStrategy()]
    )
    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)

    def __post_init__(self) -> None:
        super().__init__()
        self.__selected_strategy_: RestitutionStrategy | None = None

        # set the same vertical detector for each strategy to avoid re compute it
        for strat in self.strategies:
            if hasattr(strat, "vertical_detector"):
                setattr(strat, "vertical_detector", self.vertical_detector)

    @property
    def is_failed(self):
        if not self.is_fitted:
            raise RuntimeError("call fit() before")
        return self.__selected_strategy_ is None

    @property
    def selected_strategy_(self) -> RestitutionStrategy:
        if not self.is_fitted:
            raise RuntimeError("call fit() before")

        if self.__selected_strategy_ is None:
            raise RuntimeError("All strategies failed")

        return self.__selected_strategy_

    @property
    def failed_strategies(self) -> list[RestitutionStrategy]:
        if not self.is_fitted:
            raise RuntimeError("call fit() before")

        if self.__selected_strategy_ is None:
            return self.strategies

        idx = self.strategies.index(self.__selected_strategy_)
        return self.strategies[:idx]

    def get_transformation(self, output_width=None, output_height=22064):
        return self.selected_strategy_.get_transformation(output_width, output_height)

    def transform(self, output_path):
        self.selected_strategy_.transform(output_path)

    def _fit(self, raster_filepath):
        self.__selected_strategy_ = None

        # fit the vertical detector if is not already fitted.
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        # loop around all strategies fit them until a strategy don't failed.
        for strat in self.strategies:
            try:
                strat.fit(raster_filepath)
            except Exception:
                continue
            if not strat.is_failed:
                self.__selected_strategy_ = strat
                break

        return self
