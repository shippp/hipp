import logging
from dataclasses import dataclass, field
from pathlib import Path

from hipp.kh9pc.restitution.collimation import CollimationStrategy
from hipp.kh9pc.restitution.fiducial import FiducialStrategy
from hipp.kh9pc.restitution.flat import FlatStrategy
from hipp.kh9pc.restitution.poly import PolyStrategy
from hipp.kh9pc.restitution.base import RestitutionStrategy, Transformation

logger = logging.getLogger(__name__)


@dataclass
class MixedStrategy(RestitutionStrategy):
    strategies: list[RestitutionStrategy] = field(
        default_factory=lambda: [FiducialStrategy(), CollimationStrategy(), PolyStrategy(), FlatStrategy()]
    )
    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)

    def __post_init__(self) -> None:
        super().__init__()
        self.__selected_strategy_: RestitutionStrategy | None = None

        for i, strat in enumerate(self.strategies):
            if hasattr(strat, "vertical_detector"):
                setattr(strat, "vertical_detector", self.poly_strategy.vertical_detector)
            if hasattr(strat, "poly_strategy"):
                setattr(strat, "poly_strategy", self.poly_strategy)
            # replace any standalone PolyStrategy with the shared instance to avoid recomputation
            if isinstance(strat, PolyStrategy) and strat is not self.poly_strategy:
                self.strategies[i] = self.poly_strategy

    @property
    def is_failed(self) -> bool:
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

    @property
    def transformation_(self) -> Transformation:
        return self.selected_strategy_.transformation_

    def transform(self, output_path: str | Path) -> None:
        self.selected_strategy_.transform(output_path)

    def _fit(self, raster_filepath: Path) -> "MixedStrategy":
        self.__selected_strategy_ = None

        vd = self.poly_strategy.vertical_detector
        if not vd.is_fitted or raster_filepath != vd.raster_filepath_:
            vd.fit(raster_filepath)

        for strat in self.strategies:
            try:
                if not strat.is_fitted or raster_filepath != strat.raster_filepath_:
                    strat.fit(raster_filepath)
            except Exception:
                logger.warning("%s failed for %s", type(strat).__name__, raster_filepath.name, exc_info=True)
                continue
            if not strat.is_failed:
                self.__selected_strategy_ = strat
                break

        return self
