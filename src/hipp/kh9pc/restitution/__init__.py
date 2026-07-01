"""
Copyright (c) 2026 HIPP developers
Description: Restitution strategies for KH-9 PC images — geometric correction from
    raw mosaicked scans to standardised output frames. Strategies are listed in order
    of increasing fallback: FiducialStrategy → CollimationStrategy → PolyStrategy → FlatStrategy.
    MixedStrategy chains them automatically and selects the first that succeeds.
"""

from .collimation_strategy import CollimationStrategy
from .fiducial_strategy import FiducialStrategy
from .flat_strategy import FlatStrategy
from .mixed_strategy import MixedStrategy
from .poly_strategy import PolyStrategy
from .vertical_detector import VerticalDetector

__all__ = [
    "CollimationStrategy",
    "FiducialStrategy",
    "FlatStrategy",
    "PolyStrategy",
    "MixedStrategy",
    "VerticalDetector",
]
