from . import image_mosaic
from . import quality_control as qc
from .batch import join_images

from .restitution import (
    CollimationRectificationStrategy,
    FlatRectificationStrategy,
    ImageTransformer,
    ImageTransformerAffine,
    ImageTransformerTps,
    PolyRectificationStrategy,
    RectificationStrategy,
)

__all__ = [
    "image_mosaic",
    "ImageTransformer",
    "ImageTransformerTps",
    "ImageTransformerAffine",
    "RectificationStrategy",
    "FlatRectificationStrategy",
    "PolyRectificationStrategy",
    "CollimationRectificationStrategy",
    # legacy — still used by core.py
    "join_images",
    "qc",
]
