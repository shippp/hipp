import shutil

from . import aerial, dataquery, image, kh9pc, tools
from .aerial.aerial_preprocessing import AerialPreprocessing

__all__ = ["aerial", "AerialPreprocessing", "dataquery", "tools", "kh9pc", "image"]


if shutil.which("stereo") is None:
    raise ImportError(
        "ASP toolkit not found. Please ensure the 'stereo' executable is installed and accessible in your system PATH. Refer to https://github.com/shippp/hipp?tab=readme-ov-file#installation for installation instructions."
    )
