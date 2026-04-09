from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class OutputSize(ABC):
    """Specification for placing detected content onto an output canvas.

    Subclasses define how the output raster dimensions and the position of the
    rectified content within it are determined. The public interface is
    :meth:`apply`, which takes the raw grids produced by
    :meth:`~hipp.kh9pc.restitution.base.RectificationStrategy.compute_grid` and
    returns shifted grids together with the final raster size.

    Subclasses must implement :meth:`resolve`, which maps the detected content
    dimensions to ``(out_width, out_height, x_offset, y_offset)``.
    """

    @abstractmethod
    def resolve(self, detected_width: int, detected_height: int) -> tuple[int, int, int, int]:
        """Return ``(out_width, out_height, x_offset, y_offset)``.

        Parameters
        ----------
        detected_width : int
            Width of the content region as returned by ``compute_grid()``.
        detected_height : int
            Height of the content region as returned by ``compute_grid()``.
        """
        ...

    def apply(
        self,
        src_points: NDArray[np.floating],
        dst_points: NDArray[np.floating],
        detected_size: tuple[int, int],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
        """Place the grid onto the output canvas.

        Parameters
        ----------
        src_points : NDArray, shape (..., 2)
            Source control points returned by ``compute_grid()``. Returned unchanged.
        dst_points : NDArray, shape (..., 2)
            Destination control points returned by ``compute_grid()``. Shifted by
            the canvas offset before being returned.
        detected_size : tuple[int, int]
            ``(width, height)`` of the detected content region, as returned by
            ``compute_grid()``.

        Returns
        -------
        src_points : NDArray
            Unchanged source control points.
        dst_points : NDArray
            Destination control points shifted to the correct canvas position.
        output_size : tuple[int, int]
            Final ``(width, height)`` of the output raster.
        """
        detected_width, detected_height = detected_size
        out_w, out_h, x_off, y_off = self.resolve(detected_width, detected_height)
        dst_points = dst_points.copy()
        dst_points[..., 0] += x_off
        dst_points[..., 1] += y_off
        return src_points, dst_points, (out_w, out_h)


@dataclass(frozen=True)
class AutoSize(OutputSize):
    """Output size equals the detected content exactly — no padding, no crop.

    This is the default behaviour: the output raster is sized to fit the content
    region detected by the strategy.
    """

    def resolve(self, detected_width: int, detected_height: int) -> tuple[int, int, int, int]:
        return detected_width, detected_height, 0, 0


@dataclass(frozen=True)
class SameSize(OutputSize):
    """Output has the same pixel dimensions as the original input raster.

    The detected content is centred inside the original canvas.

    Parameters
    ----------
    width : int
        Width of the original input raster in pixels.
    height : int
        Height of the original input raster in pixels.
    """

    width: int
    height: int

    def resolve(self, detected_width: int, detected_height: int) -> tuple[int, int, int, int]:
        x_offset = (self.width - detected_width) // 2
        y_offset = (self.height - detected_height) // 2
        return self.width, self.height, x_offset, y_offset


@dataclass(frozen=True)
class FixedSize(OutputSize):
    """Fixed output dimensions with the content centred inside.

    Parameters
    ----------
    width : int
        Desired output width in pixels.
    height : int
        Desired output height in pixels.
    """

    width: int
    height: int

    def resolve(self, detected_width: int, detected_height: int) -> tuple[int, int, int, int]:
        x_offset = (self.width - detected_width) // 2
        y_offset = (self.height - detected_height) // 2
        return self.width, self.height, x_offset, y_offset


@dataclass(frozen=True)
class FixedHeightSize(OutputSize):
    """Fixed output height with detected width kept as-is. Content is centred vertically.

    Useful when different rectification strategies produce variable detected heights
    but the output must always be a consistent height (e.g. for photogrammetric
    pipelines that expect a canonical image size).

    Parameters
    ----------
    height : int
        Desired output height in pixels.
    """

    height: int

    def resolve(self, detected_width: int, detected_height: int) -> tuple[int, int, int, int]:
        y_offset = (self.height - detected_height) // 2
        return detected_width, self.height, 0, y_offset


@dataclass(frozen=True)
class MarginSize(OutputSize):
    """Add independent margins (in pixels) around the detected content on each side.

    Parameters
    ----------
    top : int
        Pixels added above the content. Default 0.
    right : int
        Pixels added to the right of the content. Default 0.
    bottom : int
        Pixels added below the content. Default 0.
    left : int
        Pixels added to the left of the content. Default 0.

    Examples
    --------
    Uniform margin on all sides::

        MarginSize(top=100, right=100, bottom=100, left=100)

    Asymmetric margins::

        MarginSize(top=200, right=50, bottom=200, left=50)
    """

    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0

    def resolve(self, detected_width: int, detected_height: int) -> tuple[int, int, int, int]:
        out_w = self.left + detected_width + self.right
        out_h = self.top + detected_height + self.bottom
        return out_w, out_h, self.left, self.top
