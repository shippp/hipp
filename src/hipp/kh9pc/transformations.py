from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from skimage.transform import AffineTransform, ThinPlateSplineTransform

from hipp.image import remap_tif_blockwise
from hipp.kh9pc.detection.base import DetectionError, JsonModel
from hipp.kh9pc.detection.detect_collimation_lines import CollimationDetectionResult
from hipp.kh9pc.detection.detect_fiducials import FiducialDetectionResult, FiducialSide
from hipp.kh9pc.detection.detect_flat_horizontal_edges import FlatDetectionResult
from hipp.kh9pc.detection.detect_poly_horizontal_edges import PolyDetectionResult
from hipp.kh9pc.fiducial_patterns import DetectedPattern, Patterns, compute_global_src_and_dst_points
from hipp.kh9pc.kh9_image_spec import COLLIMATION_LINE_DIST_PX, KH9ImageSpec
from hipp.tools import atomic_output, skip_existing


class Transformation(JsonModel):
    """Inverse warp from output pixel space to source raster space, defined by control points."""

    raster_filepath: Path
    kind: Literal["identity", "affine", "tps"]
    src_points: list[tuple[float, float]] = Field(default_factory=list)
    dst_points: list[tuple[float, float]] = Field(default_factory=list)
    crop_offset: tuple[int, int] = (0, 0)
    output_size: tuple[int, int] = (0, 0)

    @cached_property
    def deformation(self) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Output -> source mapping estimated from dst -> src control points."""
        if self.kind == "identity":
            return lambda coords: coords
        model = AffineTransform if self.kind == "affine" else ThinPlateSplineTransform
        deformation = model().from_estimate(np.array(self.dst_points), np.array(self.src_points))
        # from_estimate returns a falsy failure object instead of raising
        if not deformation:
            raise RuntimeError(f"{self.kind} transformation estimation failed for {self.raster_filepath}")
        return deformation

    def inverse_remap(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
        """Translate coords by ``crop_offset`` then apply ``deformation``."""
        coords = coords + np.array([self.crop_offset[0], self.crop_offset[1]], dtype=coords.dtype)
        return self.deformation(coords)


def compute_flat_transformation(
    result: str | Path | FlatDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Crop the detected frame and center it in the expected output size, without warping.

    Written to *output_file* as JSON, skipped if it already exists and *overwrite* is False.
    """
    if skip_existing(output_file, overwrite):
        return
    result = FlatDetectionResult.load(result)
    output_width, output_height = KH9ImageSpec.from_raster_filepath(result.raster_filepath).expected_size
    pad_x = (output_width - (result.right - result.left)) / 2
    pad_y = (output_height - result.height) / 2

    Transformation(
        raster_filepath=result.raster_filepath,
        kind="identity",
        crop_offset=(int(result.left - pad_x), int(result.top.position - pad_y)),
        output_size=(output_width, output_height),
    ).save(output_file)


def compute_poly_transformation(
    result: str | Path | PolyDetectionResult, output_file: str | Path, n_points: int = 10, overwrite: bool = False
) -> None:
    """Straighten the fitted top and bottom edges onto their median rows with an affine warp, then center.

    Written to *output_file* as JSON, skipped if it already exists and *overwrite* is False.
    """
    if skip_existing(output_file, overwrite):
        return
    _straighten_edges(PolyDetectionResult.load(result), n_points, "affine").save(output_file)


def compute_collimation_transformation(
    result: str | Path | CollimationDetectionResult,
    output_file: str | Path,
    n_points: int = 100,
    overwrite: bool = False,
) -> None:
    """Straighten the collimation lines with a TPS warp, the bottom one ``COLLIMATION_LINE_DIST_PX`` below the top.

    Written to *output_file* as JSON, skipped if it already exists and *overwrite* is False.
    """
    if skip_existing(output_file, overwrite):
        return
    collimation = CollimationDetectionResult.load(result)
    _straighten_edges(collimation, n_points, "tps", COLLIMATION_LINE_DIST_PX).save(output_file)


def compute_fiducial_transformation(
    result: str | Path | FiducialDetectionResult, output_file: str | Path, overwrite: bool = False
) -> None:
    """Map the primary fiducial patterns onto their theoretical grid with a TPS warp, then center.

    A side whose primary pattern scores below ``min_score_threshold`` is synthesised from the other side.
    Written to *output_file* as JSON, skipped if it already exists and *overwrite* is False.
    """
    if skip_existing(output_file, overwrite):
        return
    result = FiducialDetectionResult.load(result)
    if result.failure_reason:
        raise DetectionError(f"fiducial detection failed: {result.failure_reason}")
    spec = KH9ImageSpec.from_raster_filepath(result.raster_filepath)
    output_width, output_height = spec.expected_size

    def primary_pattern(side: FiducialSide, name: Patterns) -> DetectedPattern | None:
        """Primary pattern of *side* as a DetectedPattern, or None if its score is too low."""
        pattern = side.patterns[name]
        if pattern.score < result.options.min_score_threshold:
            return None
        return DetectedPattern(name, np.array(pattern.points), output_width, pattern.score)

    top_name, bottom_name = result.primary_pattern_names
    src, dst = compute_global_src_and_dst_points(
        primary_pattern(result.top, top_name), primary_pattern(result.bottom, bottom_name)
    )

    # map the vertical edges into destination space so the crop is centred on the warped image
    y_center = (dst[:, 1].min() + dst[:, 1].max()) / 2
    forward_tps = ThinPlateSplineTransform().from_estimate(src, dst)
    edges_dst = forward_tps(np.array([[result.left, y_center], [result.right, y_center]]))
    x_center = (edges_dst[0, 0] + edges_dst[1, 0]) / 2

    Transformation(
        raster_filepath=result.raster_filepath,
        kind="tps",
        src_points=src.tolist(),
        dst_points=dst.tolist(),
        crop_offset=(int(x_center - output_width / 2), int(y_center - output_height / 2)),
        output_size=(output_width, output_height),
    ).save(output_file)


def _straighten_edges(
    result: PolyDetectionResult | CollimationDetectionResult,
    n_points: int,
    kind: Literal["affine", "tps"],
    line_dist: int | None = None,
) -> Transformation:
    """Map the top and bottom curves onto horizontal rows, then center in the expected output size.

    The top row is the median of the top curve; the bottom row is *line_dist* below it, or the median of the bottom
    curve when *line_dist* is None.
    """
    output_width, output_height = KH9ImageSpec.from_raster_filepath(result.raster_filepath).expected_size

    x = np.linspace(result.left, result.right, n_points)
    y_top, y_bottom = result.top.predict(x), result.bottom.predict(x)
    top = int(np.median(y_top))
    bottom = top + line_dist if line_dist is not None else int(np.median(y_bottom))

    xx = np.concatenate((x, x))
    src = np.column_stack((xx, np.concatenate((y_top, y_bottom))))
    dst = np.column_stack((xx, np.concatenate((np.full_like(x, top), np.full_like(x, bottom)))))

    pad_x = (output_width - (result.right - result.left)) / 2
    pad_y = (output_height - (bottom - top)) / 2

    return Transformation(
        raster_filepath=result.raster_filepath,
        kind=kind,
        src_points=src.tolist(),
        dst_points=dst.tolist(),
        crop_offset=(int(result.left - pad_x), int(top - pad_y)),
        output_size=(output_width, output_height),
    )


def apply_transformation(
    transformation: str | Path | Transformation, output_path: str | Path, crop: bool = True, overwrite: bool = False
) -> None:
    """Warp the source raster with *transformation* and write it to *output_path*.

    With *crop*, the output is cropped to ``output_size`` at ``crop_offset``; otherwise it keeps the source raster size.
    Skipped if *output_path* already exists and *overwrite* is False.
    """
    if skip_existing(output_path, overwrite):
        return
    tf = Transformation.load(transformation)
    is_tps = tf.kind == "tps"
    # the warp takes minutes: an interrupted run must not leave a partial raster that later runs would skip
    with atomic_output(output_path) as tmp:
        remap_tif_blockwise(
            tf.raster_filepath,
            tmp,
            tf.inverse_remap if crop else tf.deformation,
            tf.output_size if crop else None,
            # TPS is costly to evaluate per pixel: evaluate it on a coarse grid over large blocks and interpolate
            block_size=2**13 if is_tps else 256,
            lowres_step=100 if is_tps else None,
        )
