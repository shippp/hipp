"""
Copyright (c) 2025 HIPP developers
Description: Functions to recreate in python the image_mosaic function from ASP
"""

import logging
import os
import subprocess
from collections.abc import Sequence
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

from hipp.kh9pc.types import ImageAlignment

logger = logging.getLogger(__name__)


####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################
def image_mosaic(
    image_paths: Sequence[str | Path],
    output_tif: str | Path,
    overwrite: bool = False,
    resampling: int = Resampling.cubic,
    overlap_width: int = 3000,
    bloc_height: int = 256,
    nfeature_per_block: int = 500,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 3.0,
) -> None:
    # standardize paths
    output_tif = Path(output_tif)

    # manage overwrite
    if output_tif.exists() and not overwrite:
        logger.info("Skipping image_mosaic: %s (already exists, overwrite=False)", str(output_tif))
        return

    alignments = compute_sequential_alignments(
        image_paths,
        overlap_width=overlap_width,
        bloc_height=bloc_height,
        nfeature_per_block=nfeature_per_block,
        ransac_max_trials=ransac_max_trials,
        ransac_residual_threshold=ransac_residual_threshold,
    )

    write_mosaic(alignments, output_tif, resampling=resampling)


def compute_sequential_alignments(
    image_paths: Sequence[str | Path],
    overlap_width: int = 3000,
    bloc_height: int = 256,
    nfeature_per_block: int = 500,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 3.0,
) -> list[ImageAlignment]:
    """Compute sequential alignments between images.

    Detects ORB keypoints between consecutive images, estimates RANSAC Euclidean
    transforms, and accumulates absolute transformations from the reference image.
    """
    # standardize path
    paths: list[Path] = [Path(f) for f in image_paths]

    identity = np.eye(3)
    alignments: list[ImageAlignment] = [
        ImageAlignment(
            image_path=paths[0],
            relative_transform=identity,
            absolute_transform=identity,
            n_matches=0,
            n_inliers=0,
        )
    ]

    for i in range(len(paths) - 1):
        logger.info("Matching '%s' with '%s'", str(paths[i]), str(paths[i + 1]))

        points_a, points_b = _extract_global_matches_from_overlap(
            paths[i],
            paths[i + 1],
            overlap_width,
            bloc_height,
            nfeature_per_block,
        )

        model_robust, inliers = ransac(
            (np.array(points_b, dtype=np.float32), np.array(points_a, dtype=np.float32)),
            EuclideanTransform,
            min_samples=3,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
        )

        n_inliers = int(np.sum(inliers))
        logger.info("Inliers after RANSAC: %d/%d", n_inliers, len(points_a))

        relative_transform: np.ndarray = model_robust.params
        absolute_transform: np.ndarray = alignments[i].absolute_transform @ relative_transform

        alignments.append(
            ImageAlignment(
                image_path=Path(paths[i + 1]),
                relative_transform=relative_transform,
                absolute_transform=absolute_transform,
                n_matches=len(points_a),
                n_inliers=n_inliers,
            )
        )

    return alignments


def write_mosaic(
    alignments: list[ImageAlignment],
    output_tif: str | Path,
    resampling: int = Resampling.cubic,
) -> None:
    """Warp and merge all aligned images into a single output GeoTIFF.

    Images are warped into the output pixel space using WarpedVRT and merged
    block-by-block. Valid pixels from later images do not overwrite valid pixels
    already written from earlier images.

    If any image extends above or to the left of the first image (negative coordinates
    after transformation), an offset is automatically applied to all transforms so that
    the full mosaic fits within the canvas without clipping.

    """
    # normalize path
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(exist_ok=True, parents=True)

    output_width, output_height, offset_x, offset_y = _compute_canvas(alignments)

    T_offset = np.array([[1, 0, -offset_x], [0, 1, -offset_y], [0, 0, 1]], dtype=float)

    fake_crs = rasterio.CRS.from_epsg(3857)
    dst_transform = rasterio.Affine.identity()

    profile = {
        "width": output_width,
        "height": output_height,
        "compress": "lzw",
        "driver": "GTiff",
        "BIGTIFF": "YES",
        "count": 1,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "dtype": "uint8",
    }

    n_blocks = (output_width // 256 + 1) * (output_height // 256 + 1)
    log_every = max(1, n_blocks // 5)

    logger.info("Mosaicing %d images → %s (%d×%d px)", len(alignments), str(output_tif), output_width, output_height)

    with rasterio.open(output_tif, "w+", **profile) as dst:
        for i, alignment in enumerate(alignments):
            logger.info("[%d/%d] %s", i + 1, len(alignments), alignment.image_path.name)

            adjusted_transform = T_offset @ alignment.absolute_transform

            with rasterio.open(alignment.image_path) as src:
                with WarpedVRT(
                    src,
                    src_transform=rasterio.Affine(*adjusted_transform.flatten()[:6]),
                    src_crs=fake_crs,
                    dst_crs=fake_crs,
                    resampling=resampling,
                    width=output_width,
                    height=output_height,
                    transform=dst_transform,
                ) as vrt:
                    for block_idx, (_, window) in enumerate(dst.block_windows(1)):
                        if block_idx % log_every == 0:
                            logger.debug("  %d%%", block_idx * 100 // n_blocks)
                        warped = vrt.read(1, window=window)
                        mask = warped != 0
                        if not mask.any():
                            continue
                        existing = dst.read(1, window=window)
                        dst.write(np.where(mask, warped, existing), 1, window=window)

    logger.info("Mosaic written to %s", str(output_tif))


####################################################################################################################################
#                                                   STANDALONE FUNCTIONS
####################################################################################################################################


def image_mosaic_asp(
    image_paths: list[str | Path],
    output_image_path: str | Path,
    threads: int = 0,
    cleanup: bool = True,
    dryrun: bool = False,
) -> None:
    """
    Mosaics a list of images into a single output image using the external 'image_mosaic' command.

    Parameters
    ----------
    image_paths : list[str | Path]
        List of paths to input image tiles.
    output_image_path : str | Path
        Path to the output mosaic image.
    threads : int, optional
        Number of threads to use for processing. Default is 0 (let the tool decide).
    cleanup : bool, optional
        Whether to remove temporary log and auxiliary files after processing. Default is True.
    dryrun : bool, optional
        If True, builds the command but does not execute it. Default is False.
    """
    str_image_paths = list(sorted([str(f) for f in image_paths]))

    cmd = [
        "image_mosaic",
        *str_image_paths,
        "--ot",
        "byte",
        "--overlap-width",
        "3000",
        "--threads",
        str(threads),
        "-o",
        str(output_image_path),
    ]

    logger.info("Running: %s", " ".join(cmd))

    if not dryrun:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("image_mosaic_asp failed for %s: %s", output_image_path, e)

    if cleanup:
        for f in glob(f"{output_image_path}-log-image_mosaic-*.txt") + glob(f"{output_image_path}.aux.xml"):
            os.remove(f)


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################


def _compute_canvas(alignments: list[ImageAlignment]) -> tuple[int, int, float, float]:
    """Compute output canvas dimensions and the offset needed to shift all images into positive coordinates.

    Returns
    -------
    width : int
    height : int
    offset_x : float
        Horizontal shift to apply so the leftmost pixel lands at x=0.
    offset_y : float
        Vertical shift to apply so the topmost pixel lands at y=0.
    """
    all_corners: list[np.ndarray] = []
    for alignment in alignments:
        with rasterio.open(alignment.image_path) as src:
            w, h = src.width, src.height
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float).T
        transformed = (alignment.absolute_transform @ corners)[:2]
        all_corners.append(transformed)

    stacked = np.hstack(all_corners)
    min_x, min_y = stacked[0].min(), stacked[1].min()
    width = int(np.ceil(stacked[0].max() - min_x))
    height = int(np.ceil(stacked[1].max() - min_y))
    return width, height, min_x, min_y


def _extract_global_matches_from_overlap(
    image_a_path: str | Path,
    image_b_path: str | Path,
    overlap_width: int = 3000,
    bloc_height: int = 1024,
    nfeature_per_block: int = 500,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Extract matched keypoints between the overlapping edge of two images, in horizontal blocks.

    Assumes image A is on the left and image B is on the right.
    """
    points_a, points_b = [], []

    with rasterio.open(image_a_path) as src_a, rasterio.open(image_b_path) as src_b:
        width_a = src_a.width
        height_a = src_a.height
        height_b = src_b.height

        if height_a != height_b:
            raise ValueError(f"Both images must have the same height for block-wise matching ({height_a} != {height_b}).")

        for i in range(0, height_a, bloc_height):
            current_block_height = min(bloc_height, height_a - i)

            window_a = Window(
                col_off=width_a - overlap_width, row_off=i, width=overlap_width, height=current_block_height
            )
            window_b = Window(col_off=0, row_off=i, width=overlap_width, height=current_block_height)

            block_a = src_a.read(1, window=window_a)
            block_b = src_b.read(1, window=window_b)

            pts_a, pts_b = _match_orb_keypoints(block_a, block_b, nfeatures=nfeature_per_block)

            points_a.extend([(pt[0] + (width_a - overlap_width), pt[1] + i) for pt in pts_a])
            points_b.extend([(pt[0], pt[1] + i) for pt in pts_b])

    return points_a, points_b


def _match_orb_keypoints(
    image_a: cv2.typing.MatLike, image_b: cv2.typing.MatLike, nfeatures: int = 500
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Detect ORB keypoints and return matched coordinates between two grayscale images.

    Returns
    -------
    pts_a : list of tuple[float, float]
        Matched keypoint coordinates from image A.
    pts_b : list of tuple[float, float]
        Matched keypoint coordinates from image B.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)  # type: ignore[attr-defined]

    kp_a, des_a = orb.detectAndCompute(image_a, None)
    kp_b, des_b = orb.detectAndCompute(image_b, None)

    if des_a is None or des_b is None:
        return [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_a, des_b), key=lambda x: x.distance)

    pts_a = [kp_a[m.queryIdx].pt for m in matches]
    pts_b = [kp_b[m.trainIdx].pt for m in matches]

    return pts_a, pts_b
