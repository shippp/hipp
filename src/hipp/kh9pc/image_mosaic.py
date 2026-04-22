"""
Copyright (c) 2025 HIPP developers
Description: Functions to recreate in python the image_mosaic function from ASP
"""

from dataclasses import dataclass
from glob import glob
import logging
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

logger = logging.getLogger(__name__)


@dataclass
class ImageAlignment:
    """Alignment result for a single image in a sequential alignment chain.

    Attributes
    ----------
    image_path : Path
        Path to the image file.
    relative_transform : np.ndarray
        3x3 homogeneous transformation matrix relative to the previous image
        (identity for the first/reference image).
    absolute_transform : np.ndarray
        3x3 homogeneous transformation matrix in the global/mosaic coordinate system,
        accumulated from the reference image.
    n_matches : int
        Total number of ORB keypoint matches found before RANSAC filtering
        (0 for the reference image).
    n_inliers : int
        Number of inlier matches kept after RANSAC filtering
        (0 for the reference image).
    """

    image_path: Path
    relative_transform: np.ndarray
    absolute_transform: np.ndarray
    n_matches: int
    n_inliers: int


####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################


def compute_sequential_alignments(
    image_paths: list[str],
    overlap_width: int = 3000,
    bloc_height: int = 256,
    nfeature_per_block: int = 500,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 3.0,
) -> list[ImageAlignment]:
    """Compute sequential alignments between images.

    Detects ORB keypoints between consecutive images, estimates RANSAC Euclidean
    transforms, and accumulates absolute transformations from the reference image.

    Parameters
    ----------
    image_paths : list[str]
        Ordered list of image file paths to align. The first image is the reference
        (identity transform). Each subsequent image is aligned to the previous one.
    overlap_width : int, default 3000
        Width in pixels of the overlapping region used for keypoint matching.
    bloc_height : int, default 256
        Height of blocks (in pixels) used for local keypoint detection.
    nfeature_per_block : int, default 500
        Number of ORB features to detect per block.
    ransac_max_trials : int, default 1000
        Maximum number of RANSAC iterations for robust transform estimation.
    ransac_residual_threshold : float, default 3.0
        Maximum inlier residual for RANSAC.

    Returns
    -------
    list[ImageAlignment]
        One entry per input image, holding relative and absolute 3x3 homogeneous
        transformation matrices plus match statistics.
    """
    identity = np.eye(3)
    alignments: list[ImageAlignment] = [
        ImageAlignment(
            image_path=Path(image_paths[0]),
            relative_transform=identity,
            absolute_transform=identity,
            n_matches=0,
            n_inliers=0,
        )
    ]

    for i in range(len(image_paths) - 1):
        logger.info("Matching '%s' with '%s'", image_paths[i], image_paths[i + 1])

        points_a, points_b = _extract_global_matches_from_overlap(
            image_paths[i],
            image_paths[i + 1],
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
                image_path=Path(image_paths[i + 1]),
                relative_transform=relative_transform,
                absolute_transform=absolute_transform,
                n_matches=len(points_a),
                n_inliers=n_inliers,
            )
        )

    return alignments


def write_mosaic(
    alignments: list[ImageAlignment],
    output_tif: str,
    resampling: int = rasterio.warp.Resampling.cubic,
) -> None:
    """Warp and merge all aligned images into a single output GeoTIFF.

    Images are warped into the output pixel space using WarpedVRT and merged
    block-by-block. Valid pixels from later images do not overwrite valid pixels
    already written from earlier images.

    If any image extends above or to the left of the first image (negative coordinates
    after transformation), an offset is automatically applied to all transforms so that
    the full mosaic fits within the canvas without clipping.

    Parameters
    ----------
    alignments : list[ImageAlignment]
        Alignments as returned by :func:`compute_sequential_alignments`.
    output_tif : str
        Path to the output GeoTIFF file.
    resampling : int, default rasterio.warp.Resampling.cubic
        Resampling method from ``rasterio.warp.Resampling``.
    """
    output_width, output_height, offset_x, offset_y = _compute_canvas(alignments)
    n_images = len(alignments)

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

    os.makedirs(os.path.dirname(output_tif) or ".", exist_ok=True)
    n_blocks = (output_width // 256 + 1) * (output_height // 256 + 1)

    logger.info("Mosaicing %d images → %s (%d×%d px)", n_images, output_tif, output_width, output_height)

    with rasterio.open(output_tif, "w+", **profile) as dst:
        for i, alignment in enumerate(alignments):
            logger.info("[%d/%d] %s", i + 1, n_images, alignment.image_path.name)

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
                    log_every = max(1, n_blocks // 50)
                    for block_idx, (_, window) in enumerate(dst.block_windows(1)):
                        warped = vrt.read(1, window=window)
                        mask = warped != 0
                        if not mask.any():
                            continue
                        existing = dst.read(1, window=window)
                        dst.write(np.where(mask, warped, existing), 1, window=window)
                        if block_idx % log_every == 0:
                            logger.debug("  warping block %d/%d", block_idx, n_blocks)

    logger.info("Mosaic written to %s", output_tif)


####################################################################################################################################
#                                                   STANDALONE FUNCTIONS
####################################################################################################################################


def image_mosaic_asp(
    image_paths: list[str | Path],
    output_image_path: str | Path,
    threads: int = 0,
    cleanup: bool = True,
    verbose: bool = True,
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
    verbose : bool, optional
        If True, prints detailed progress and command information. Default is True.
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

    if verbose:
        print(" ".join(cmd))

    if not dryrun:
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {output_image_path}: {e}")

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
    image_a_path: str,
    image_b_path: str,
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

        assert height_a == height_b, "Both images must have the same height for block-wise matching."

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
