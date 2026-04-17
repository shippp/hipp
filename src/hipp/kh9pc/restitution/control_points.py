import numpy as np
from numpy.typing import NDArray

from hipp.kh9pc.restitution.detectors import CollimationDetector, FlatDetector, PolyDetector


def poly_control_points(
    detector: PolyDetector,
    grid_shape: tuple[int, int] = (100, 50),
) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
    """Compute a dense control-point grid from polynomial edge fits.

    Parameters
    ----------
    detector : PolyDetector
        A fitted detector providing top/bottom polynomial edge models.
    grid_shape : tuple[int, int]
        ``(n_cols, n_rows)`` of the control-point grid. Default ``(100, 50)``.

    Returns
    -------
    src_points : NDArray, shape (N, 2)
        Control points in the distorted source image.
    dst_points : NDArray, shape (N, 2)
        Corresponding control points in the rectified destination image,
        normalised to ``[0, width] × [0, height]``.
    detected_size : tuple[int, int]
        ``(width, height)`` of the detected content region.
    """
    left, right = detector.vertical_edges
    detected_width = right - left

    x_src = np.linspace(left, right, grid_shape[0])
    y_top_src = detector.top.poly.predict(x_src.reshape(-1, 1)).ravel()
    y_bottom_src = detector.bottom.poly.predict(x_src.reshape(-1, 1)).ravel()
    detected_height = int(np.abs(np.mean(y_bottom_src - y_top_src)))

    x_dst = np.linspace(0, detected_width, grid_shape[0])

    src_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)
    dst_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)
    for i, (xi_src, xi_dst, yt, yb) in enumerate(zip(x_src, x_dst, y_top_src, y_bottom_src)):
        src_points[i, :, 0] = xi_src
        src_points[i, :, 1] = np.linspace(yt, yb, grid_shape[1])
        dst_points[i, :, 0] = xi_dst
        dst_points[i, :, 1] = np.linspace(0, detected_height, grid_shape[1])

    return src_points.reshape(-1, 2), dst_points.reshape(-1, 2), (detected_width, detected_height)


def collimation_control_points(
    detector: CollimationDetector,
    collimation_line_dist: int = 21770,
    grid_shape: tuple[int, int] = (100, 50),
) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
    """Compute a dense control-point grid from collimation-line fits.

    The detected height is fixed to ``collimation_line_dist`` rather than
    inferred from the polynomial fit, since this distance is a physically
    calibrated constant of the KH-9 camera.

    Parameters
    ----------
    detector : CollimationDetector
        A fitted detector providing top/bottom collimation line models.
    collimation_line_dist : int
        Known distance (px) between the two collimation lines. Default 21770.
    grid_shape : tuple[int, int]
        ``(n_cols, n_rows)`` of the control-point grid. Default ``(100, 50)``.

    Returns
    -------
    src_points : NDArray, shape (N, 2)
    dst_points : NDArray, shape (N, 2)
    detected_size : tuple[int, int]
    """
    left, right = detector.vertical_edges
    detected_width = right - left
    detected_height = collimation_line_dist

    x_src = np.linspace(left, right, grid_shape[0])
    y_top_src = detector.top.model.predict(x_src.reshape(-1, 1)).ravel()
    y_bottom_src = detector.bottom.model.predict(x_src.reshape(-1, 1)).ravel()
    x_dst = np.linspace(0, detected_width, grid_shape[0])

    src_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)
    dst_points = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=float)
    for i, (xi_src, xi_dst, yt, yb) in enumerate(zip(x_src, x_dst, y_top_src, y_bottom_src)):
        src_points[i, :, 0] = xi_src
        src_points[i, :, 1] = np.linspace(yt, yb, grid_shape[1])
        dst_points[i, :, 0] = xi_dst
        dst_points[i, :, 1] = np.linspace(0, detected_height, grid_shape[1])

    return src_points.reshape(-1, 2), dst_points.reshape(-1, 2), (detected_width, detected_height)


def flat_control_points(
    detector: FlatDetector,
) -> tuple[NDArray[np.floating], NDArray[np.floating], tuple[int, int]]:
    """Compute a 4-corner control-point grid from flat horizontal edge detection.

    Since the edges are assumed to be straight horizontal lines, only the
    four corners are needed to fully define the affine rectification.

    Parameters
    ----------
    detector : FlatDetector
        A fitted detector providing top/bottom edge positions.

    Returns
    -------
    src_points : NDArray, shape (4, 2)
    dst_points : NDArray, shape (4, 2)
    detected_size : tuple[int, int]
    """
    left, right = detector.vertical_edges
    detected_width = right - left
    detected_height = detector.bottom.position - detector.top.position

    src_points = np.array(
        [
            [left, detector.top.position],
            [left, detector.bottom.position],
            [right, detector.top.position],
            [right, detector.bottom.position],
        ],
        dtype=float,
    )
    dst_points = np.array(
        [
            [0, 0],
            [0, detected_height],
            [detected_width, 0],
            [detected_width, detected_height],
        ],
        dtype=float,
    )
    return src_points, dst_points, (detected_width, detected_height)
