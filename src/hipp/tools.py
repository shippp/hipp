"""
Module: tools.py
Author: godinlu
Date: 28
Description: Generic tools
"""

from typing import Any

import cv2

from hipp.image import apply_clahe


def points_picker(
    image: cv2.typing.MatLike, point_count: int = 1, clahe_enhancement: bool = True
) -> list[tuple[int, int]]:
    """Pick points interactively on a image, only when Ctrl is pressed.

    Args:
        image (np.ndarray): The original large image.
        point_count (int, optional): Number of points to pick. Defaults to 1.
        clahe_enhancement (bool, optional): Apply clahe enhancement. Defaults to True.

    Returns:
        list[tuple[int, int]]: Points in coordinates of the original image.
    """
    picked_points: list[tuple[int, int]] = []
    clone = image.copy()

    if clahe_enhancement:
        clone = apply_clahe(clone)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal picked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if Ctrl is held down (flags & cv2.EVENT_FLAG_CTRLKEY)
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if len(picked_points) < point_count:
                    # Scale back coordinates to the original image size
                    picked_points.append((x, y))

                    # Draw a small circle on the resized image
                    if len(clone.shape) == 2:  # Grayscale image (single channel)
                        cv2.circle(clone, (x, y), 5, (255,), -1)  # Use white circle
                    else:  # RGB image (three channels)
                        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)  # Use green circle

                    # Update the window title
                    title = f"Pick Points with 'Ctrl + Click' ({len(picked_points)}/{point_count} points picked)"
                    cv2.setWindowTitle("Pick Points", title)

                    cv2.imshow("Pick Points", clone)

    cv2.namedWindow("Pick Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Pick Points", clone)
    cv2.setMouseCallback("Pick Points", mouse_callback)

    # Initial window title
    title = f"Pick Points with 'Ctrl + Click' (0/{point_count} points picked)"
    cv2.setWindowTitle("Pick Points", title)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(picked_points) >= point_count:
            break
        if key == ord("q"):
            break
        if cv2.getWindowProperty("Pick Points", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return picked_points
