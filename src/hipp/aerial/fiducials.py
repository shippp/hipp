import csv
import os
from typing import Callable, Generic, TypeVar

import cv2
import numpy as np

import hipp.math

T = TypeVar("T")
U = TypeVar("U")


class Fiducials(dict[str, T], Generic[T]):
    """
    A specialized dictionary for handling fiducial markers, such as corners and midsides,
    commonly used in image processing or geometric referencing tasks.

    This class extends the built-in `dict` type and provides utility methods for managing
    and transforming specific fiducial points identified by conventional keys.

    The expected keys for fiducials are defined as:
        - Corners: ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
        - Midsides: ["midside_left", "midside_top", "midside_right", "midside_bottom"]
    """

    corners_label = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
    midsides_label = ["midside_left", "midside_top", "midside_right", "midside_bottom"]

    def get_corners(self) -> dict[str, T]:
        """Return a dictionary containing only the corner fiducials, in circular order."""
        return {key: self[key] for key in Fiducials.corners_label}

    def get_midsides(self) -> dict[str, T]:
        """Return a dictionary containing only the midside fiducials, in circular order"""
        return {key: self[key] for key in Fiducials.midsides_label}

    def get_fiducials(self) -> dict[str, T]:
        """Return a dictionary containing the subset of fiducials present in the instance,
        including corners and/or midsides, depending on availability."""
        result = {}
        if self.as_corners():
            result.update(self.get_corners())
        if self.as_midsides():
            result.update(self.get_midsides())
        return result

    def as_corners(self) -> bool:
        """Check whether the instance contains all four corner keys."""
        return all(key in self for key in self.corners_label)

    def as_midsides(self) -> bool:
        """Check whether the instance contains all four midside keys"""
        return all(key in self for key in self.midsides_label)

    def apply(self, func: Callable[[T], U]) -> "Fiducials[U]":
        """Apply a function to each value and return a new Fiducials object with updated values."""
        result = Fiducials[U]()
        for key, value in self.items():
            result[key] = func(value)
        return result

    def apply_with_key(self, func: Callable[[str, T], U]) -> "Fiducials[U]":
        """Apply a function that takes (key, value) to each item."""
        result = Fiducials[U]()
        for key, value in self.items():
            result[key] = func(key, value)
        return result


class FiducialsCoordinate(Fiducials[tuple[float, float] | None]):
    """
    Specialized container for storing and manipulating 2D fiducial marker coordinates.

    This class extends `Fiducials`, where each value is either a 2D point `(x, y)` or `None`
    if the fiducial was not detected or is invalid. It provides methods for estimating geometric
    properties such as the principal point, validating angular consistency, transforming coordinates,
    and converting to a camera reference frame.

    The fiducials are typically expected to follow a rectangular layout with the following keys:
    - Corners: ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
    - Midsides: ["midside_left", "midside_top", "midside_right", "midside_bottom"]

    This abstraction is designed to facilitate spatial calibration and transformation workflows
    in imaging pipelines, including those involving scanners, drones, or cameras in remote sensing.

    Additional Functionalities
    --------------------------
    - `compute_principal_point()`: Estimate the image principal point using the detected fiducials.
    - `validate_angles()`: Check if angles between fiducials are close to 90°, indicating a valid rectangular layout.
    - `estimate_transformation_matrix()`: Compute a transformation to align this set of fiducials with another.
    - `convert_in_camera_reference()`: Transform the coordinates into a camera-centric coordinate frame.
    - `transform()`: Apply an arbitrary 3x3 transformation matrix to all coordinates.

    Examples
    --------
    >>> fc = FiducialsCoordinate({
    ...     "corner_top_left": (0.0, 0.0),
    ...     "corner_top_right": (100.0, 0.0),
    ...     "corner_bottom_right": (100.0, 100.0),
    ...     "corner_bottom_left": (0.0, 100.0),
    ... })
    >>> principal = fc.compute_principal_point()
    >>> fc_cam = fc.convert_in_camera_reference(scanning_resolution_mm=0.02)

    Notes
    -----
    This class assumes a planar (2D) geometry of fiducial markers. The spatial reasoning and transformation
    logic rely on the assumption that the markers form a quadrilateral, typically aligned with the image borders.
    """

    def compute_principal_point(self) -> tuple[float, float] | None:
        """
        Estimates the principal point of an image based on detected fiducial markers.

        The principal point is computed using both diagonal midpoints and perpendicular offsets
        from adjacent fiducial segments. The algorithm uses two types of fiducial markers:
        - Corners: ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
        - Midsides: ["mid_left", "mid_top", "mid_right", "mid_bottom"]

        For each group (corners and midsides), the following logic is applied:
        1. For each fiducial and its diagonal counterpart (i and (i+2)%4), compute the midpoint if both exist.
        2. For each adjacent pair (i and (i+1)%4), compute the midpoint of the segment and create a point
        perpendicular to the segment direction, offset by half the segment length.

        All valid midpoints and orthogonal points are averaged to return the final principal point estimate.

        Args:
            detected_fiducials (dict): A dictionary mapping fiducial names to (x, y) coordinates or None.

        Returns:
            tuple[float, float] or None: The estimated principal point as an (x, y) tuple,
                                        or None if no valid points were available.
        """

        # Here we calculate the barycenter for the direction of the orthogonal vector
        valid_points = [np.array(coord) for coord in self.get_fiducials().values() if coord is not None]
        barycenter = np.mean(valid_points, axis=0)

        orthogonal_points = []
        midpoints = []
        for fiducial_names in [Fiducials.corners_label, Fiducials.midsides_label]:
            for i in range(4):
                # Get the points and check they are not None
                p_ortho_1 = self.get(fiducial_names[i])
                p_ortho_2 = self.get(fiducial_names[(i + 1) % 4])
                p_diag_1 = self.get(fiducial_names[i])
                p_diag_2 = self.get(fiducial_names[(i + 2) % 4])

                # Diagonal: compute the midpoint
                if p_diag_1 is not None and p_diag_2 is not None:
                    midpoint_diag = (np.array(p_diag_1) + np.array(p_diag_2)) / 2
                    midpoints.append(midpoint_diag)

                # Orthogonal: compute the orthogonal point at the center of the adjacent segment
                if p_ortho_1 is not None and p_ortho_2 is not None:
                    p1 = np.array(p_ortho_1)
                    p2 = np.array(p_ortho_2)
                    mid = (p1 + p2) / 2

                    # Direction vector of the segment
                    direction = p2 - p1
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        # Unit orthogonal vector
                        perp = np.array([-direction[1], direction[0]]) / norm

                        # Ensure the perpendicular points towards the barycenter
                        to_center = barycenter - mid
                        if np.dot(perp, to_center) < 0:
                            perp = -perp  # Flip direction

                        # Scale the orthogonal offset (here: half the length of the segment)
                        orth_point = mid + perp * (norm / 2)
                        orthogonal_points.append(orth_point)

        # Compute the average of all valid points (diagonals + orthogonals)
        all_points = midpoints + orthogonal_points
        if all_points:
            principal_point = np.mean(all_points, axis=0)
            self["principal_point"] = (float(principal_point[0]), float(principal_point[1]))
        else:
            self["principal_point"] = None
        return self["principal_point"]

    def validate_angles(self, degree_threshold: float = 0.005) -> Fiducials[bool]:
        """
        Evaluate which corner or midside points in a quadrilateral detection are geometrically valid
        based on angle closeness to 90 degrees.

        This function dynamically adapts to whether the input contains corners, midsides, or both.

        Args:
            degree_threshold (float): Allowed deviation from 90° to consider an angle valid.

        Returns:
            Fiducials[bool]: Dictionary mapping each evaluated point name to True (valid) or False (suspect).
        """
        result = Fiducials[bool]()

        # Detect whether to use corners, midsides, or both
        groups_to_check = []
        if self.as_corners():
            groups_to_check.append(Fiducials.corners_label)
        if self.as_midsides():
            groups_to_check.append(Fiducials.midsides_label)
        for group in groups_to_check:
            for i in range(4):
                point_names = [group[(i - 1) % 4], group[i], group[(i + 1) % 4]]
                points = [self[name] for name in point_names]
                angle = hipp.math.angle_between_three_points(*points)  # type: ignore[arg-type]

                for name in point_names:
                    if abs(90 - angle) < degree_threshold:
                        result[name] = True
                    elif name not in result:
                        result[name] = False
        return result

    def estimate_transformation_matrix(self, fiducials_coordinate: "FiducialsCoordinate") -> cv2.typing.MatLike:
        """
        Estimate a 2D transformation matrix (homogeneous) between two sets of fiducial coordinates.

        If only one fiducial point is available, the transformation reduces to a translation.
        If two or more points are available, a full similarity transformation is estimated
        (translation, rotation, and uniform scaling).

        Parameters:
            fiducials_coordinate (FiducialsCoordinate): The target fiducials used to estimate the transformation.

        Returns:
            cv2.typing.MatLike: A 3x3 transformation matrix in homogeneous coordinates.

        Raises:
            ValueError: If the fiducials sets do not share the same keys, or if no valid points are available.
        """
        # Retrieve source and destination fiducial dictionaries
        src_dict = self.get_fiducials()
        dst_dict = fiducials_coordinate.get_fiducials()

        # Ensure both dictionaries contain the same fiducial keys
        if not src_dict.keys() == dst_dict.keys():
            raise ValueError("source fiducials must contain the same keys as dest fiducials")

        # Filter keys where both source and destination have valid (non-None) coordinates
        not_none_keys = [key for key in src_dict if src_dict[key] is not None and dst_dict[key] is not None]

        if len(not_none_keys) == 0:
            raise ValueError("At least one valid fiducial is required to estimate the transformation.")

        # Create arrays of corresponding points
        src_points = np.array([src_dict[key] for key in not_none_keys])
        dst_points = np.array([dst_dict[key] for key in not_none_keys])

        return hipp.math.estimate_transformation_matrix(src_points, dst_points)

    def convert_in_camera_reference(
        self, scanning_resolution_mm: float = 0.02, flip_y: bool = True
    ) -> "FiducialsCoordinate":
        """
        Convert fiducial coordinates from pixel space to camera reference frame.

        This method transforms the coordinates based on a physical scanning resolution
        and recenters them with respect to the principal point. It optionally flips the Y-axis
        to match the camera's coordinate system convention (typically Y down in image space to Y up in camera space).

        The transformation is composed of:
            1. A translation that centers the coordinates on the principal point.
            2. A scaling to convert pixel units to millimeters.
            3. An optional Y-axis flip for compatibility with camera coordinate frames.

        Parameters
        ----------
        scanning_resolution_mm : float, optional
            The physical resolution of the scan in millimeters per pixel. Default is 0.02 mm/px.
            This factor is used to scale the coordinates from pixel units to millimeters.

        flip_y : bool, optional
            Whether to apply a vertical flip (Y-axis inversion). Default is True.
            This is useful for converting from image coordinates (Y-down) to standard camera coordinates (Y-up).

        Returns
        -------
        FiducialsCoordinate
            A new `FiducialsCoordinate` object with transformed coordinates in the camera reference frame.

        Raises
        ------
        ValueError
            If the principal point cannot be computed, which is required for centering the transformation.
        """
        principal_point = self.compute_principal_point()
        if principal_point is None:
            raise ValueError("Can't compute the principal point")

        # Step 1: translation
        T = hipp.math.affine_matrix(translate_x=-principal_point[0], translate_y=-principal_point[1])

        # Step 2: scale (and flip y if needed)
        S = hipp.math.affine_matrix(
            scale_x=scanning_resolution_mm, scale_y=scanning_resolution_mm * (-1 if flip_y else 1)
        )

        # Compose: scale * translate
        transformation_matrix = S @ T

        return self.transform(transformation_matrix)  # type: ignore [arg-type]

    def transform(self, transformation_matrix: cv2.typing.MatLike) -> "FiducialsCoordinate":
        """
        Apply a 2D affine or projective transformation to all fiducial coordinates.

        This method uses a 3x3 transformation matrix (homogeneous coordinates) to transform
        each coordinate in the fiducial set. It supports translation, rotation, scaling,
        shearing, or any combination of linear transformations expressible via matrix multiplication.

        Coordinates that are `None` are left unchanged (preserved as `None` in the result).

        Parameters
        ----------
        transformation_matrix : cv2.typing.MatLike
            A 3x3 transformation matrix (homogeneous coordinates) as used in OpenCV. This matrix
            defines the spatial transformation to be applied to each 2D point.

        Returns
        -------
        FiducialsCoordinate
            A new `FiducialsCoordinate` instance with all valid coordinates transformed
            by the given matrix. Invalid or missing coordinates remain as `None`.
        """
        return FiducialsCoordinate(
            self.apply(lambda coord: None if coord is None else hipp.math.transform_coord(coord, transformation_matrix))
        )


def detected_fiducials_to_csv(all_detections: dict[str, FiducialsCoordinate], csv_file_path: str) -> None:
    """
    Save all detections into a csv file
    """
    res: list[dict[str, float | str | None]] = []
    for key in sorted(all_detections.keys()):
        elem: dict[str, float | str | None] = {}
        elem["image_file_name"] = os.path.basename(key)
        for name, coord in all_detections[key].items():
            if coord is None:
                elem[f"{name}_x"] = None
                elem[f"{name}_y"] = None
            else:
                x, y = coord
                elem[f"{name}_x"] = x
                elem[f"{name}_y"] = y
        res.append(elem)

    fieldnames = list(res[0].keys())

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(res)


def detected_fiducials_from_csv(csv_file_path: str) -> dict[str, FiducialsCoordinate]:
    """
    Load fiducial detections from a CSV file exported by `detected_fiducials_to_csv`.

    Returns:
        A dictionary where keys are image file names and values are dictionaries of fiducial positions.
    """
    detections: dict[str, FiducialsCoordinate] = {}

    with open(csv_file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_file_name = row["image_file_name"]
            fiducials = FiducialsCoordinate()

            for key, value in row.items():
                if key == "image_file_name":
                    continue
                if key.endswith("_x"):
                    name = key[:-2]
                    x_str = value
                    y_str = row.get(f"{name}_y")

                    if not x_str or not y_str or x_str == "None" or y_str == "None":
                        fiducials[name] = None
                    else:
                        try:
                            x = float(x_str)
                            y = float(y_str)
                            fiducials[name] = (x, y)
                        except ValueError:
                            fiducials[name] = None  # Corrupted or invalid float value

            detections[image_file_name] = fiducials

    return detections
