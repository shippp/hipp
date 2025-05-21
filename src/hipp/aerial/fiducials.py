from typing import Callable, Generic, TypeVar, cast

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

import hipp.math

T = TypeVar("T")
U = TypeVar("U")


class Fiducials(dict[str, T], Generic[T]):
    corners_label = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left"]
    midsides_label = ["midside_left", "midside_top", "midside_right", "midside_bottom"]

    def get_corners(self) -> dict[str, T]:
        return {key: self[key] for key in Fiducials.corners_label}

    def get_midsides(self) -> dict[str, T]:
        return {key: self[key] for key in Fiducials.midsides_label}

    def get_fiducials(self) -> dict[str, T]:
        result = {}
        if self.as_corners():
            result.update(self.get_corners())
        if self.as_midsides():
            result.update(self.get_midsides())
        return result

    def as_corners(self) -> bool:
        return all(key in self for key in self.corners_label)

    def as_midsides(self) -> bool:
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

        if len(not_none_keys) == 1:
            # If only one point is available, compute a pure translation
            src_pt = src_points[0]
            dst_pt = dst_points[0]
            dx, dy = dst_pt[0] - src_pt[0], dst_pt[1] - src_pt[1]

            # Construct 3x3 homogeneous transformation matrix (translation only)
            transform_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
        else:
            # Estimate a similarity transform (translation, rotation, scale) using at least two points
            transform = SimilarityTransform()
            transform.estimate(src_points, dst_points)
            transform_matrix = transform.params.astype(np.float32)

        return cast(cv2.typing.MatLike, transform_matrix)

    def convert_in_camera_reference(
        self, scanning_resolution_mm: float = 0.02, flip_y: bool = True
    ) -> "FiducialsCoordinate":
        principal_point = self.compute_principal_point()
        if principal_point is None:
            raise ValueError("Can't compute the principal point")

        # 1. Translation matrix to center
        T = np.array([[1, 0, -principal_point[0]], [0, 1, -principal_point[1]], [0, 0, 1]], dtype=np.float32)

        # 2. Scaling matrix
        S = np.array([[scanning_resolution_mm, 0, 0], [0, scanning_resolution_mm, 0], [0, 0, 1]], dtype=np.float32)

        # Optional Y flip
        Y_FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

        # Final transformation: scale * translate
        transformation_matrix = S @ T

        if flip_y:
            transformation_matrix = Y_FLIP @ transformation_matrix

        return self.transform(transformation_matrix)

    def transform(self, transformation_matrix: cv2.typing.MatLike) -> "FiducialsCoordinate":
        return FiducialsCoordinate(
            self.apply(lambda coord: None if coord is None else hipp.math.transform_coord(coord, transformation_matrix))
        )
