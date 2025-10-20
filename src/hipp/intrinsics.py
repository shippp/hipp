"""
Copyright (c) 2025 HIPP developers

Description:
This module provides a class to manage Intrinsics camera parameters
"""

import numpy as np
import pandas as pd


class Intrinsics:
    """
    Represents the intrinsic camera calibration parameters, including focal length,
    pixel pitch, and fiducial mark coordinates in millimeters.

    This class provides convenient methods to:
    - Initialize camera intrinsics from a dictionary, list, or CSV file.
    - Validate fiducial key naming consistency.
    - Export and import intrinsic parameters to/from disk.
    - Classify unordered fiducial coordinates into canonical corner/midside positions.

    Attributes
    ----------
    fiducials_keys : list[str]
        The canonical order and naming of fiducial marks (corners and midsides).
    focal_length : float
        The focal length of the camera in millimeters.
    pixel_pitch : float
        The physical size of one pixel in millimeters.
    true_fiducials_mm : pandas.Series
        Series containing fiducial mark coordinates in millimeters, indexed by
        the expected keys (e.g. "corner_top_left_x", "corner_top_left_y", ...).
    """

    fiducials_keys = [
        "corner_top_left",
        "corner_top_right",
        "corner_bottom_right",
        "corner_bottom_left",
        "midside_left",
        "midside_top",
        "midside_right",
        "midside_bottom",
    ]

    def __init__(
        self,
        focal_length: float,
        pixel_pitch: float,
        true_fiducials_mm: pd.Series | dict[str, float] | None = None,
        principal_point: tuple[float, float] | None = None,
    ):
        """
        Initialize camera intrinsics.

        Parameters
        ----------
        focal_length : float
            Camera focal length in millimeters.
        pixel_pitch : float
            Pixel pitch in millimeters (physical size of one pixel).
        true_fiducials_mm : pandas.Series or dict[str, float]
            Fiducial coordinates (x, y) in millimeters. Must contain only valid keys
            matching the expected fiducial key names with "_x" and "_y" suffixes.

        Raises
        ------
        KeyError
            If one or more fiducial keys are not recognized.
        """
        self.focal_length: float = focal_length
        self.pixel_pitch: float = pixel_pitch

        fiducials_keys = [key + suffix for key in Intrinsics.fiducials_keys for suffix in ["_x", "_y"]]
        tmp_tfs = pd.Series({} if true_fiducials_mm is None else true_fiducials_mm)
        for key in tmp_tfs.index:
            if key not in fiducials_keys:
                raise KeyError(f"Unrecognized key `{key}` for true_fiducials_mm")

        self.true_fiducials_mm: pd.Series = tmp_tfs.reindex(fiducials_keys)
        self.principal_point = (np.nan, np.nan) if principal_point is None else principal_point

    def to_csv(self, csv_file: str) -> None:
        """
        Export intrinsic parameters to a CSV file.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file where the intrinsics will be saved.
        """
        data = {"focal_length": self.focal_length, "pixel_pitch": self.pixel_pitch}
        data.update(self.true_fiducials_mm.to_dict())
        data.update({"principal_point_x": self.principal_point[0], "principal_point_y": self.principal_point[1]})
        renamed_data = {
            (f"{k}_mm" if k.endswith(("_x", "_y")) else k): v
            for k, v in data.items()
        }
        df = pd.DataFrame([renamed_data])
        df.to_csv(csv_file, index=False)

    @classmethod
    def from_csv(cls, csv_file: str) -> "Intrinsics":
        """
        Load intrinsic parameters from a CSV file.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file to read.

        Returns
        -------
        Intrinsics
            An initialized instance of the Intrinsics class.
        """
        df_row = pd.read_csv(csv_file).iloc[0]
        df_row.index = df_row.index.str.replace("_mm", "", regex=False)
        focal_length = float(df_row["focal_length"])
        pixel_pitch = float(df_row["pixel_pitch"])
        principal_point = float(df_row["principal_point_x"]), float(df_row["principal_point_y"])

        fiducials_keys = [key + suffix for key in Intrinsics.fiducials_keys for suffix in ["_x", "_y"]]
        true_fiducials_mm = df_row[fiducials_keys]
        return cls(focal_length, pixel_pitch, true_fiducials_mm, principal_point)

    @classmethod
    def from_list(
        cls, focal_length: float, pixel_pitch: float, fiducial_coords: list[tuple[float, float]]
    ) -> "Intrinsics":
        """
        Build an Intrinsics instance from a list of fiducial coordinates.

        Parameters
        ----------
        focal_length : float
            Camera focal length in millimeters.
        pixel_pitch : float
            Pixel pitch in millimeters.
        fiducial_coords : list[tuple[float, float]]
            List of fiducial coordinates (x, y) in millimeters, unordered.

        Returns
        -------
        Intrinsics
            An initialized instance with properly classified fiducial marks.
        """
        classif = cls.classify_fiducials(fiducial_coords)
        true_fiducials_mm = {
            k + suffix: fiducial_coords[v][i] for k, v in classif.items() for i, suffix in enumerate(["_x", "_y"])
        }
        if "principal_point" in classif:
            principal_point = true_fiducials_mm["principal_point_x"], true_fiducials_mm["principal_point_y"]
            del true_fiducials_mm["principal_point_x"]
            del true_fiducials_mm["principal_point_y"]
        else:
            principal_point = None
        return cls(focal_length, pixel_pitch, true_fiducials_mm, principal_point)

    @staticmethod
    def classify_fiducials(fiducial_coords: list[tuple[float, float]]) -> dict[str, int]:
        """
        Classify unordered fiducial coordinates into canonical names.

        The method determines whether each fiducial corresponds to a corner or a midside
        position based on its relative location within the image plane. It supports both
        4-point and 8-point fiducial patterns.

        Parameters
        ----------
        fiducial_coords : list[tuple[float, float]]
            List of fiducial coordinates (x, y), in any order.

        Returns
        -------
        dict[str, int]
            Mapping from canonical fiducial key name to index in the input list.

        Raises
        ------
        ValueError
            If the number of fiducial points is not 4 or 8.
        """

        arr = np.array(fiducial_coords)

        # Create the grid 3x3 of with min and max bounding box and a small margin
        margin = 1.1
        x_lin = np.linspace(arr[:, 0].min() * margin, arr[:, 0].max() * margin, 4)  # 3 blocs => 4 limites
        y_lin = np.linspace(arr[:, 1].min() * margin, arr[:, 1].max() * margin, 4)

        # digitize to know wich point is in which block
        x_idx = np.digitize(arr[:, 0], x_lin) - 1
        y_idx = np.digitize(arr[:, 1], y_lin) - 1

        y_idx = 2 - y_idx  # inverse to have 0 = top, 2 = bottom

        # mapping for the name
        grid_mapping = {
            (0, 0): "corner_top_left",
            (0, 2): "corner_bottom_left",
            (2, 0): "corner_top_right",
            (2, 2): "corner_bottom_right",
            (0, 1): "midside_left",
            (1, 0): "midside_top",
            (2, 1): "midside_right",
            (1, 2): "midside_bottom",
            (1, 1): "principal_point",
        }

        # classified points
        mapping = {}
        for i, (xi, yi) in enumerate(zip(x_idx, y_idx)):
            if (xi, yi) in grid_mapping:
                mapping[grid_mapping[(xi, yi)]] = i

        return mapping
