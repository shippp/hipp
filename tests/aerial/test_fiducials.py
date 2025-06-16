import copy

import numpy as np

import hipp.math
from hipp.aerial.fiducials import Fiducials, FiducialsCoordinate

dict1 = {"corner_top_right": 54, "corner_bottom_left": 78, "corner_top_left": None, "corner_bottom_right": 12}

dict2 = {"midside_right": 63, "midside_bottom": 84, "midside_left": 21, "midside_top": 42}

dict3 = {**dict1, **dict2}


class TestFiducials:
    def test_get_corners(self) -> None:
        f1 = Fiducials(dict1)
        corners = f1.get_corners()
        assert list(corners.keys()) == [
            "corner_top_left",
            "corner_top_right",
            "corner_bottom_right",
            "corner_bottom_left",
        ]
        assert list(corners.values()) == [None, 54, 12, 78]

    def test_get_midsides(self) -> None:
        f3 = Fiducials(dict3)
        corners = f3.get_midsides()
        assert list(corners.keys()) == ["midside_left", "midside_top", "midside_right", "midside_bottom"]
        assert list(corners.values()) == [21, 42, 63, 84]

    def test_get_fiducials(self) -> None:
        f3 = Fiducials(dict3)
        all_fiducials = f3.get_fiducials()
        expected_keys = [
            "corner_top_left",
            "corner_top_right",
            "corner_bottom_right",
            "corner_bottom_left",
            "midside_left",
            "midside_top",
            "midside_right",
            "midside_bottom",
        ]
        assert set(all_fiducials.keys()) == set(expected_keys)

    def test_as_corners(self) -> None:
        f1 = Fiducials(dict1)
        assert f1.as_corners() is True

        incomplete = dict(dict1)
        del incomplete["corner_top_left"]
        f_incomplete = Fiducials(incomplete)
        assert f_incomplete.as_corners() is False

    def test_as_midsides(self) -> None:
        f2 = Fiducials(dict2)
        assert f2.as_midsides() is True

        incomplete = dict(dict2)
        del incomplete["midside_top"]
        f_incomplete = Fiducials(incomplete)
        assert f_incomplete.as_midsides() is False

    def test_apply(self) -> None:
        f2 = Fiducials(dict2)
        # Apply a simple function: double the value
        f_applied = f2.apply(lambda x: x * 2)
        expected_values = [63 * 2, 84 * 2, 21 * 2, 42 * 2]
        assert list(f_applied.values()) == expected_values

    def test_apply_with_key(self) -> None:
        f2 = Fiducials(dict2)
        # Append key length to the value (for test purposes)
        f_applied = f2.apply_with_key(lambda k, v: v + len(k))
        for key in dict2:
            assert f_applied[key] == dict2[key] + len(key)


coord_dict1 = {
    "corner_bottom_left": (-109.990, -110.002),  # 1
    "corner_top_right": (110.010, 109.999),  # 2
    "corner_top_left": (-109.989, 109.9995),  # 3
    "corner_bottom_right": (109.998, -110.002),  # 4
    "midside_left": (-111.998, -0.004),  # 5
    "midside_right": (112.004, 0.000),  # 6
    "midside_top": (-0.014, 111.993),  # 7
    "midside_bottom": (0.000, -112.002),  # 8
}

coord_dict2 = {
    "corner_bottom_left": None,  # 1
    "corner_top_right": (110.010, 109.999),  # 2
    "corner_top_left": (-109.989, 109.9995),  # 3
    "corner_bottom_right": None,  # 4
    "midside_left": None,  # 5
    "midside_right": (112.004, 0.000),  # 6
    "midside_top": (-0.014, 111.993),  # 7
    "midside_bottom": None,  # 8
}


class TestFiducialsCoordinate:
    def test_transform(self) -> None:
        matrix = hipp.math.affine_matrix(scale_x=2, scale_y=2)
        fc1 = FiducialsCoordinate(coord_dict1)

        fc_doubled = fc1.transform(matrix)
        for key in fc1.get_fiducials():
            assert np.allclose(np.array(fc1[key]), np.array(fc_doubled[key]) / 2)

    def test_compute_principal_point(self) -> None:
        Y_FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

        fc1 = FiducialsCoordinate(coord_dict1)
        fc2 = FiducialsCoordinate(coord_dict2)
        fc3 = fc2.transform(Y_FLIP)

        principal1 = fc1.compute_principal_point()
        principal2 = fc2.compute_principal_point()
        principal3 = fc3.compute_principal_point()

        assert np.allclose(principal1, (0, 0), atol=1e-2)  # type: ignore [arg-type]
        assert np.allclose(principal2, (0, 0), atol=1e-2)  # type: ignore [arg-type]
        assert np.allclose(principal3, (0, 0), atol=1e-2)  # type: ignore [arg-type]

    def test_validate_angles(self) -> None:
        fc1 = FiducialsCoordinate(coord_dict1)
        fc2 = copy.deepcopy(fc1)
        fc2["corner_bottom_left"] = (-119.990, -120.002)

        angles_valid1 = fc1.validate_angles()
        angles_valid2 = fc2.validate_angles()

        assert all(list(angles_valid1.values()))

        # test if all other fiducials are still valid with angles but not the modified one
        assert all(angles_valid2[key] for key in angles_valid2 if key != "corner_bottom_left")
        assert not angles_valid2["corner_bottom_left"]

    def test_estimate_transformation_matrix(self) -> None:
        matrix = hipp.math.affine_matrix(
            rotation_deg=45, scale_x=2, scale_y=2, translate_x=10, translate_y=10, shear_deg=2
        )
        fc1 = FiducialsCoordinate(coord_dict1)
        fc2 = fc1.transform(matrix)

        estimate_matrix = fc1.estimate_transformation_matrix(fc2)
        assert np.allclose(estimate_matrix, matrix)

    def test_convert_in_camera_reference(self) -> None:
        fc1 = FiducialsCoordinate(coord_dict1)
        fc2 = FiducialsCoordinate(coord_dict2)

        assert fc1.compute_principal_point() != (0, 0)
        assert fc2.compute_principal_point() != (0, 0)

        fc1_converted = fc1.convert_in_camera_reference(1)
        fc2_converted = fc2.convert_in_camera_reference(1)

        assert fc1_converted["principal_point"] == (0, 0)
        assert fc2_converted["principal_point"] == (0, 0)
