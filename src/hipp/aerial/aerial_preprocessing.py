"""
Copyright (c) 2025 HIPP developers
Description: Contain the AerialPreprocessing class
"""

import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import hipp.aerial.core as core
import hipp.aerial.quality_control as qc
from hipp.aerial.fiducials import compute_principal_point, filter_by_angle, filter_scores_by_local_median
from hipp.image import resize_img

CORNER_FIDUCIAL_NAME = "corner_fiducial.png"
MIDSIDE_FIDUCIAL_NAME = "midside_fiducial.png"
SUBPIXEL_CORNER_FIDUCIAL_NAME = "subpixel_" + CORNER_FIDUCIAL_NAME
SUBPIXEL_MIDSIDE_FIDUCIAL_NAME = "subpixel_" + MIDSIDE_FIDUCIAL_NAME


class AerialPreprocessing:
    """
    A class for managing the preprocessing pipeline of aerial images using fiducial markers for spatial calibration.

    This class provides tools for detecting fiducials (corner and midside), generating and loading fiducial templates,
    and performing quality control through visualizations. It is particularly useful for aligning large batches of
    scanned or photographed aerial images using consistent reference points.

    Attributes:
        images_directory (str): Path to the directory containing input `.tif` images.
        fiducials_directory (str): Path to the directory where fiducial templates are stored or will be saved.
        qc_directory (str): Path to the directory where quality control images will be saved.

    Methods:
        create_fiducial_template(...):
            Generates and stores fiducial templates (regular and subpixel) from a selected image, either for corners
            or midsides.

        detect_fiducials(subpixel_factor=8, grid_size=3, quality_control=True, progress_bar=True) -> list[dict[str, dict[str, object]]]:
            Detects fiducials in all `.tif` images in the input directory and optionally generates quality control plots.

        load_fiducials_template() -> dict[str, cv2.typing.MatLike]:
            Loads fiducial templates from the disk and returns them in a dictionary format, ready to be passed to the
            fiducial detection function. Raises an error if the necessary templates are missing.
    """

    def __init__(
        self,
        images_directory: str,
        output_directory: str | None = None,
        fiducials_directory: str | None = None,
        qc_directory: str | None = None,
    ):
        """
        Initialize the preprocessing object with directories for images, output, fiducials, and quality control.

        Args:
            images_directory (str): Path to the directory containing input images.
            output_directory (str | None, optional): Directory to save processed images.
                Defaults to a subfolder 'output_images' within the parent of images_directory.
            fiducials_directory (str | None, optional): Directory containing fiducials data.
                Defaults to a subfolder 'fiducials' within the parent of images_directory.
            qc_directory (str | None, optional): Directory for saving quality control outputs.
                Defaults to a subfolder 'qc' within the parent of images_directory.

        Raises:
            FileNotFoundError: If the images_directory does not exist or contains no .tif files.
        """
        if not os.path.exists(images_directory):
            raise FileNotFoundError(f"The images directory {images_directory} does not exist")

        self.images_directory = images_directory

        # define all path in terms of images_directory if their are not provided
        project_dir = os.path.dirname(images_directory)
        self.output_directory = (
            os.path.join(project_dir, "output_images") if output_directory is None else output_directory
        )
        self.fiducials_directory = (
            os.path.join(project_dir, "fiducials") if fiducials_directory is None else fiducials_directory
        )
        self.qc_directory = os.path.join(project_dir, "qc") if qc_directory is None else qc_directory

        tif_files = sorted(glob.glob(os.path.join(images_directory, "*.tif"))) + sorted(
            glob.glob(os.path.join(images_directory, "*.TIF"))
        )

        if not tif_files:
            raise FileNotFoundError(f"No .tif file found in directory '{images_directory}'.")

        self.first_image_path = tif_files[0]

    def create_fiducial_template(
        self,
        distance_around_fiducial: int = 100,
        subpixel_distance_around_fiducial: int = 100,
        corner: bool = False,
        midside: bool = False,
        fiducial_coordinate: tuple[int, int] | None = None,
        subpixel_center_coordinate: tuple[int, int] | None = None,
        overwrite: bool = False,
    ) -> dict[str, tuple[int, int] | None]:
        """
        Creates and saves fiducial templates (standard and subpixel) for later matching tasks.

        This method extracts image patches centered around a fiducial marker (either corner or midside)
        from a grayscale reference image. It generates both a standard-resolution template and a higher-resolution
        subpixel template, which can be used for more precise detection.

        The method supports overwriting existing templates and allows manual specification of the fiducial
        center (standard and subpixel) if known.

        Args:
            distance_around_fiducial (int): Half-width of the square patch (in pixels) to extract for the standard template.
            subpixel_distance_around_fiducial (int): Half-width of the patch (in pixels) to extract for the high-resolution template.
            corner (bool): Whether to generate a template for a corner fiducial. Must be mutually exclusive with `midside`.
            midside (bool): Whether to generate a template for a midside fiducial. Must be mutually exclusive with `corner`.
            fiducial_coordinate (tuple[int, int] | None): The (x, y) pixel location of the fiducial center in the original image.
                If None, an interactive tool or default behavior should be used to define the location.
            subpixel_center_coordinate (tuple[int, int] | None): Pixel location of the center in the upsampled image space
                for creating the subpixel template. If None, it defaults to the same logic as `fiducial_coordinate` but in higher resolution.
            overwrite (bool): If True, any existing saved templates will be regenerated and overwritten.

        Raises:
            ValueError: If both `corner` and `midside` are True or both are False (must specify exactly one).

        Notes:
            - Templates are saved in the directory defined by `self.fiducials_directory`.
            - Template filenames depend on the type (`corner` or `midside`) and resolution (standard or subpixel).
            - Subpixel templates are created by resizing the standard template to a higher resolution before cropping.
            - This method assumes that `self.first_images` contains a path to the reference grayscale image.
        """
        # Ensure exactly one fiducial type is selected
        if (not corner and not midside) or (corner and midside):
            raise ValueError("Need either corner of midside")

        os.makedirs(self.fiducials_directory, exist_ok=True)

        # Determine filenames based on fiducial type
        fiducial_name = MIDSIDE_FIDUCIAL_NAME if midside else CORNER_FIDUCIAL_NAME
        fiducial_path = os.path.join(self.fiducials_directory, fiducial_name)

        # Create and save the regular-resolution fiducial template if needed
        if not os.path.exists(fiducial_path) or overwrite:
            img = cv2.imread(self.first_image_path, cv2.IMREAD_GRAYSCALE)

            fiducial, fiducial_coordinate = core.create_fiducial_template_from_image(
                img, fiducial_coordinate, distance_around_fiducial
            )
            cv2.imwrite(fiducial_path, fiducial)
        else:
            fiducial = cv2.imread(fiducial_path, cv2.IMREAD_GRAYSCALE)

        # Prepare subpixel (high-res) version of the fiducial template
        subpixel_fiducial_name = SUBPIXEL_MIDSIDE_FIDUCIAL_NAME if midside else SUBPIXEL_CORNER_FIDUCIAL_NAME
        subpixel_fiducial_path = os.path.join(self.fiducials_directory, subpixel_fiducial_name)

        if not os.path.exists(subpixel_fiducial_path) or overwrite:
            fiducial = resize_img(fiducial)
            subpixel_fiducial, subpixel_center_coordinate = core.create_fiducial_template_from_image(
                fiducial, subpixel_center_coordinate, subpixel_distance_around_fiducial
            )
            cv2.imwrite(subpixel_fiducial_path, subpixel_fiducial)
        return {"fiducial_coordinate": fiducial_coordinate, "subpixel_center_coordinate": subpixel_center_coordinate}

    def detect_fiducials(
        self,
        subpixel_factor: float = 8,
        grid_size: int = 3,
        quality_control: bool = True,
        progress_bar: bool = True,
        max_workers: int = 5,
    ) -> pd.DataFrame:
        qc_detections = os.path.join(self.qc_directory, "fiducial_detections")
        if quality_control:
            os.makedirs(qc_detections, exist_ok=True)

        template_fiducial_paths = self.get_fiducial_template_paths()
        image_paths = [
            os.path.join(self.images_directory, f)
            for f in sorted(os.listdir(self.images_directory))
            if f.endswith(".tif")
        ]

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for image_path in image_paths:
                qc_output_path = (
                    os.path.join(qc_detections, os.path.basename(image_path).replace(".tif", ".png"))
                    if quality_control
                    else None
                )
                futures.append(
                    executor.submit(
                        core.detect_fiducials,
                        image_path,
                        **template_fiducial_paths,
                        subpixel_factor=subpixel_factor,
                        grid_size=grid_size,
                        qc_output_path=qc_output_path,
                    )
                )
            iterable = tqdm(as_completed(futures), total=len(futures)) if progress_bar else as_completed(futures)

            for future in iterable:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[!] Error: {e}")

        df = pd.DataFrame(results)
        df = df.set_index("image_id").sort_index()

        if quality_control:
            qc.plot_detection_score_boxplot(df, os.path.join(self.qc_directory, "Detection_matching_score_boxplot.png"))
        return df

    def filter_detected_fiducials(
        self,
        detected_fiducials_df: pd.DataFrame,
        score_threshold: float = 0.1,
        angle_threshold: float = 0.005,
        quality_control: bool = True,
    ) -> pd.DataFrame:
        # filtering with local median and remove score columns
        filtered_scores = filter_scores_by_local_median(detected_fiducials_df, score_threshold)

        filtered_angles = filter_by_angle(detected_fiducials_df, angle_threshold)

        # combine both filtering
        combined = filtered_scores.copy()
        for col in filtered_scores.columns:
            if col in filtered_angles.columns:
                combined[col] = combined[col].fillna(filtered_angles[col])

        # compute principal points and store them in principal_point_x, principal_point_y
        combined[["principal_point_x", "principal_point_y"]] = combined.apply(
            lambda row: pd.Series(compute_principal_point(row)), axis=1
        )
        # Check for missing principal points
        missing_mask = combined[["principal_point_x", "principal_point_y"]].isna().any(axis=1)
        if missing_mask.any():
            missing_ids = combined.index[missing_mask].tolist()
            warnings.warn(
                f"Principal point could not be computed for {len(missing_ids)} detection(s): {missing_ids}",
                UserWarning,
            )
        if quality_control:
            fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)
            fig.suptitle("Comparison of Fiducial Deviations")
            fig.supylabel("Sum of absolute deviations to mean (px)")
            qc.plot_fiducial_deviation(detected_fiducials_df, axs[0], title="before filtering")
            qc.plot_fiducial_deviation(combined, axs[1], title="after filtering")
            plt.tight_layout()
            fig.savefig(os.path.join(self.qc_directory, "filtered_deviation.png"), dpi=300, bbox_inches="tight")
            plt.show()

        return combined

    def get_fiducial_template_paths(self) -> dict[str, str]:
        paths = {
            "corner_fiducial_path": os.path.join(self.fiducials_directory, CORNER_FIDUCIAL_NAME),
            "midside_fiducial_path": os.path.join(self.fiducials_directory, MIDSIDE_FIDUCIAL_NAME),
            "subpixel_corner_fiducial_path": os.path.join(self.fiducials_directory, SUBPIXEL_CORNER_FIDUCIAL_NAME),
            "subpixel_midside_fiducial_path": os.path.join(self.fiducials_directory, SUBPIXEL_MIDSIDE_FIDUCIAL_NAME),
        }
        return {key: path for key, path in paths.items() if os.path.exists(path)}

    def images_restitution(
        self,
        fiducials_detections_df: pd.DataFrame,
        true_fiducials_mm: dict[str, float],
        scanning_resolution_mm: float = 0.02,
        image_square_dim: int | None = 10800,
        interpolation_flag: int = cv2.INTER_CUBIC,
        clahe_enhancement: bool = True,
        quality_control: bool = True,
        max_workers: int = 5,
        dry_run: bool = False,
    ) -> dict[str, cv2.typing.MatLike]:
        """
        Perform batch image restitution on a set of fiducial detections using reference coordinates.

        This method applies the `image_restitution` function to a collection of images (or fiducial sets),
        computing an affine transformation for each one. If `image_path` and `output_image_path` are provided,
        the method also performs image transformation, cropping, and optional enhancement (CLAHE).
        Parallel processing is used to speed up execution.

        Returns
        -------
        dict[str, cv2.typing.MatLike]
            A dictionary mapping image file paths to the computed 3×3 transformation matrices.

        Notes
        -----
        - This method uses multiprocessing (`ProcessPoolExecutor`) to parallelize image restitution tasks.
        - If `dry_run=True`, image files are not required and only transformation matrices are estimated.
        - If the fiducial markers are located outside the cropped region (especially in cases of large cropping),
          they will be excluded from the final transformed image.
        - RMSE plots comparing detected and reference fiducials are saved in the quality control directory
          if `quality_control=True`.
        """
        qc_restitution = os.path.join(self.qc_directory, "images_restitution")
        if quality_control:
            os.makedirs(qc_restitution, exist_ok=True)

        if dry_run:
            # Serial execution without image processing (transformation matrix only)
            results = {
                index: core.image_restitution(
                    row,
                    true_fiducials_mm,
                    None,
                    None,
                    scanning_resolution_mm,
                    image_square_dim,
                    interpolation_flag,
                    clahe_enhancement,
                )
                for index, row in fiducials_detections_df.iterrows()
            }
        else:
            results = {}
            future_to_key = {}
            # Parallel execution with image processing using process pool
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for index, row in fiducials_detections_df.iterrows():
                    image_path = os.path.join(self.images_directory, f"{index}.tif")
                    output_image_path = os.path.join(self.output_directory, f"{index}.tif")

                    # Submit task for each image to the executor
                    future = executor.submit(
                        core.image_restitution,
                        row,
                        true_fiducials_mm,
                        image_path,
                        output_image_path,
                        scanning_resolution_mm,
                        image_square_dim,
                        interpolation_flag,
                        clahe_enhancement,
                    )
                    future_to_key[future] = index

                # Progress bar with result collection
                for future in tqdm(
                    as_completed(future_to_key), total=len(future_to_key), desc="Images restitution", unit="image"
                ):
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        print(f"Preproccessing error of {key} : {e}")

        # Extract transformation matrices from results
        transformations_matrixs = {key: val["transformation_matrix"] for key, val in results.items()}
        if quality_control:
            # Generate QC RMSE plots
            rmse_before_transform = {key: val["rmse_before_transformation"] for key, val in results.items()}
            rmse_after_transform = {key: val["rmse_after_transformation"] for key, val in results.items()}

            qc.plot_rmse_after_vs_before(
                rmse_before_transform, rmse_after_transform, os.path.join(qc_restitution, "Fiducials_RMSE.png")
            )
        return transformations_matrixs

    def plot_fiducial_templates(self) -> None:
        """
        Plots available fiducial and subpixel fiducial templates in a 2x2 grid.

        Templates are loaded using `load_fiducials_template()`.
        Only existing templates are displayed. Missing ones are skipped.

        Raises:
            FileNotFoundError: If no templates are found.
        """
        template_paths = self.get_fiducial_template_paths()

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.flatten()

        # Plot each template if it exists
        for i, key in enumerate(template_paths):
            fiducial_image = cv2.imread(template_paths[key], cv2.IMREAD_GRAYSCALE)
            axes[i].imshow(fiducial_image, cmap="gray")
            axes[i].set_title(key)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
