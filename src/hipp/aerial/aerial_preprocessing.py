"""
Module: preprocessing.py
Author: godinlu
Date: 29
Description: Contain the AerialPreprocessing class
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import hipp.aerial.quality_control as qc
from hipp.aerial.core import create_fiducial_template_from_image, detect_fiducials
from hipp.image import resize_img
from hipp.typing import DetectedFiducials

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
        output_directory: str = "./preprocess_images",
        fiducials_directory: str = "./fiducials",
        qc_directory: str = "./qc",
    ):
        if not os.path.exists(images_directory):
            raise FileNotFoundError(f"The images directory {images_directory} does not exist")

        self.images_directory = images_directory
        self.output_directory = output_directory
        self.fiducials_directory = fiducials_directory
        self.qc_directory = qc_directory

        tif_files = sorted(glob.glob(os.path.join(images_directory, "*.tif"))) + sorted(
            glob.glob(os.path.join(images_directory, "*.TIF"))
        )

        if not tif_files:
            raise FileNotFoundError(f"No .tif file found in directory '{images_directory}'.")

        self.first_images = tif_files[0]

        os.makedirs(self.fiducials_directory, exist_ok=True)
        os.makedirs(self.qc_directory, exist_ok=True)

    def create_fiducial_template(
        self,
        distance_around_fiducial: int = 100,
        subpixel_distance_around_fiducial: int = 100,
        corner: bool = False,
        midside: bool = False,
        fiducial_coordinate: tuple[int, int] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Creates and saves fiducial templates (standard and subpixel) for later matching.

        This method extracts a fiducial pattern (either a corner or midside marker) from a reference image
        and saves it as a grayscale template. A higher-resolution version is also created for subpixel
        matching by resizing the original template and extracting a sub-region around the fiducial.
        These templates are saved to the appropriate directory and used in downstream detection routines.

        Args:
            distance_around_fiducial: Pixel distance from the fiducial center to include in the regular template.
            subpixel_distance_around_fiducial: Pixel distance to include in the subpixel (higher-res) template.
            corner: Whether to generate a template for a corner fiducial. Mutually exclusive with `midside`.
            midside: Whether to generate a template for a midside fiducial. Mutually exclusive with `corner`.
            fiducial_coordinate: The (x, y) coordinate in the image where the fiducial is located.
                                 If None, an interactive picker or default mechanism must be used.
            overwrite: If True, existing templates will be regenerated and overwritten.

        Raises:
            ValueError: If neither `corner` nor `midside` is specified, or if both are set to True.

        Notes:
            - Templates are saved as grayscale `.png` or `.tif` images depending on the implementation.
            - Subpixel templates are useful for refined matching accuracy, especially in high-precision contexts.
        """
        # Ensure exactly one fiducial type is selected
        if (not corner and not midside) or (corner and midside):
            raise ValueError("Need either corner of midside")

        # Determine filenames based on fiducial type
        fiducial_name = MIDSIDE_FIDUCIAL_NAME if midside else CORNER_FIDUCIAL_NAME
        fiducial_path = os.path.join(self.fiducials_directory, fiducial_name)

        # Create and save the regular-resolution fiducial template if needed
        if not os.path.exists(fiducial_path) or overwrite:
            img = cv2.imread(self.first_images, cv2.IMREAD_GRAYSCALE)

            fiducial = create_fiducial_template_from_image(img, fiducial_coordinate, distance_around_fiducial)
            cv2.imwrite(fiducial_path, fiducial)
        else:
            fiducial = cv2.imread(fiducial_path, cv2.IMREAD_GRAYSCALE)

        # Prepare subpixel (high-res) version of the fiducial template
        subpixel_fiducial_name = SUBPIXEL_MIDSIDE_FIDUCIAL_NAME if midside else SUBPIXEL_CORNER_FIDUCIAL_NAME
        subpixel_fiducial_path = os.path.join(self.fiducials_directory, subpixel_fiducial_name)

        if not os.path.exists(subpixel_fiducial_path) or overwrite:
            fiducial = resize_img(fiducial)
            subpixel_fiducial = create_fiducial_template_from_image(fiducial, None, subpixel_distance_around_fiducial)
            cv2.imwrite(subpixel_fiducial_path, subpixel_fiducial)

    def detect_fiducials(
        self,
        subpixel_factor: float = 8,
        grid_size: int = 3,
        quality_control: bool = True,
        progress_bar: bool = True,
        max_workers: int = 4,
    ) -> dict[str, DetectedFiducials]:
        """
        Detects fiducial markers in a batch of grayscale `.tif` images located in the input directory using multithreading.

        This method loads previously generated corner and midside fiducial templates (both standard and subpixel),
        then applies fiducial detection to each image in parallel using multithreading. Subpixel refinement is applied
        when high-resolution templates are available. Optionally, quality control images are generated for visual inspection
        of the detected subpixel centers.

        The area around each fiducial used for quality control visualization is automatically inferred from the template size.

        Args:
            subpixel_factor: Upscaling factor used during subpixel refinement. Higher values yield more precise results
                             but increase computation time.
            grid_size: The number of subdivisions along one dimension of the image (must be an odd number, e.g., 3 for 3×3).
            quality_control: If True, generates and saves quality control images showing detected fiducials and labels.
            progress_bar: If True, displays a progress bar during image processing.
            max_workers: Maximum number of threads used for parallel processing.

        Returns:
            A dictionary where keys are image file paths and values are the detection results for each image.
            Each fiducial detection entry contains:
                - "approx_center": (x, y) coordinates from initial template matching,
                - "approx_score": correlation score of the coarse match,
                - "subpixel_center": refined center (if subpixel template is provided),
                - "subpixel_score": subpixel template matching score (if applicable).

        Raises:
            FileNotFoundError: If any required fiducial template file is missing.

        Notes:
            - Fiducial templates must be pre-generated using `create_fiducial_template`.
            - Input images are assumed to be single-channel (grayscale) `.tif` files.
            - Quality control images are saved in PNG format under `self.qc_directory`.
        """
        qc_detection_dir = os.path.join(self.qc_directory, "fiducials_detection")
        individuals_qc_dir = os.path.join(qc_detection_dir, "individuals")
        if quality_control:
            os.makedirs(individuals_qc_dir, exist_ok=True)

        # Load previously generated fiducial templates (corner and midside, standard and subpixel)
        fiducials_template = self.load_fiducials_template()

        # Automatically determine the region around the fiducial to extract for quality control images
        # This is based on the size of one of the loaded templates (assumes at least one is present)
        any_template = (
            fiducials_template["corner_fiducial"]
            if "corner_fiducial" in fiducials_template
            else fiducials_template["midside_fiducial"]
        )
        distance_around_fiducial = max(*any_template.shape[:2]) // 2  # Half-size used to crop around the center

        # Find all .tif images in the input directory
        tif_files = glob.glob(os.path.join(self.images_directory, "*.tif"))
        results = {}

        # Function that processes a single image: detects fiducials and optionally generates a QC image
        def process_image(image_path: str) -> tuple[str, DetectedFiducials]:
            result = detect_fiducials(
                image_path=image_path,
                **fiducials_template,  # Unpack the loaded templates into the function
                subpixel_factor=subpixel_factor,
                grid_size=grid_size,
            )
            if quality_control:
                # Create a diagnostic image showing fiducials and their subpixel centers
                qc_img = qc.generate_fiducial_qc_image_from_detection(
                    image_path,
                    result,
                    distance_around_fiducial=distance_around_fiducial,
                    grid_cols=4,
                )
                # Save QC image with same name as original image but in PNG format
                qc_path = os.path.join(individuals_qc_dir, os.path.basename(image_path).replace(".tif", ".png"))
                cv2.imwrite(qc_path, qc_img)
            return image_path, result

        # Run detection in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, path) for path in tif_files]

            # Optionally wrap with tqdm progress bar
            futures_iter = (
                tqdm(as_completed(futures), total=len(futures), desc="Fiducials detection")
                if progress_bar
                else as_completed(futures)
            )

            # Collect results as they complete
            for future in futures_iter:
                image_path, detection_result = future.result()
                results[image_path] = detection_result

        if quality_control:
            deviation_boxplot = qc.plot_fiducial_center_deviation_boxplots(results)
            plt.close()
            score_boxplot = qc.plot_fiducial_score_boxplots(results)
            plt.close()
            deviation_boxplot.savefig(os.path.join(qc_detection_dir, "deviation_boxplot.png"))
            score_boxplot.savefig(os.path.join(qc_detection_dir, "score_boxplot.png"))

        return results

    def load_fiducials_template(self) -> dict[str, cv2.typing.MatLike]:
        """
        Loads the fiducial and subpixel fiducial templates from the fiducials_directory.

        Raises an error if none of the expected templates are found.

        Returns:
            A dictionary with keys:
                - "corner_fiducial"
                - "midside_fiducial"
                - "subpixel_corner_fiducial"
                - "subpixel_midside_fiducial"
            and values corresponding to the loaded grayscale images.
            Only entries for found files are included.

        Raises:
            FileNotFoundError: If none of the fiducial templates are found in the directory.
        """
        templates = {}
        paths = {
            "corner_fiducial": os.path.join(self.fiducials_directory, CORNER_FIDUCIAL_NAME),
            "midside_fiducial": os.path.join(self.fiducials_directory, MIDSIDE_FIDUCIAL_NAME),
            "subpixel_corner_fiducial": os.path.join(self.fiducials_directory, SUBPIXEL_CORNER_FIDUCIAL_NAME),
            "subpixel_midside_fiducial": os.path.join(self.fiducials_directory, SUBPIXEL_MIDSIDE_FIDUCIAL_NAME),
        }

        for key, path in paths.items():
            if os.path.exists(path):
                templates[key] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if not templates:
            raise FileNotFoundError("No fiducial templates found in directory: " + self.fiducials_directory)

        return templates
