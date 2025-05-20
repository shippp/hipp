"""
Copyright (c) 2025 HIPP developers
Description: Contain the AerialPreprocessing class
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm

import hipp.aerial.core as core
import hipp.aerial.quality_control as qc
from hipp.image import resize_img
from hipp.typing import Fiducials

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

    def create_fiducial_template(
        self,
        distance_around_fiducial: int = 100,
        subpixel_distance_around_fiducial: int = 100,
        corner: bool = False,
        midside: bool = False,
        fiducial_coordinate: tuple[int, int] | None = None,
        subpixel_center_coordinate: tuple[int, int] | None = None,
        overwrite: bool = False,
    ) -> None:
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
            img = cv2.imread(self.first_images, cv2.IMREAD_GRAYSCALE)

            fiducial = core.create_fiducial_template_from_image(img, fiducial_coordinate, distance_around_fiducial)
            cv2.imwrite(fiducial_path, fiducial)
        else:
            fiducial = cv2.imread(fiducial_path, cv2.IMREAD_GRAYSCALE)

        # Prepare subpixel (high-res) version of the fiducial template
        subpixel_fiducial_name = SUBPIXEL_MIDSIDE_FIDUCIAL_NAME if midside else SUBPIXEL_CORNER_FIDUCIAL_NAME
        subpixel_fiducial_path = os.path.join(self.fiducials_directory, subpixel_fiducial_name)

        if not os.path.exists(subpixel_fiducial_path) or overwrite:
            fiducial = resize_img(fiducial)
            subpixel_fiducial = core.create_fiducial_template_from_image(
                fiducial, subpixel_center_coordinate, subpixel_distance_around_fiducial
            )
            cv2.imwrite(subpixel_fiducial_path, subpixel_fiducial)

    def detect_fiducials(
        self,
        subpixel_factor: float = 8,
        grid_size: int = 3,
        quality_control: bool = True,
        progress_bar: bool = True,
        max_workers: int = 4,
    ) -> tuple[dict[str, Fiducials], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """
        Detects fiducial markers in a batch of grayscale `.tif` images using multithreaded template matching.

        This method loads previously generated fiducial templates (corner and midside types, both standard and subpixel),
        then applies coarse and optionally subpixel-level detection to each image in parallel. Detection results are
        returned along with confidence scores for each detected point. If enabled, quality control images and plots are
        generated to visualize detection accuracy and score distributions.

        Args:
            subpixel_factor (float): Upscaling factor for subpixel refinement. Higher values increase precision but also processing time.
            grid_size (int): Number of subdivisions per image dimension (must be odd, e.g., 3 for a 3×3 grid).
            quality_control (bool): If True, saves quality control (QC) images and summary plots for visual inspection.
            progress_bar (bool): If True, displays a progress bar during processing.
            max_workers (int): Number of threads used for parallel image processing.

        Returns:
            Tuple containing three dictionaries:
                - detections (dict[str, Fiducials]):
                    Mapping from image file paths to detected fiducial positions.
                    Each detection contains:
                        * midside_* or corner_* keys → tuple[float, float]: detected center coordinates
                        * "principal_point" → tuple[float, float]: estimated principal point from valid segments

                - scores (dict[str, dict[str, float]]):
                    Mapping from image paths to initial template matching scores for each detected fiducial.

                - subpixel_scores (dict[str, dict[str, float]]):
                    Mapping from image paths to subpixel template matching scores for each fiducial (if applicable).

        Raises:
            FileNotFoundError: If any required template file is missing (standard or subpixel templates).

        Notes:
            - Templates must be created beforehand using `create_fiducial_template`.
            - Input images must be grayscale `.tif` files and are loaded from `self.images_directory`.
            - Quality control plots include deviation boxplots, principal point deviation bar plots, and matching score distributions.
              These are saved under `self.qc_directory/fiducials_detection/`.
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
        all_detections = {}
        all_scores, all_subpixel_scores = {}, {}

        # Function that processes a single image: detects fiducials and optionally generates a QC image
        def process_image(image_path: str) -> tuple[str, Fiducials, dict[str, float], dict[str, float]]:
            fiducials_detection, scores, subpixel_scores = core.detect_fiducials(
                image_path=image_path,
                **fiducials_template,  # Unpack the loaded templates into the function
                subpixel_factor=subpixel_factor,
                grid_size=grid_size,
            )
            if quality_control:
                # Create a diagnostic image showing fiducials and their subpixel centers
                qc_img = qc.generate_fiducial_qc_image_from_detection(
                    image_path,
                    fiducials_detection,
                    distance_around_fiducial=distance_around_fiducial,
                    grid_cols=4,
                )
                # Save QC image with same name as original image but in PNG format
                qc_path = os.path.join(individuals_qc_dir, os.path.basename(image_path).replace(".tif", ".png"))
                cv2.imwrite(qc_path, qc_img)
            return image_path, fiducials_detection, scores, subpixel_scores

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
                image_path, fiducials_detection, scores, subpixel_scores = future.result()
                fiducials_detection["principal_point"] = core.compute_principal_point_from_valid_segments(
                    fiducials_detection
                )
                all_detections[image_path] = fiducials_detection
                all_scores[image_path] = scores
                all_subpixel_scores[image_path] = subpixel_scores

        # save all the plot if quality control is actived
        if quality_control:
            qc.save_fiducials_detection_qc(all_detections, all_scores, all_subpixel_scores, qc_detection_dir)

        return all_detections, all_scores, all_subpixel_scores

    def process_fiducials_detection(
        self,
        all_detections: dict[str, Fiducials],
        all_scores: dict[str, dict[str, float]],
        all_subpixel_scores: dict[str, dict[str, float]],
        degree_threshold: float = 0.05,
        score_margin: float = 0.1,
        quality_control: bool = True,
    ) -> dict[str, Fiducials]:
        qc_detection_dir = os.path.join(self.qc_directory, "fiducials_detection")
        if quality_control:
            os.makedirs(qc_detection_dir, exist_ok=True)

        processed_detections = core.process_fiducials_detection(
            all_detections, all_scores, all_subpixel_scores, degree_threshold, score_margin
        )

        if quality_control:
            qc.save_process_fiducials_detection_qc(all_detections, processed_detections, qc_detection_dir)
        return processed_detections

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

    def images_restitution(
        self,
        fiducials_detections: dict[str, Fiducials],
        true_fiducials_mm: dict[str, tuple[float, float]],
        scanning_resolution_mm: float = 0.025,
        image_square_dim: int = 10800,
        interpolation_flag: int = cv2.INTER_CUBIC,
        transform_coords: bool = True,
        transform_image: bool = True,
        crop_image: bool = True,
        clahe_enhancement: bool = True,
        quality_control: bool = True,
    ) -> dict[str, dict[str, float]]:
        """
        Performs batch image rectification based on fiducial detection and known reference positions.

        For each image in the batch, this method:
        - Computes an affine transformation to align detected fiducials with known real-world positions.
        - Optionally warps the image using this transformation.
        - Optionally crops the image around the principal point to a fixed square size.
        - Optionally enhances the image contrast using CLAHE.
        - Computes quality control (QC) metrics to evaluate the transformation accuracy.
        - Saves the rectified image and QC visualizations if requested.

        Args:
            fiducials_detections (dict[str, dict[str, tuple[float, float]]]): Mapping of image paths to their detected
                fiducials in pixel coordinates.
            true_fiducials_mm (dict[str, tuple[float, float]]): Ground truth fiducial coordinates in millimeters.
            scanning_resolution_mm (float, optional): Pixel resolution in mm/pixel. Defaults to 0.025.
            image_square_dim (int, optional): Dimension (in pixels) of the final cropped image. Defaults to 10800.
            interpolation_flag (int, optional): Interpolation method used for image warping (e.g., cv2.INTER_CUBIC).
            transform_coords (bool, optional): Whether to compute and apply the coordinate transformation.
            transform_image (bool, optional): Whether to warp the image using the computed transformation.
            crop_image (bool, optional): Whether to crop the image around the transformed principal point.
            clahe_enhancement (bool, optional): Whether to enhance the image contrast using CLAHE.
            quality_control (bool, optional): Whether to compute QC metrics and generate QC plots. Defaults to True.

        Returns:
            dict[str, dict[str, float]]: Dictionary mapping each image path to a dictionary of QC metrics
            (e.g., RMSE between true and transformed fiducial positions).

        Notes:
            - Rectified images are saved to the configured output directory.
            - If `quality_control` is enabled, deviation boxplots are saved to the QC directory.
            - This method is useful for bulk processing of high-resolution scans requiring precise geometric alignment.
        """
        qc_restitution = os.path.join(self.qc_directory, "images_restitution")
        if quality_control:
            os.makedirs(qc_restitution, exist_ok=True)

        metrics = {}
        for image_path, detection in fiducials_detections.items():
            image, metadata = core.image_restitution(
                image_path,
                detection,
                true_fiducials_mm,
                scanning_resolution_mm,
                image_square_dim,
                interpolation_flag,
                transform_coords,
                transform_image,
                crop_image,
                clahe_enhancement,
            )
            metrics[image_path] = qc.compute_metrics_from_image_restitution(metadata)

            if image is not None:
                os.makedirs(self.output_directory, exist_ok=True)
                output_image_path = os.path.join(self.output_directory, os.path.basename(image_path))
                cv2.imwrite(output_image_path, image)

        if quality_control:
            plot_rmse = qc.plot_coordinates_transformations(metrics)
            plot_rmse.savefig(os.path.join(qc_restitution, "deviation_boxplot.png"))
        return metrics
