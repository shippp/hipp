"""
Module: preprocessing.py
Author: godinlu
Date: 29
Description: Contain the AerialPreprocessing class
"""

import glob
import os

import cv2
from tqdm import tqdm

import hipp.aerial.quality_control as qc
from hipp.aerial.core import create_fiducial_template_from_image, detect_fiducials
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
        self, subpixel_factor: float = 8, grid_size: int = 3, quality_control: bool = True, progress_bar: bool = True
    ) -> list[dict[str, dict[str, object]]]:
        """
        Detects fiducial markers in a batch of grayscale `.tif` images located in the input directory.

        This method loads previously generated corner and midside fiducial templates (both standard and subpixel),
        then applies them to each image in the dataset using template matching. Subpixel refinement is applied
        if the subpixel templates are available. Optionally, quality control figures are saved for each image to
        visually validate the detection accuracy.

        Args:
            subpixel_factor: Scaling factor used for subpixel precision. A higher value provides finer matching
                             but increases computation time.
            grid_size: Size of the grid into which the image is split to localize the fiducials.
                       Must be an odd integer (e.g., 3 for a 3x3 grid).
            quality_control: If True, generates and saves diagnostic plots showing detected fiducial positions
                             and subpixel improvement metrics.
            progress_bar: If True, shows a progress bar while processing the images using `tqdm`.

        Returns:
            A list of dictionaries, each corresponding to an image, containing the detection results for all
            fiducials found. Each fiducial entry includes approximate and possibly subpixel-refined positions,
            along with matching confidence scores.

        Raises:
            FileNotFoundError: If any required fiducial template is missing from disk.

        Notes:
            - Templates must be generated in advance using `create_fiducial_template`.
            - This method assumes grayscale `.tif` images and saves quality control figures in PNG format.
        """
        fiducials_template = self.load_fiducials_template()

        results = []

        # List all `.tif` images in the directory
        tif_files = [f for f in os.listdir(self.images_directory) if f.endswith(".tif")]
        iterator = tqdm(tif_files, desc="Fiducials detection") if progress_bar else tif_files

        for filename in iterator:
            image = cv2.imread(os.path.join(self.images_directory, filename), cv2.IMREAD_GRAYSCALE)

            # Run fiducial detection for the current image
            fiducials_res = detect_fiducials(
                image=image,
                **fiducials_template,  # give all fiducials templates in the dict
                subpixel_factor=subpixel_factor,
                grid_size=grid_size,
            )

            results.append(fiducials_res)

            # Generate and save quality control figure
            if quality_control:
                fig = qc.find_fiducials_quality_control(fiducials_res, image)
                fig_path = os.path.join(self.qc_directory, filename.replace(".tif", ".png"))
                fig.savefig(fig_path)

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
