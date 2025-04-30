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
from hipp.aerial.core import create_fiducial_template_from_image, find_fiducials
from hipp.image import resize_img

CORNER_FIDUCIAL_NAME = "corner_fiducial.png"
MIDSIDE_FIDUCIAL_NAME = "midside_fiducial.png"
SUBPIXEL_CORNER_FIDUCIAL_NAME = "subpixel_" + CORNER_FIDUCIAL_NAME
SUBPIXEL_MIDSIDE_FIDUCIAL_NAME = "subpixel_" + MIDSIDE_FIDUCIAL_NAME


class AerialPreprocessing:
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
        corner: bool = False,
        midside: bool = False,
        fiducial_coordinate: tuple[int, int] | None = None,
        overwrite: bool = False,
    ) -> None:
        if (not corner and not midside) or (corner and midside):
            raise ValueError("Need either corner of midside")

        fiducial_name = MIDSIDE_FIDUCIAL_NAME if midside else CORNER_FIDUCIAL_NAME
        fiducial_path = os.path.join(self.fiducials_directory, fiducial_name)

        if not os.path.exists(fiducial_path) or overwrite:
            img = cv2.imread(self.first_images, cv2.IMREAD_GRAYSCALE)

            fiducial = create_fiducial_template_from_image(img, fiducial_coordinate, distance_around_fiducial)
            cv2.imwrite(fiducial_path, fiducial)
        else:
            fiducial = cv2.imread(fiducial_path, cv2.IMREAD_GRAYSCALE)

        subpixel_fiducial_name = SUBPIXEL_MIDSIDE_FIDUCIAL_NAME if midside else SUBPIXEL_CORNER_FIDUCIAL_NAME
        subpixel_fiducial_path = os.path.join(self.fiducials_directory, subpixel_fiducial_name)

        if not os.path.exists(subpixel_fiducial_path) or overwrite:
            fiducial = resize_img(fiducial)
            subpixel_fiducial = create_fiducial_template_from_image(fiducial, None, distance_around_fiducial)
            cv2.imwrite(subpixel_fiducial_path, subpixel_fiducial)

    def detect_fiducials(
        self, subpixel_factor: float = 8, grid_size: int = 3, quality_control: bool = True, progress_bar: bool = True
    ) -> list[dict[str, dict[str, object]]]:
        corner_fiducial_path = os.path.join(self.fiducials_directory, CORNER_FIDUCIAL_NAME)
        midside_fiducial_path = os.path.join(self.fiducials_directory, MIDSIDE_FIDUCIAL_NAME)
        subpixel_corner_fiducial_path = os.path.join(self.fiducials_directory, SUBPIXEL_CORNER_FIDUCIAL_NAME)
        subpixel_midside_fiducial_path = os.path.join(self.fiducials_directory, SUBPIXEL_MIDSIDE_FIDUCIAL_NAME)

        corner_fiducial = cv2.imread(corner_fiducial_path, cv2.IMREAD_GRAYSCALE)
        midside_fiducial = cv2.imread(midside_fiducial_path, cv2.IMREAD_GRAYSCALE)
        subpixel_corner_fiducial = cv2.imread(subpixel_corner_fiducial_path, cv2.IMREAD_GRAYSCALE)
        subpixel_midside_fiducial = cv2.imread(subpixel_midside_fiducial_path, cv2.IMREAD_GRAYSCALE)

        results = []

        tif_files = [f for f in os.listdir(self.images_directory) if f.endswith(".tif")]
        iterator = tqdm(tif_files, desc="Fiducials detection") if progress_bar else tif_files

        for filename in iterator:
            image = cv2.imread(os.path.join(self.images_directory, filename), cv2.IMREAD_GRAYSCALE)
            fiducials_res = find_fiducials(
                image=image,
                corner_fiducial=corner_fiducial,
                midside_fiducial=midside_fiducial,
                subpixel_corner_fiducial=subpixel_corner_fiducial,
                subpixel_midside_fiducial=subpixel_midside_fiducial,
                subpixel_factor=subpixel_factor,
                grid_size=grid_size,
            )

            results.append(fiducials_res)

            if quality_control:
                fig = qc.find_fiducials_quality_control(fiducials_res, image)
                fig_path = os.path.join(self.qc_directory, filename.replace(".tif", ".png"))
                fig.savefig(fig_path)

        return results
