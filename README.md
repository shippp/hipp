# hipp

[![PyPI - Version](https://img.shields.io/pypi/v/hipp.svg)](https://pypi.org/project/hipp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hipp.svg)](https://pypi.org/project/hipp)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
![Static Badge](https://img.shields.io/badge/type%20checked-mypy-039dfc)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

HIPP (Historical Image Pre-Processing) is a python library to pre-process scanned historical (film based) aerial and satellite images in preparation for Structure from Motion surface reconstruction and photogrammetric analysis.

Examples of images that are currently supported or planned to be supported include:

- aerial images with fiducial (or pseudo fiducial) markers, such as from:
  - the [aerial photo single frame dataset](https://doi.org//10.5066/F7610XKM) of [USGS Earth Explorer](https://earthexplorer.usgs.gov/) (EE)
  - the North American Glacier Photography (NAGAP) database from <https://arcticdata.io/>
- declassified US reconnaissance satellite images, in particular:
  - KH-9 Hexagon panoramic camera images, i.e. [Declass 3 dataset](https://doi.org/10.5066/F7WD3Z10) from EE
  - KH-9 Hexagon mapping camera images, i.e. [Declass 2 dataset](https://doi.org/10.5066/F74X5684) from EE (to be included)
  - KH-4/4A/4B Corona images, i.e. [Declass 1 dataset](https://doi.org/10.5066/F78P5XZM) from EE (to be included)

-----

## Gallery of supported pictures

### Aerial color image of Casa Grande, Arizona acquired on 6 Sep 1978 (quickview)

![aerial_casa_grande](https://ims.cr.usgs.gov/browse/aircraft/phoenix/aerial/3DTQ/3DTQ06031/3DTQ06031_006.jpg)

### KH-9 panoramic camera image (B/W) of South Iceland, acquired on 22 Aug 1980 (quickview)

![D3C1216-200533A022](https://github.com/user-attachments/assets/90438e6b-39e2-4fac-840d-aed3a0b4cd61)

<!-- ![KH9_pc](https://ims.cr.usgs.gov/browse/declass3/1216-2/00533/A/D3C1216-200533A023.jpg) -->

### KH-9 mapping camera image (B/W) of South Iceland, acquired on 22 Aug 1980 (quickview)

![KH9_mc](https://ims.cr.usgs.gov/browse/declassii/1216-5/00280/DZB1216-500280L004001-00147.jpg)

## Features

### Data query and download

- Download of imagery is supported through our sister package [usgsxplore](https://github.com/adehecq/usgs_explorer).

### Preprocessing of Aerial Images

- **Detection of fiducial markers**
  - Built-in application to generate fiducial marker templates
  - Detection of fiducial marker coordinates using OpenCV template matching
  - Sub-pixel accuracy for fiducial detection
  - Supports detection of 4 midside and/or 4 corner fiducials
  - Filtering of low-confidence matches
  - Estimates the principal point based on valid fiducials
  - **Quality Control Outputs:**
    - Cropped windows around detected fiducials for visual inspection
    - Distribution plots of principal point deviations and individual fiducial coordinates
    - Matching score distributions
    - RMSE of fiducial coordinates before and after affine transformation

- **Detection of fiducial marker proxies (pseudo-fiducial)** *(feature in development)*

- **Image Restitution**
  - Computes the appropriate geometric transformation between detected and calibrated fiducial positions:
    - 1 point → Translation
    - 2 points → Similarity transformation
    - 3+ points → Affine transformation
  - Crops the image around the estimated principal point to a standard size
  - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance features in the image
  - Computes the full affine transformation matrix (including crop transformation)

See this [notebook](notebooks/aerial_preprocessing.ipynb) for example.

### Preprocessing of KH-9 Panoramic Camera Satellite Images

- **Image Joining**
  - Joins split images into a single composite image
  - Requires input images named sequentially (e.g. `ImageId_a`, `ImageId_b`, `ImageId_c`, …)
  - A small overlap between image parts is required for proper stitching
  - Performs stitching using matched keypoints and successive affine transformations for robust, geometry-aware merging.

- **Image Restitutions**
  - Estimates the top and bottom collimation lines using a second-degree polynomial fit.
  - Determines the vertical boundaries (x1 and x2) of the region of interest (ROI).
  - Computes the geometric transformation needed to crop and restitute the ROI using a Thin Plate Spline (TPS) deformation model.

See this [notebook](notebooks/kh9pc_preprocessing.ipynb) for example of full preprocessing.

See this [notebook](notebooks/kh9pc_collimation_rectification.ipynb) for a detailed example of collimation rectification.

### Preprocessing of KH-9 Mapping Camera Satellite Images *(feature in development)*

-----

## Installation

```bash
pip install hipp
```

## License

`hipp` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

The data you create with `hipp` depend on the input datasets you use.
