# hipp

[![PyPI - Version](https://img.shields.io/pypi/v/hipp.svg)](https://pypi.org/project/hipp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hipp.svg)](https://pypi.org/project/hipp)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

HIPP (Historical Image Pre-Processing) is a python library to pre-process scanned historical images for Structure from Motion surface reconstruction and photogrammetric analysis.

-----

## Features

### Preprocessing of Aerial Images

- **Fiducial Marker Detection**
  - Built-in application to generate fiducial marker templates
  - Detection of fiducial marker coordinates using OpenCV template matching
  - Sub-pixel accuracy for fiducial detection
  - Supports detection of 4 midside and/or 4 corner fiducials
  - Replaces low-confidence matches with `None`, based on a matching score threshold
  - Estimates the principal point based on valid fiducials
  - **Quality Control Outputs:**
    - Cropped windows around detected fiducials for visual inspection
    - Distribution plots of principal point deviations and individual fiducial coordinates
    - Matching score distributions
    - RMSE of fiducial coordinates before and after affine transformation

- **Fiducial Marker Proxy Detection** *(feature in development)*

- **Image Restitution**
  - Computes the appropriate geometric transformation between detected and calibrated fiducial positions:
    - 1 point → Translation
    - 2 points → Similarity transformation
    - 3+ points → Affine transformation
  - Crops the image around the estimated principal point to a standard size
  - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance features for SfM (Structure from Motion)
  - Computes the full affine transformation matrix (including crop transformation)

See this [notebook](notebooks/aerial_preprocessing.ipynb) for example.

### Preprocessing of KH-9 Panoramic Camera Satellite Images

- **Image Joining**
  - Joins split images into a single composite image
  - Requires input images named sequentially (e.g. `ImageId_a`, `ImageId_b`, `ImageId_c`, …)
  - A small overlap between image parts is required for proper stitching
  - Uses [`image_mosaic`](https://stereopipeline.readthedocs.io/en/latest/tools/image_mosaic.html) from the [ASP toolkit](https://stereopipeline.readthedocs.io/en/latest/introduction.html)

- **Image Cropping**
  - Built-in interactive tool to manually select corners of the region of interest
  - Rotates and crops the image to align the selected top edge horizontally

See this [notebook](notebooks/kh9pc_preprocessing.ipynb) for example.

### Preprocessing of KH-9 Mapping Camera Satellite Images *(feature in development)*

-----

## Installation

This pipeline requires the [ASP toolkit](https://stereopipeline.readthedocs.io/en/latest/introduction.html) to be installed and accessible in your system's PATH.

### Steps to install the ASP toolkit (2025-06-22 daily build)

```bash
# Navigate to your home directory
cd ~

# Download the 2025-06-22 daily build of the ASP toolkit
wget https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/2025-06-22-daily-build/StereoPipeline-3.6.0-alpha-2025-06-22-x86_64-Linux.tar.bz2

# Extract the downloaded archive
tar xvf StereoPipeline-3.6.0-alpha-2025-06-22-x86_64-Linux.tar.bz2 

# (Optional) Remove the archive to free up space
rm -f StereoPipeline-3.6.0-alpha-2025-06-22-x86_64-Linux.tar.bz2

# To permanently add the ASP executable subdirectory to your PATH, add to your shell configuration (e.g., ~/.bashrc), a line similar to:
export PATH="${PATH}":"$HOME/StereoPipeline-3.6.0-alpha-2025-06-22-x86_64-Linux/bin"
```

After completing these steps, verify that the ASP toolkit is correctly installed and accessible by running:

```bash
which stereo
```

### Installing hipp

Once the ASP toolkit is properly installed and available in your PATH, you can install **hipp** using pip:
  
```bash
pip install hipp
```

## License

`hipp` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
