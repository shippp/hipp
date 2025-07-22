"""
Copyright (c) 2025 HIPP developers
Description: Functions to recreate in python the image_mosaic function from ASP
"""

import os
import re
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_image_mosaicing_qc(
    qc_directory: str, vmax_percentile: int = 97, scale_factor: int = 8, keep: bool = True
) -> None:
    scene_tiles = defaultdict(list)

    # Group image tiles by scene ID (assumed to be the prefix before the first underscore)
    for filename in sorted(os.listdir(qc_directory)):
        match = re.match(r"diff_[a-z]_[a-z]_(.+)\.tif", filename)
        if match:
            base_name = match.group(1)
            scene_tiles[base_name].append(os.path.join(qc_directory, filename))

    for base_name, paths in scene_tiles.items():
        fig, axes = plt.subplots(1, len(paths), figsize=(15, 10))
        axes = axes.flatten()
        for i, path in enumerate(paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(
                img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor), interpolation=cv2.INTER_CUBIC
            ).astype(np.uint8)
            vmax = np.percentile(img_resized, vmax_percentile)
            im = axes[i].imshow(img_resized, cmap="viridis", vmin=0, vmax=vmax)
            axes[i].set_title(f"{chr(ord('a') + i)} - {chr(ord('a') + i + 1)}\nMAE={np.mean(img_resized):.2f}")
            axes[i].axis("off")
            fig.colorbar(im, ax=axes[i])

            if not keep:
                os.remove(path)

        plt.suptitle("Overlapping images absolute differences", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(qc_directory, f"diff_{base_name}.png"))
        plt.show()
