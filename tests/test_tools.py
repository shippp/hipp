import os
import tempfile

import numpy as np
import rasterio

from hipp.tools import optimize_geotif_file


def test_optimize_geotif_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = os.path.join(tmpdir, "test.tif")

        data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        with rasterio.open(tif_path, "w") as dst:
            dst.write(data, 1)

        # test the tif file is not compressed yet
        with rasterio.open(tif_path) as src:
            assert src.profile.get("compress") is None

        optimize_geotif_file(tif_path)

        with rasterio.open(tif_path) as src:
            profile = src.profile
            assert profile.get("compress", "").lower() == "lzw"
            assert profile.get("tiled") is True
            assert profile.get("blockxsize") == 256
            assert profile.get("blockysize") == 256
            assert profile.get("driver") == "GTiff"
