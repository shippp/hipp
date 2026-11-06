"""
Copyright (c) 2026 HIPP developers
Description: Corner GCPs of a restituted KH-9 PC image from the USGS scene metadata.
"""

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS


def read_corners(entity_id: str, scenes_metadata_file: str | Path) -> list[tuple[float, float]]:
    """USGS footprint corners (lon, lat) of *entity_id*, clockwise from the image top-left corner."""
    gdf = gpd.read_file(scenes_metadata_file)
    gdf.columns = gdf.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    row = gdf.set_index("entity_id").loc[entity_id]

    ul = float(row.get("nw_corner_long_dec", row.get("ul_lon"))), float(row.get("nw_corner_lat_dec", row.get("ul_lat")))
    ur = float(row.get("ne_corner_long_dec", row.get("ur_lon"))), float(row.get("ne_corner_lat_dec", row.get("ur_lat")))
    lr = float(row.get("se_corner_long_dec", row.get("lr_lon"))), float(row.get("se_corner_lat_dec", row.get("lr_lat")))
    ll = float(row.get("sw_corner_long_dec", row.get("ll_lon"))), float(row.get("sw_corner_lat_dec", row.get("ll_lat")))

    # Aft camera frames are rotated 180 degrees w.r.t. forward ones
    return [ul, ur, lr, ll] if row["camera"] == "F" else [lr, ll, ul, ur]


def set_corner_gcps(raster_filepath: str | Path, corners: list[tuple[float, float]]) -> None:
    """Attach *corners* (from ``read_corners``) to the raster corners as GCPs (EPSG:4326), in place."""
    with rasterio.open(raster_filepath, "r+") as dst:
        pixel_corners = [(0, 0), (dst.width, 0), (dst.width, dst.height), (0, dst.height)]
        # GDAL rejects GCPs with z=None
        gcps = [
            GroundControlPoint(row=line, col=col, x=lon, y=lat, z=0.0)
            for (col, line), (lon, lat) in zip(pixel_corners, corners)
        ]
        dst.gcps = (gcps, CRS.from_epsg(4326))
