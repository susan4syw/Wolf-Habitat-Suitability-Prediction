"""
Preprocessing module: reproject, clip, rasterize, and compute distance rasters.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box


WA_PROJ = "EPSG:32610"   # UTM Zone 10N – all metric operations in this CRS
WA_CRS  = "EPSG:4326"


# ---------------------------------------------------------------------------
# Reprojection helpers
# ---------------------------------------------------------------------------

def to_projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(WA_PROJ)


def clip_to_wa(gdf: gpd.GeoDataFrame, wa_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    wa_proj = wa_boundary.to_crs(gdf.crs)
    return gpd.clip(gdf, wa_proj)


# ---------------------------------------------------------------------------
# Reference raster grid
# ---------------------------------------------------------------------------

def make_reference_grid(wa_boundary: gpd.GeoDataFrame,
                         resolution_m: int = 1000
                         ) -> tuple[dict, rasterio.transform.Affine]:
    """
    Build a raster grid aligned to the WA bounding box at *resolution_m* metres.
    Returns (profile_dict, transform).
    """
    wa_proj = wa_boundary.to_crs(WA_PROJ)
    minx, miny, maxx, maxy = wa_proj.total_bounds
    # pad slightly
    minx -= resolution_m; miny -= resolution_m
    maxx += resolution_m; maxy += resolution_m

    ncols = int((maxx - minx) / resolution_m)
    nrows = int((maxy - miny) / resolution_m)
    transform = from_bounds(minx, miny, maxx, maxy, ncols, nrows)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": ncols,
        "height": nrows,
        "count": 1,
        "crs": rasterio.CRS.from_epsg(32610),
        "transform": transform,
        "nodata": -9999.0,
    }
    return profile, transform


# ---------------------------------------------------------------------------
# Rasterize vector layers
# ---------------------------------------------------------------------------

def rasterize_layer(gdf: gpd.GeoDataFrame,
                    profile: dict,
                    burn_value: float = 1.0,
                    field: str | None = None) -> np.ndarray:
    """
    Rasterize a GeoDataFrame into a grid described by *profile*.
    Returns a 2-D float32 array (nrows, ncols).
    """
    gdf_proj = gdf.to_crs(profile["crs"].to_epsg())
    shapes = []
    for _, row in gdf_proj.iterrows():
        val = float(row[field]) if (field and field in gdf_proj.columns) else burn_value
        if row.geometry is not None and not row.geometry.is_empty:
            shapes.append((row.geometry.__geo_interface__, val))

    if not shapes:
        return np.zeros((profile["height"], profile["width"]), dtype="float32")

    arr = rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0.0,
        dtype="float32",
    )
    return arr


# ---------------------------------------------------------------------------
# Euclidean distance rasters (metres)
# ---------------------------------------------------------------------------

def distance_raster(presence_arr: np.ndarray,
                    resolution_m: float) -> np.ndarray:
    """
    Compute Euclidean distance (in metres) from each cell to the nearest
    non-zero cell in *presence_arr*.
    """
    # distance_transform_edt returns distances in pixels
    dist_px = distance_transform_edt(presence_arr == 0)
    return (dist_px * resolution_m).astype("float32")


# ---------------------------------------------------------------------------
# Elevation reprojection to reference grid
# ---------------------------------------------------------------------------

def reproject_dem_to_grid(dem_data: np.ndarray,
                           dem_transform,
                           dem_crs,
                           profile: dict) -> np.ndarray:
    """
    Reproject/resample the DEM to match the reference grid defined by *profile*.
    """
    import rasterio.warp

    dst_arr = np.empty((profile["height"], profile["width"]), dtype="float32")
    dst_arr[:] = profile["nodata"]

    rasterio.warp.reproject(
        source=dem_data,
        destination=dst_arr,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=rasterio.enums.Resampling.bilinear,
    )
    return dst_arr


# ---------------------------------------------------------------------------
# WA mask (1 inside state boundary, 0 outside)
# ---------------------------------------------------------------------------

def make_wa_mask(wa_boundary: gpd.GeoDataFrame, profile: dict) -> np.ndarray:
    wa_proj = wa_boundary.to_crs(profile["crs"].to_epsg())
    shapes = [(geom.__geo_interface__, 1) for geom in wa_proj.geometry
              if geom is not None and not geom.is_empty]
    if not shapes:
        return np.ones((profile["height"], profile["width"]), dtype="float32")
    mask = rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="float32",
    )
    return mask
