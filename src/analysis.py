"""
Spatial analysis module:
  - Buffer zones around wolf packs
  - Threat overlay (roads, deer hunting, livestock, land ownership)
  - Habitat suitability reclassification and weighted overlay
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.features import rasterize

WA_PROJ = "EPSG:32610"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Buffer analysis
# ---------------------------------------------------------------------------

def buffer_wolf_packs(wolf_packs: gpd.GeoDataFrame,
                       radius_m: float = 8047.0) -> gpd.GeoDataFrame:
    """
    Create 5-mile (≈8 047 m) buffer zones around wolf pack locations.
    Returns a dissolved GeoDataFrame of the union.
    """
    packs_proj = wolf_packs.to_crs(WA_PROJ)
    buffered = packs_proj.copy()
    buffered["geometry"] = packs_proj.buffer(radius_m)
    dissolved = buffered.dissolve().reset_index(drop=True)
    dissolved["buffer_radius_m"] = radius_m
    return dissolved.to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# Clip threats to buffered zone
# ---------------------------------------------------------------------------

def clip_threats_to_buffer(
    roads: gpd.GeoDataFrame,
    deer_hunting: gpd.GeoDataFrame,
    livestock: gpd.GeoDataFrame,
    land_ownership: gpd.GeoDataFrame,
    wolf_buffer: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Clip each threat layer to the buffered wolf pack zone.
    Returns a dict of clipped GeoDataFrames.
    """
    threats: dict[str, gpd.GeoDataFrame] = {}
    for name, layer in [
        ("roads", roads),
        ("deer_hunting", deer_hunting),
        ("livestock", livestock),
        ("land_ownership", land_ownership),
    ]:
        try:
            clipped = gpd.clip(layer.to_crs(wolf_buffer.crs), wolf_buffer)
            threats[name] = clipped if not clipped.empty else layer.iloc[:0]
        except Exception:
            threats[name] = layer.iloc[:0]
    return threats


# ---------------------------------------------------------------------------
# Reclassification helpers (1–10 scale)
# ---------------------------------------------------------------------------

def reclassify_elevation(dem: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """
    Score 1–10. Elevation < 8 000 ft (2 438 m) → score 10; higher → lower score.
    Linear ramp: 0 m → 10, 2438 m → 10, 3000 m → 5, 4000+ m → 1.
    """
    threshold_ft = 8000
    threshold_m = threshold_ft * 0.3048  # 2438 m

    score = np.where(
        dem <= threshold_m,
        10.0,
        np.clip(10.0 - 9.0 * (dem - threshold_m) / (4000.0 - threshold_m), 1.0, 10.0),
    ).astype("float32")
    score[dem == nodata] = nodata
    return score


def reclassify_distance(dist_m: np.ndarray, max_dist_m: float = 50_000.0,
                         nodata: float = -9999.0) -> np.ndarray:
    """
    Score 1–10: closer to threat → lower score; farther → higher score.
    Linear from 1 (distance=0) to 10 (distance=max_dist_m).
    """
    score = np.clip(1.0 + 9.0 * dist_m / max_dist_m, 1.0, 10.0).astype("float32")
    score[dist_m == nodata] = nodata
    return score


def reclassify_ownership(ownership_arr: np.ndarray,
                          tribal_code: float = 5.0,
                          nodata: float = -9999.0) -> np.ndarray:
    """
    Tribal lands (code 5) → score 3 (higher threat from illegal hunting).
    Federal/NPS (code 1) → score 9. State (code 2) → score 7.
    Private (code 3) → score 4. Other → score 6.
    """
    mapping = {
        0: 6.0,   # unclassified / background
        1: 9.0,   # Federal (USFS, BLM)
        2: 7.0,   # State
        3: 4.0,   # Private
        4: 6.0,   # Other public
        5: 3.0,   # Tribal
    }
    score = np.full_like(ownership_arr, 6.0, dtype="float32")
    for code, val in mapping.items():
        score[ownership_arr == code] = val
    score[ownership_arr == nodata] = nodata
    return score


# ---------------------------------------------------------------------------
# Encode ownership layer to numeric codes
# ---------------------------------------------------------------------------

_OWNERSHIP_CODES = {
    "Federal": 1,
    "USFS": 1,
    "BLM": 1,
    "NPS": 1,
    "State": 2,
    "Private": 3,
    "Other": 4,
    "Tribal": 5,
    "Indian": 5,
    "AmericanIndian": 5,
}

def encode_ownership(land_ownership: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a numeric 'own_code' column based on the ownership type."""
    col = None
    for c in ["ADMIN_AGENCY_CODE", "ownership_type", "OWN_TYPE", "AdmUnit"]:
        if c in land_ownership.columns:
            col = c
            break
    if col is None:
        land_ownership = land_ownership.copy()
        land_ownership["own_code"] = 1
        return land_ownership

    land_ownership = land_ownership.copy()
    land_ownership["own_code"] = land_ownership[col].apply(
        lambda v: next(
            (code for key, code in _OWNERSHIP_CODES.items() if key.lower() in str(v).lower()),
            4,
        )
    )
    return land_ownership


# ---------------------------------------------------------------------------
# Weighted overlay → final suitability map
# ---------------------------------------------------------------------------

# Weights must sum to 1.0
WEIGHTS = {
    "elevation":     0.30,
    "road_dist":     0.25,
    "deer_dist":     0.25,
    "land_ownership": 0.20,
}

def weighted_overlay(
    elev_score: np.ndarray,
    road_dist_score: np.ndarray,
    deer_dist_score: np.ndarray,
    ownership_score: np.ndarray,
    wa_mask: np.ndarray,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    Combine reclassified layers using WEIGHTS into a 1–10 suitability score.
    Cells outside Washington or with nodata in any layer are set to nodata.
    """
    layers = [elev_score, road_dist_score, deer_dist_score, ownership_score]
    bad = np.zeros_like(elev_score, dtype=bool)
    for layer in layers:
        bad |= (layer == nodata)
    bad |= (wa_mask == 0)

    w = WEIGHTS
    suitability = (
        w["elevation"]      * np.where(bad, 0, elev_score)
        + w["road_dist"]    * np.where(bad, 0, road_dist_score)
        + w["deer_dist"]    * np.where(bad, 0, deer_dist_score)
        + w["land_ownership"] * np.where(bad, 0, ownership_score)
    ).astype("float32")
    suitability[bad] = nodata
    return suitability


# ---------------------------------------------------------------------------
# Save raster helper
# ---------------------------------------------------------------------------

def save_raster(arr: np.ndarray, profile: dict, path: Path) -> None:
    p = dict(profile)
    p["count"] = 1
    p["dtype"] = "float32"
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)
