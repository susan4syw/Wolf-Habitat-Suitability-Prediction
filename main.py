#!/usr/bin/env python
"""
main.py — Wolf Habitat Suitability Pipeline for Washington State

Runs all three analyses end-to-end:
  1. Wolf pack habitat mapping
  2. Threat analysis with 5-mile buffers
  3. Habitat suitability scoring + logistic regression model
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src import data_loader as dl
from src import preprocessing as pp
from src import analysis as an
from src import ml_model as ml
from src import visualization as viz

PROCESSED = Path("data/processed")
OUTPUT    = Path("output")
RESOLUTION_M = 1000   # 1 km grid cells


def main():
    print("\n" + "=" * 60)
    print("  WOLF HABITAT SUITABILITY — WASHINGTON STATE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load all data layers
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data layers ...")

    wa_boundary   = dl.load_wa_boundary()
    wolf_packs    = dl.load_wolf_packs()
    roads         = dl.load_roads()
    deer_hunting  = dl.load_deer_hunting_areas()
    livestock     = dl.load_livestock_farms()
    land_ownership = dl.load_land_ownership()
    dem_data, dem_transform, dem_crs = dl.load_elevation_raster(dl.WA_BBOX)

    print(f"  Wolf packs    : {len(wolf_packs):>4} records")
    print(f"  Roads         : {len(roads):>4} records")
    print(f"  Deer hunting  : {len(deer_hunting):>4} records")
    print(f"  Livestock     : {len(livestock):>4} records")
    print(f"  Land ownership: {len(land_ownership):>4} records")

    # ------------------------------------------------------------------
    # 2. Build reference raster grid
    # ------------------------------------------------------------------
    print(f"\n[2/6] Building {RESOLUTION_M} m reference raster grid ...")
    profile, transform = pp.make_reference_grid(wa_boundary, RESOLUTION_M)
    wa_mask = pp.make_wa_mask(wa_boundary, profile)
    print(f"  Grid size: {profile['height']} rows × {profile['width']} cols")

    # ------------------------------------------------------------------
    # 3. Wolf pack buffer + threat overlay (Map 2 prep)
    # ------------------------------------------------------------------
    print("\n[3/6] Computing 5-mile wolf pack buffers & threat overlay ...")
    wolf_buffer = an.buffer_wolf_packs(wolf_packs, radius_m=8047.0)
    threats     = an.clip_threats_to_buffer(
        roads, deer_hunting, livestock, land_ownership, wolf_buffer
    )

    # ------------------------------------------------------------------
    # 4. Rasterize layers → distance rasters → reclassify
    # ------------------------------------------------------------------
    print("\n[4/6] Rasterizing layers and computing distance rasters ...")

    # Elevation
    elev_grid = pp.reproject_dem_to_grid(dem_data, dem_transform, dem_crs, profile)

    # Roads distance
    road_arr   = pp.rasterize_layer(roads, profile, burn_value=1.0)
    dist_roads = pp.distance_raster(road_arr, RESOLUTION_M)

    # Deer hunting distance
    deer_arr   = pp.rasterize_layer(deer_hunting, profile, burn_value=1.0)
    dist_deer  = pp.distance_raster(deer_arr, RESOLUTION_M)

    # Land ownership (encoded)
    land_enc = an.encode_ownership(land_ownership)
    own_arr  = pp.rasterize_layer(land_enc, profile,
                                   burn_value=1.0, field="own_code")

    # Apply nodata mask
    nodata = float(profile["nodata"])
    for arr in [elev_grid, dist_roads, dist_deer, own_arr]:
        arr[wa_mask == 0] = nodata

    # Reclassify on 1–10 scale
    print("  Reclassifying layers (1–10 scale) ...")
    elev_score  = an.reclassify_elevation(elev_grid, nodata)
    road_score  = an.reclassify_distance(dist_roads, max_dist_m=50_000, nodata=nodata)
    deer_score  = an.reclassify_distance(dist_deer,  max_dist_m=40_000, nodata=nodata)
    own_score   = an.reclassify_ownership(own_arr, nodata=nodata)

    # Weighted overlay → suitability (1–10)
    suitability = an.weighted_overlay(
        elev_score, road_score, deer_score, own_score, wa_mask, nodata
    )

    # Save intermediate rasters
    an.save_raster(suitability, profile, PROCESSED / "suitability.tif")
    an.save_raster(elev_score,  profile, PROCESSED / "elev_score.tif")
    an.save_raster(dist_roads,  profile, PROCESSED / "dist_roads.tif")
    an.save_raster(dist_deer,   profile, PROCESSED / "dist_deer.tif")

    # ------------------------------------------------------------------
    # 5. Machine learning: logistic regression
    # ------------------------------------------------------------------
    print("\n[5/6] Training logistic regression model ...")

    X, y = ml.build_feature_matrix(
        elev_grid, dist_roads, dist_deer, own_arr,
        suitability, nodata=nodata, threshold=6.0, max_samples=50_000,
    )
    print(f"  Training samples: {len(y):,}  |  suitable: {y.sum():,} ({y.mean()*100:.1f}%)")

    results = ml.train_and_evaluate(X, y)
    ml.print_results(results)
    ml.save_model(results["model"], OUTPUT / "wolf_habitat_model.pkl")

    ml_prob = ml.predict_suitability_map(
        results["model"], elev_grid, dist_roads, dist_deer,
        own_arr, wa_mask, nodata=nodata,
    )
    an.save_raster(ml_prob, profile, PROCESSED / "ml_probability.tif")

    # ------------------------------------------------------------------
    # 6. Generate maps
    # ------------------------------------------------------------------
    print("\n[6/6] Generating maps ...")

    viz.map1_wolf_habitat(wolf_packs, wa_boundary, wolf_buffer)
    viz.map2_threat_analysis(
        wa_boundary, wolf_packs, wolf_buffer,
        roads, deer_hunting, livestock, land_ownership,
    )
    viz.map3_suitability(suitability, profile, wa_boundary, wolf_packs, ml_prob, nodata)
    viz.plot_feature_importance(results["feature_importance"], results)
    viz.plot_roc_curve(results["model"], X, y)
    viz.make_folium_map(
        wolf_packs, wolf_buffer, roads, deer_hunting, livestock, land_ownership
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Maps     → output/maps/")
    print(f"  Figures  → output/figures/")
    print(f"  Model    → output/wolf_habitat_model.pkl")
    print(f"  Rasters  → data/processed/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
