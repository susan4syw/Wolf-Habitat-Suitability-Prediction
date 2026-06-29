"""
Visualization module: static Matplotlib maps and an interactive Folium map.

Map 1 – Wolf pack habitat (location + land cover context)
Map 2 – Threat analysis (roads, deer hunting, livestock, land ownership, buffer)
Map 3 – Habitat suitability (weighted overlay + ML probability)
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import geopandas as gpd
import rasterio
from rasterio.plot import show as rio_show

try:
    import contextily as ctx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False

try:
    import folium
    from folium import plugins
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

OUTPUT = Path(__file__).parent.parent / "output"
MAPS   = OUTPUT / "maps"
FIGS   = OUTPUT / "figures"
MAPS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

_WEBMERCATOR = "EPSG:3857"


def _add_basemap(ax, crs):
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron,
                            zoom="auto", alpha=0.5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Map 1: Wolf Pack Habitat
# ---------------------------------------------------------------------------

def map1_wolf_habitat(wolf_packs: gpd.GeoDataFrame,
                       wa_boundary: gpd.GeoDataFrame,
                       wolf_buffer: gpd.GeoDataFrame,
                       save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 9))

    wa_wm  = wa_boundary.to_crs(_WEBMERCATOR)
    wp_wm  = wolf_packs.to_crs(_WEBMERCATOR)
    buf_wm = wolf_buffer.to_crs(_WEBMERCATOR)

    wa_wm.boundary.plot(ax=ax, color="#444444", linewidth=1.2, zorder=3)
    buf_wm.plot(ax=ax, color="#a8dadc", alpha=0.35, zorder=2, label="5-mile buffer")
    wp_wm.plot(ax=ax, color="#e63946", marker="*", markersize=120,
               zorder=5, label="Wolf pack")

    # Label pack names if available
    name_col = next((c for c in ["pack_name", "PACK_NAME", "Name", "name"]
                     if c in wolf_packs.columns), None)
    if name_col:
        for _, row in wp_wm.iterrows():
            ax.annotate(row[name_col],
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color="#1d3557", zorder=6)

    _add_basemap(ax, _WEBMERCATOR)
    ax.set_axis_off()
    ax.set_title("Map 1 – Wolf Pack Habitats in Washington State",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    if save:
        path = MAPS / "map1_wolf_habitat.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# Map 2: Threat Analysis
# ---------------------------------------------------------------------------

def map2_threat_analysis(wa_boundary: gpd.GeoDataFrame,
                          wolf_packs: gpd.GeoDataFrame,
                          wolf_buffer: gpd.GeoDataFrame,
                          roads: gpd.GeoDataFrame,
                          deer_hunting: gpd.GeoDataFrame,
                          livestock: gpd.GeoDataFrame,
                          land_ownership: gpd.GeoDataFrame,
                          save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 9))

    def _wm(g): return g.to_crs(_WEBMERCATOR)

    _wm(wa_boundary).boundary.plot(ax=ax, color="#222", linewidth=1.2, zorder=10)

    # Land ownership background
    if not land_ownership.empty and "ADMIN_AGENCY_CODE" in land_ownership.columns:
        own_wm = _wm(land_ownership)
        colour_map = {
            "Federal": "#cce5cc", "USFS": "#cce5cc", "BLM": "#d4eacc",
            "NPS": "#b8e0b8", "State": "#dbe9f4", "Private": "#f9f4e8",
            "Tribal": "#f4e0c8", "Other": "#eeeeee",
        }
        def _own_color(v):
            for k, c in colour_map.items():
                if k.lower() in str(v).lower():
                    return c
            return "#eeeeee"
        for _, row in own_wm.iterrows():
            clr = _own_color(row["ADMIN_AGENCY_CODE"])
            gpd.GeoDataFrame(geometry=[row.geometry], crs=_WEBMERCATOR).plot(
                ax=ax, color=clr, alpha=0.6, zorder=1)

    # Buffer
    _wm(wolf_buffer).plot(ax=ax, color="none", edgecolor="#457b9d",
                          linewidth=1.8, linestyle="--", zorder=6, label="5-mile buffer")
    # Deer hunting
    if not deer_hunting.empty:
        _wm(deer_hunting).plot(ax=ax, color="#fca311", alpha=0.55, zorder=4,
                                label="Deer hunting areas")
    # Roads
    if not roads.empty:
        _wm(roads).plot(ax=ax, color="#e63946", linewidth=0.5, zorder=7, label="Roads")
    # Livestock
    if not livestock.empty:
        _wm(livestock).plot(ax=ax, color="#6a0572", marker="^", markersize=80,
                             zorder=9, label="High livestock (>600 k head)")
    # Wolf packs
    _wm(wolf_packs).plot(ax=ax, color="#1d3557", marker="*", markersize=100,
                          zorder=8, label="Wolf pack")

    # Legend patches for ownership
    legend_patches = [
        mpatches.Patch(color="#cce5cc", label="Federal land"),
        mpatches.Patch(color="#dbe9f4", label="State land"),
        mpatches.Patch(color="#f9f4e8", label="Private land"),
        mpatches.Patch(color="#f4e0c8", label="Tribal land"),
    ]

    _add_basemap(ax, _WEBMERCATOR)
    ax.set_axis_off()
    ax.set_title("Map 2 – Potential Threats to Wolf Pack Habitats",
                 fontsize=14, fontweight="bold", pad=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches,
              labels=labels + [p.get_label() for p in legend_patches],
              loc="lower left", fontsize=8, framealpha=0.88, ncol=2)

    fig.tight_layout()
    if save:
        path = MAPS / "map2_threats.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# Map 3: Habitat Suitability
# ---------------------------------------------------------------------------

_SUIT_CMAP = LinearSegmentedColormap.from_list(
    "suitability",
    ["#d62828", "#f77f00", "#fcbf49", "#eae2b7", "#aad576", "#52b788", "#1b4332"],
    N=256,
)

def map3_suitability(suitability: np.ndarray,
                      profile: dict,
                      wa_boundary: gpd.GeoDataFrame,
                      wolf_packs: gpd.GeoDataFrame,
                      ml_prob: np.ndarray | None = None,
                      nodata: float = -9999.0,
                      save: bool = True) -> plt.Figure:

    ncols = 2 if ml_prob is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11 * ncols, 9))
    if ncols == 1:
        axes = [axes]

    def _plot_raster(ax, arr, title):
        masked = np.ma.masked_where(arr == nodata, arr)
        vmin, vmax = (1, 10) if title.startswith("Weighted") else (0, 1)
        im = ax.imshow(
            masked,
            extent=_raster_extent(profile),
            origin="upper",
            cmap=_SUIT_CMAP,
            vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        # WA boundary in map CRS (match raster CRS)
        wa_native = wa_boundary.to_crs(profile["crs"].to_epsg())
        wa_native.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=5)
        # Wolf packs
        wp_native = wolf_packs.to_crs(profile["crs"].to_epsg())
        wp_native.plot(ax=ax, color="white", marker="*", markersize=80,
                       zorder=6, label="Wolf pack")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_axis_off()
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Score")

    _plot_raster(axes[0], suitability, "Weighted Overlay Suitability (1–10)")
    if ml_prob is not None:
        _plot_raster(axes[1], ml_prob, "ML Predicted P(Suitable)")

    fig.suptitle("Map 3 – Wolf Pack Habitat Suitability in Washington State",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save:
        path = MAPS / "map3_suitability.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


def _raster_extent(profile: dict):
    """Return (left, right, bottom, top) for imshow extent."""
    t = profile["transform"]
    left   = t.c
    top    = t.f
    right  = left + t.a * profile["width"]
    bottom = top  + t.e * profile["height"]
    return (left, right, bottom, top)


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(importance: dict[str, float],
                             cv_results: dict,
                             save: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Coefficients
    ax = axes[0]
    features = list(importance.keys())
    values   = list(importance.values())
    colors   = ["#e63946" if v < 0 else "#2a9d8f" for v in values]
    ax.barh(features, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-odds coefficient (standardised)")
    ax.set_title("Feature Importance\n(Logistic Regression Coefficients)")

    # CV metrics
    ax2 = axes[1]
    metrics = ["Accuracy", "ROC-AUC"]
    means   = [cv_results["cv_accuracy_mean"], cv_results["cv_roc_auc_mean"]]
    stds    = [cv_results["cv_accuracy_std"],  cv_results["cv_roc_auc_std"]]
    bars = ax2.bar(metrics, means, yerr=stds, capsize=8,
                   color=["#457b9d", "#e9c46a"], edgecolor="white", width=0.4)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title("5-Fold Cross-Validation Performance")
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                 f"{mean:.3f}", ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    if save:
        path = FIGS / "feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(model, X: np.ndarray, y: np.ndarray, save: bool = True) -> plt.Figure:
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name="Logistic Regression")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC Curve – Wolf Habitat Suitability\n(trained on full dataset)")
    fig.tight_layout()
    if save:
        path = FIGS / "roc_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ---------------------------------------------------------------------------
# Interactive Folium map
# ---------------------------------------------------------------------------

def make_folium_map(wolf_packs: gpd.GeoDataFrame,
                    wolf_buffer: gpd.GeoDataFrame,
                    roads: gpd.GeoDataFrame,
                    deer_hunting: gpd.GeoDataFrame,
                    livestock: gpd.GeoDataFrame,
                    land_ownership: gpd.GeoDataFrame,
                    save: bool = True):
    if not _HAS_FOLIUM:
        print("  folium not installed — skipping interactive map.")
        return None

    centre = [47.5, -120.5]
    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB positron")

    def _add_layer(gdf, name, **style):
        if gdf is None or gdf.empty:
            return
        gdf_wgs = gdf.to_crs("EPSG:4326")
        geo_j = gdf_wgs.__geo_interface__
        folium.GeoJson(
            geo_j,
            name=name,
            style_function=lambda _, **kw: kw,
            **style,
        ).add_to(m)

    # Land ownership
    if not land_ownership.empty:
        folium.GeoJson(
            land_ownership.to_crs("EPSG:4326").__geo_interface__,
            name="Land Ownership",
            style_function=lambda feat: {
                "fillColor": "#d4eacc", "color": "#999", "weight": 0.3,
                "fillOpacity": 0.4,
            },
        ).add_to(m)

    # Deer hunting areas
    if not deer_hunting.empty:
        folium.GeoJson(
            deer_hunting.to_crs("EPSG:4326").__geo_interface__,
            name="Deer Hunting Areas",
            style_function=lambda feat: {
                "fillColor": "#fca311", "color": "#fca311",
                "weight": 1, "fillOpacity": 0.4,
            },
        ).add_to(m)

    # 5-mile buffer
    folium.GeoJson(
        wolf_buffer.to_crs("EPSG:4326").__geo_interface__,
        name="5-Mile Buffer",
        style_function=lambda feat: {
            "fillColor": "#a8dadc", "color": "#457b9d",
            "weight": 2, "fillOpacity": 0.2, "dashArray": "6 4",
        },
    ).add_to(m)

    # Roads (subsample to keep file small)
    if not roads.empty:
        sample = roads.sample(min(200, len(roads)), random_state=0)
        folium.GeoJson(
            sample.to_crs("EPSG:4326").__geo_interface__,
            name="Roads (sample)",
            style_function=lambda feat: {"color": "#e63946", "weight": 1},
        ).add_to(m)

    # Livestock
    if not livestock.empty:
        for _, row in livestock.to_crs("EPSG:4326").iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=10, color="#6a0572", fill=True,
                tooltip=f"{row.get('county','farm')} | {row.get('livestock_count','')} head",
            ).add_to(m)

    # Wolf packs
    name_col = next((c for c in ["pack_name", "PACK_NAME", "Name", "name"]
                     if c in wolf_packs.columns), None)
    for _, row in wolf_packs.to_crs("EPSG:4326").iterrows():
        label = row[name_col] if name_col else "Wolf Pack"
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            tooltip=label,
            icon=folium.Icon(color="red", icon="paw", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    if save:
        path = MAPS / "interactive_map.html"
        m.save(str(path))
        print(f"  Saved → {path}")
    return m
