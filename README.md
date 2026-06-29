# Wolf Habitat Suitability Prediction — Washington State

A Python reimplementation of a GIS analysis originally performed in ArcGIS Pro, mapping wolf pack habitats, identifying threats, and scoring habitat suitability across Washington State using open public data.

---

## Project Overview

This project answers three spatial questions:

1. **Where are wolf packs currently located in Washington State?**
2. **What threats do these habitats face?** (roads, deer hunting, livestock, land ownership)
3. **How suitable is each area for wolf habitation?** (weighted overlay + machine learning)

---

## Methods

### Map 1 — Wolf Pack Habitat
Wolf pack centroids are sourced from the [WDFW Open Data portal](https://geodataservices.wdfw.wa.gov/). A 5-mile (8 047 m) buffer is drawn around each pack to define the active habitat zone. Analysis shows packs predominantly occupy forested northeastern Washington.

### Map 2 — Threat Analysis
Potential threats within the buffered zones are identified and overlaid:
- **Roads** — road accidents and habitat fragmentation (WA DOT)
- **Deer hunting areas** — legal hunting near wolf territories (WDFW Game Management Units)
- **Livestock farms** — operations with >600 000 head (WSDA / USDA NASS proxy)
- **Land ownership** — federal, state, private, and tribal lands (BLM Surface Management Agency)

### Map 3 — Habitat Suitability Scoring
Each factor is reclassified on a 1–10 scale and combined via weighted overlay:

| Factor | Weight | Reclassification logic |
|--------|--------|------------------------|
| Elevation | 30% | Score 10 for <8 000 ft (2 438 m); decreases linearly above |
| Distance from roads | 25% | Score increases with distance (farther = safer) |
| Distance from deer hunting areas | 25% | Score increases with distance |
| Land ownership | 20% | Federal = 9, State = 7, Private = 4, Tribal = 3 |

### Machine Learning — Logistic Regression
A logistic regression model predicts binary suitability (`score ≥ 6` = suitable) from the four raw input features. Evaluated with **5-fold stratified cross-validation** reporting accuracy and ROC-AUC.

---

## Data Sources

| Layer | Source | Access |
|-------|--------|--------|
| Wolf pack locations | WDFW Open Data | ArcGIS REST API (auto-downloaded) |
| Roads | WA State DOT | ArcGIS REST API (auto-downloaded) |
| Deer hunting (GMUs) | WDFW | ArcGIS REST API (auto-downloaded) |
| Livestock (>600 k head) | USDA NASS 2022 Census | Compiled county centroids |
| Land ownership | BLM Surface Management Agency | ArcGIS REST API (auto-downloaded) |
| Elevation (DEM) | USGS 3DEP via `py3dep` | Auto-downloaded (90 m SRTM fallback) |
| WA State boundary | US Census (PublicaMundi) | Auto-downloaded |

All layers are automatically downloaded on first run and cached in `data/raw/`. If any API is unavailable, the pipeline falls back to realistic synthetic data so it always runs end-to-end.

---

## Project Structure

```
wolf-habitat-wa/
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py       # Download & cache all spatial datasets
│   ├── preprocessing.py     # Reproject, clip, rasterize, distance transforms
│   ├── analysis.py          # Buffer zones, reclassification, weighted overlay
│   ├── ml_model.py          # Logistic regression training & evaluation
│   └── visualization.py     # Static Matplotlib maps + interactive Folium map
├── data/
│   ├── raw/                 # Downloaded source data (auto-created)
│   └── processed/           # Intermediate rasters (auto-created)
└── output/
    ├── maps/                # PNG maps + interactive HTML
    │   ├── map1_wolf_habitat.png
    │   ├── map2_threats.png
    │   ├── map3_suitability.png
    │   └── interactive_map.html
    ├── figures/
    │   ├── feature_importance.png
    │   └── roc_curve.png
    └── wolf_habitat_model.pkl
```

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/susan4syw/Wolf-Habitat-Suitability-Prediction.git
cd Wolf-Habitat-Suitability-Prediction

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

All maps and figures are written to `output/`. The interactive map (`output/maps/interactive_map.html`) can be opened directly in any browser.

---

## Results

| Metric | 5-Fold CV Mean ± Std |
|--------|----------------------|
| Accuracy | 0.9700 ± 0.0014 |
| ROC-AUC  | 0.9945 ± 0.0004 |

Feature coefficients show that **distance from roads** and **elevation** are the strongest predictors of suitability, consistent with wolves preferring remote, lower-elevation forested terrain.

---

## Dependencies

Key libraries: `geopandas`, `rasterio`, `shapely`, `scipy`, `scikit-learn`, `matplotlib`, `folium`, `contextily`, `py3dep`, `pygris`.

See [requirements.txt](requirements.txt) for pinned versions.

---

## License

MIT — see [LICENSE](LICENSE).

*Original GIS analysis performed in ArcGIS Pro for ESRM 250 (University of Washington). This Python reimplementation uses only open-source tools and publicly available data.*
