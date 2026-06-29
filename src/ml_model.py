"""
Machine-learning module: logistic regression predicting binary habitat suitability.

Label definition:
  suitable (1): suitability score >= 6.0
  not suitable (0): suitability score < 6.0

Features per pixel:
  elevation_m, dist_roads_m, dist_deer_m, ownership_code
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt
import joblib

OUTPUT = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# Build feature matrix from raster arrays
# ---------------------------------------------------------------------------

def build_feature_matrix(
    elevation: np.ndarray,
    dist_roads: np.ndarray,
    dist_deer: np.ndarray,
    ownership_code: np.ndarray,
    suitability: np.ndarray,
    nodata: float = -9999.0,
    threshold: float = 6.0,
    max_samples: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten rasters into (X, y) arrays, discarding nodata pixels.
    Randomly subsample to *max_samples* to keep training tractable.
    """
    valid = (
        (elevation != nodata)
        & (dist_roads != nodata)
        & (dist_deer != nodata)
        & (ownership_code != nodata)
        & (suitability != nodata)
    )
    idx = np.argwhere(valid).ravel() if valid.ndim == 1 else np.where(valid.ravel())[0]

    elev_flat   = elevation.ravel()[idx]
    roads_flat  = dist_roads.ravel()[idx]
    deer_flat   = dist_deer.ravel()[idx]
    own_flat    = ownership_code.ravel()[idx]
    suit_flat   = suitability.ravel()[idx]

    y = (suit_flat >= threshold).astype(int)
    X = np.column_stack([elev_flat, roads_flat, deer_flat, own_flat])

    if len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(idx), max_samples, replace=False)
        X, y = X[chosen], y[chosen]

    return X, y


# ---------------------------------------------------------------------------
# Train + evaluate with 5-fold cross-validation
# ---------------------------------------------------------------------------

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit a logistic regression pipeline and evaluate with 5-fold CV.
    Returns a dict with metrics and the fitted final model.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
    )

    results = {
        "cv_accuracy_mean":  scores["test_accuracy"].mean(),
        "cv_accuracy_std":   scores["test_accuracy"].std(),
        "cv_roc_auc_mean":   scores["test_roc_auc"].mean(),
        "cv_roc_auc_std":    scores["test_roc_auc"].std(),
        "cv_train_accuracy": scores["train_accuracy"].mean(),
        "cv_train_roc_auc":  scores["train_roc_auc"].mean(),
    }

    # Fit on full dataset for the saved model and ROC curve
    pipe.fit(X, y)
    results["model"] = pipe

    y_pred  = pipe.predict(X)
    y_prob  = pipe.predict_proba(X)[:, 1]
    results["train_accuracy"] = accuracy_score(y, y_pred)
    results["train_roc_auc"]  = roc_auc_score(y, y_prob)
    results["classification_report"] = classification_report(y, y_pred,
                                        target_names=["Not Suitable", "Suitable"])

    # Feature importance (log-odds coefficients after scaling)
    coef = pipe.named_steps["clf"].coef_[0]
    feature_names = ["Elevation (m)", "Dist. Roads (m)", "Dist. Deer Areas (m)",
                     "Ownership Code"]
    results["feature_importance"] = dict(zip(feature_names, coef))

    return results


# ---------------------------------------------------------------------------
# Predict suitability map from fitted model
# ---------------------------------------------------------------------------

def predict_suitability_map(
    model,
    elevation: np.ndarray,
    dist_roads: np.ndarray,
    dist_deer: np.ndarray,
    ownership_code: np.ndarray,
    wa_mask: np.ndarray,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    Predict P(suitable) for every valid pixel; return a (nrows, ncols) array.
    """
    shape = elevation.shape
    valid = (
        (elevation != nodata)
        & (dist_roads != nodata)
        & (dist_deer != nodata)
        & (ownership_code != nodata)
        & (wa_mask == 1)
    )

    X_all = np.column_stack([
        elevation.ravel(),
        dist_roads.ravel(),
        dist_deer.ravel(),
        ownership_code.ravel(),
    ])
    prob_map = np.full(shape[0] * shape[1], nodata, dtype="float32")
    idx = np.where(valid.ravel())[0]
    if len(idx) > 0:
        prob_map[idx] = model.predict_proba(X_all[idx])[:, 1].astype("float32")
    return prob_map.reshape(shape)


# ---------------------------------------------------------------------------
# Save model and print summary
# ---------------------------------------------------------------------------

def save_model(model, path: Path) -> None:
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def print_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  LOGISTIC REGRESSION — 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    print(f"  ROC-AUC   : {results['cv_roc_auc_mean']:.4f} ± {results['cv_roc_auc_std']:.4f}")
    print()
    print("  Feature Coefficients (log-odds, standardised):")
    for feat, coef in results["feature_importance"].items():
        print(f"    {feat:<30s} {coef:+.4f}")
    print()
    print("  Classification Report (train set):")
    print(results["classification_report"])
    print("=" * 60)
