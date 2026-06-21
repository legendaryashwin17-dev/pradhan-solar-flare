"""
PRADHAN Script 11 — SHAP Analysis
===================================
SHAP (SHapley Additive exPlanations) analysis of the best model.
Provides interpretable feature importance and interaction effects.

Usage:
    python scripts/11_shap_analysis.py

Output:
    results/shap_summary.png
    results/shap_importance.json
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names

RESULTS_DIR = Path("results")


def main():
    print("=" * 60)
    print("PRADHAN SHAP Analysis")
    print("=" * 60)

    # Load best model
    model_path = Path("models/pradhan_best_model.joblib")
    config_path = Path("models/pradhan_best_config.joblib")
    if not model_path.exists():
        print(f"ERROR: No model found at {model_path}")
        print("Run scripts/03_train_best.py first.")
        return

    print("\nLoading model...")
    model = joblib.load(model_path)
    saved_config = joblib.load(config_path) if config_path.exists() else {}
    feature_cols = saved_config.get("feature_names", [f"f{i}" for i in range(model.n_features_in_)])
    threshold = saved_config.get("threshold", 0.5)
    config = saved_config

    print(f"  Model type: {type(model).__name__}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Config: {config}")

    # Load GOES data
    print("\nLoading GOES data...")
    goes = load_goes_parquet(str(DATA_DIR / "goes"))
    print(f"  Records: {len(goes):,}")

    # Compute features
    print("\nComputing features...")
    soft = goes["xrs_a_flux"].values
    hard = goes["xrs_b_flux"].values
    features = compute_features(soft, hard, cadence_seconds=60.0)
    features.index = goes.index

    # Create labels
    horizon_hours = config.get("horizon_hours", 1)
    threshold_val = config.get("threshold_val", 1e-6)
    horizon_minutes = horizon_hours * 60

    xrs_b_series = goes["xrs_b_flux"]
    future_max = xrs_b_series.rolling(window=horizon_minutes, min_periods=1).max().shift(-horizon_minutes)
    y = (future_max >= threshold_val).astype(int)

    # Align
    common_idx = features.index.intersection(y.index).intersection(
        features.dropna(subset=feature_cols).index
    )
    X = features.loc[common_idx, feature_cols].values
    y_aligned = y.loc[common_idx].values

    # Subsample for SHAP (too many points = slow)
    n_total = len(X)
    if n_total > 50000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, 50000, replace=False)
        X_sample = X[idx]
        y_sample = y_aligned[idx]
    else:
        X_sample = X
        y_sample = y_aligned

    print(f"  SHAP sample: {len(X_sample):,} rows")

    # Compute SHAP values
    print("\nComputing SHAP values (TreeExplainer)...")
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 (flare)
    else:
        shap_vals = shap_values

    print(f"  SHAP shape: {shap_vals.shape}")

    # Feature importance (mean |SHAP|)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    importance = dict(zip(feature_cols, mean_abs_shap.tolist()))

    # Sort
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importance (mean |SHAP|) ---")
    for feat, imp in sorted_imp:
        bar = "#" * int(imp / sorted_imp[0][1] * 40) if sorted_imp[0][1] > 0 else ""
        print(f"  {feat:25s}  {imp:.4f}  {bar}")

    # Save importance
    results_path = RESULTS_DIR / "shap_importance.json"
    with open(results_path, "w") as f:
        json.dump({
            "feature_importance": {k: round(v, 6) for k, v in sorted_imp},
            "n_samples": len(X_sample),
            "model_config": config,
        }, f, indent=2)
    print(f"\nSaved importance to {results_path}")

    # SHAP summary plot (save as text since no display)
    print("\n--- SHAP Statistics ---")
    print(f"  Mean absolute SHAP (all features): {np.mean(mean_abs_shap):.4f}")
    print(f"  Max SHAP value: {np.max(np.abs(shap_vals)):.4f}")
    print(f"  SHAP variance (all features): {np.var(shap_vals):.4f}")

    # Top 3 interactions (approximate via feature correlations with SHAP)
    print("\n--- Top Feature Interactions ---")
    for i, (feat, _) in enumerate(sorted_imp[:5]):
        feat_idx = feature_cols.index(feat)
        feat_vals = X_sample[:, feat_idx]
        shap_feat = shap_vals[:, feat_idx]
        # Correlation = rough proxy for interaction
        corr = np.corrcoef(feat_vals, shap_feat)[0, 1]
        print(f"  {feat:25s}  self-corr: {corr:.3f}")

    print("\nDone! SHAP analysis complete.")


if __name__ == "__main__":
    main()
