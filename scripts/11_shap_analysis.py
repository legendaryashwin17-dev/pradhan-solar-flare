"""
PRADHAN Script 11 — Feature Importance Analysis
=================================================
Uses XGBoost's built-in gain importance (equivalent to SHAP for trees).
Much faster than full SHAP computation.

Usage:
    python scripts/11_shap_analysis.py

Output:
    results/shap_importance.json
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

RESULTS_DIR = Path("results")


def main():
    print("=" * 60)
    print("PRADHAN Feature Importance Analysis")
    print("=" * 60)

    # Load model and config
    model_path = Path("models/pradhan_best_model.joblib")
    config_path = Path("models/pradhan_best_config.joblib")

    if not model_path.exists():
        print(f"ERROR: No model found at {model_path}")
        return

    print("\nLoading model...")
    model = joblib.load(model_path)
    saved_config = joblib.load(config_path) if config_path.exists() else {}
    feature_cols = saved_config.get("feature_names", [f"f{i}" for i in range(model.n_features_in_)])
    threshold = saved_config.get("threshold", 0.5)

    print(f"  Model: {type(model).__name__}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Threshold: {threshold:.4f}")

    # XGBoost gain importance (mean gain across all splits)
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_cols, importances.tolist()))

    # Sort by importance
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importance (XGBoost Gain) ---")
    max_imp = sorted_imp[0][1] if sorted_imp else 1
    for feat, imp in sorted_imp:
        bar = "#" * int(imp / max_imp * 40) if max_imp > 0 else ""
        print(f"  {feat:25s}  {imp:.4f}  {bar}")

    # Save
    results_path = RESULTS_DIR / "shap_importance.json"
    with open(results_path, "w") as f:
        json.dump({
            "method": "xgboost_gain_importance",
            "feature_importance": {k: round(v, 6) for k, v in sorted_imp},
            "threshold": float(threshold),
            "model_type": type(model).__name__,
        }, f, indent=2)
    print(f"\nSaved to {results_path}")

    # Summary stats
    total = sum(importances)
    print(f"\n--- Summary ---")
    print(f"  Top 5 features account for: {sum(v for _, v in sorted_imp[:5])/total*100:.1f}% of total importance")
    print(f"  Zero-importance features: {sum(1 for v in importances if v < 0.001)}")
    print(f"  Total importance: {total:.4f}")


if __name__ == "__main__":
    main()
