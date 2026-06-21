"""
PRADHAN Script 12 — Cross-Instrument Validation
=================================================
Tests the GOES-trained model on SoLEXS data from Aditya-L1.
Checks if GOES-learned patterns transfer to SoLEXS observations.

Usage:
    python scripts/12_test_on_solexs.py

Output:
    results/solexs_validation.json
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


def load_solexs_data(solexs_parquet: str) -> pd.DataFrame:
    """Load combined SoLEXS parquet."""
    path = Path(solexs_parquet)
    if not path.exists():
        raise FileNotFoundError(f"SoLEXS parquet not found: {solexs_parquet}")
    return pd.read_parquet(path)


def compute_solexs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from SoLEXS data that match GOES feature set.

    SoLEXS SDD2 measures 2-22 keV (roughly equivalent to GOES XRS-B 1-8 Å).
    We compute analogous features using SoLEXS rate as the flux proxy.
    """
    rate = df["rate"].values.astype(float)
    time_idx = df.index

    features = pd.DataFrame(index=time_idx)

    # Basic flux features
    features["soft"] = rate  # SoLEXS rate as soft proxy
    features["hard"] = rate  # Same channel (no separate hard channel)
    features["soft_log"] = np.log1p(rate)
    features["hard_log"] = np.log1p(rate)
    features["hard_soft_ratio"] = np.ones_like(rate)  # 1.0 (same channel)

    # Derivatives (cadence-dependent)
    rate_series = pd.Series(rate, index=time_idx)
    features["dsoft"] = rate_series.diff().fillna(0).values
    features["dhard"] = rate_series.diff().fillna(0).values

    # Rolling means
    for window, label in [(60, "1m"), (300, "5m")]:
        rolled = rate_series.rolling(window=window, min_periods=1).mean()
        features[f"soft_mean_{label}"] = rolled.values
        features[f"hard_mean_{label}"] = rolled.values

    # Rolling std
    for window, label in [(60, "1m"), (300, "5m")]:
        rolled = rate_series.rolling(window=window, min_periods=1).std().fillna(0)
        features[f"soft_std_{label}"] = rolled.values
        features[f"hard_std_{label}"] = rolled.values

    # Cross-correlation (same channel → 1.0)
    features["soft_hard_corr"] = np.ones_like(rate)

    # Cross-correlation (lag-1)
    shifted = rate_series.shift(1).fillna(0)
    features["xcorr"] = np.correlate(rate, shifted.values, mode="full")[:len(rate)]

    # Second derivatives
    features["dhard_soft_ratio"] = np.zeros_like(rate)
    features["ddsoft"] = rate_series.diff().diff().fillna(0).values

    # Spectral features
    features["spectral_hardening"] = np.ones_like(rate)  # hard/soft = 1
    features["neupert_proxy"] = rate.cumsum()  # proxy for accumulated energy

    return features


def main():
    print("=" * 60)
    print("PRADHAN Cross-Instrument Validation")
    print("GOES-trained model -> SoLEXS test")
    print("=" * 60)

    # Load model
    model_path = Path("models/pradhan_best_model.joblib")
    config_path = Path("models/pradhan_best_config.joblib")

    if not model_path.exists():
        print(f"ERROR: No model at {model_path}")
        return

    model = joblib.load(model_path)
    config = joblib.load(config_path) if config_path.exists() else {}
    feature_cols = config.get("feature_names", [f"f{i}" for i in range(model.n_features_in_)])
    threshold = config.get("threshold", 0.5)

    print(f"\nModel: {type(model).__name__}")
    print(f"Features: {len(feature_cols)}")
    print(f"Threshold: {threshold:.4f}")

    # Load SoLEXS data
    solexs_path = DATA_DIR / "pradan_solexs" / "solexs_combined.parquet"
    print(f"\nLoading SoLEXS data from {solexs_path}...")
    solexs = load_solexs_data(str(solexs_path))
    print(f"  Records: {len(solexs):,}")
    print(f"  Time range: {solexs.index.min()} to {solexs.index.max()}")
    print(f"  Rate range: {solexs['rate'].min():.2f} to {solexs['rate'].max():.2f}")

    # Subsample for speed (use last 30 days)
    if len(solexs) > 300000:
        print("  Subsampling to last 300k records for speed...")
        solexs = solexs.iloc[-300000:]

    # Compute features
    print("\nComputing SoLEXS features...")
    features = compute_solexs_features(solexs)
    print(f"  Computed {len(features.columns)} features")

    # Check feature alignment
    missing = set(feature_cols) - set(features.columns)
    extra = set(features.columns) - set(feature_cols)
    if missing:
        print(f"  WARNING: Missing features: {missing}")
        print(f"  Filling missing with 0")
        for m in missing:
            features[m] = 0.0
    if extra:
        print(f"  Extra features (will be ignored): {extra}")

    # Get aligned data
    X = features[feature_cols].values
    print(f"  Input shape: {X.shape}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Predict
    print("\nRunning predictions...")
    proba = model.predict_proba(X)[:, 1]
    predictions = (proba >= threshold).astype(int)

    # Statistics
    n_total = len(proba)
    n_positive = predictions.sum()
    mean_proba = proba.mean()
    max_proba = proba.max()

    print(f"\n--- Prediction Statistics ---")
    print(f"  Total samples: {n_total:,}")
    print(f"  Predicted flares: {n_positive:,} ({n_positive/n_total*100:.1f}%)")
    print(f"  Mean probability: {mean_proba:.4f}")
    print(f"  Max probability: {max_proba:.4f}")
    print(f"  Probability > 0.5: {(proba > 0.5).sum():,}")
    print(f"  Probability > 0.9: {(proba > 0.9).sum():,}")

    # Time series analysis
    print("\n--- Time Series Analysis ---")
    solexs["flare_probability"] = proba
    solexs["predicted_flare"] = predictions

    # Group by date
    daily = solexs.groupby(solexs.index.date).agg({
        "flare_probability": ["mean", "max"],
        "predicted_flare": "sum",
        "rate": "mean",
    })
    daily.columns = ["mean_prob", "max_prob", "n_flare_minutes", "mean_rate"]

    high_prob_days = daily[daily["max_prob"] > 0.5]
    print(f"  Days with max prob > 0.5: {len(high_prob_days)}")
    if len(high_prob_days) > 0:
        print(f"  Top 5 high-probability days:")
        for date, row in high_prob_days.nlargest(5, "max_prob").iterrows():
            print(f"    {date}: max_prob={row['max_prob']:.3f}, mean_rate={row['mean_rate']:.1f}")

    # Save results
    results = {
        "total_samples": n_total,
        "predicted_flares": int(n_positive),
        "prediction_rate": float(n_positive / n_total),
        "mean_probability": float(mean_proba),
        "max_probability": float(max_proba),
        "high_prob_days": len(high_prob_days),
        "time_range": {
            "start": str(solexs.index.min()),
            "end": str(solexs.index.max()),
        },
        "note": "GOES-trained model applied to SoLEXS data. SoLEXS measures 2-22 keV (similar to GOES XRS-B 1-8 Å). Cross-instrument validation tests if GOES-learned patterns transfer.",
    }

    out_path = Path("results/solexs_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n--- Interpretation ---")
    print("  If SoLEXS predictions show similar flare rates as GOES test set,")
    print("  the model generalizes across instruments.")
    print("  Differences may indicate:")
    print("    - Different energy band responses")
    print("    - Different cadence effects (1s vs 1min)")
    print("    - Different noise characteristics")


if __name__ == "__main__":
    main()
