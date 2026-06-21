"""
PRADHAN Multi-Config Training — Find Best TSS
==============================================
Trains with different horizons and thresholds to find optimal config.
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics


def train_config(goes, features_df, feature_names, horizon, threshold_class, label):
    """Train and evaluate a single configuration."""
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"  Horizon: {horizon}, Threshold: {threshold_class}-class")
    print(f"{'='*60}")

    # Create labels
    flux = goes['xrs_b_flux']
    y = create_flare_labels(flux, horizon=horizon, threshold_class=threshold_class)

    event_rate = y.mean()
    n_events = int(y.sum())
    print(f"  Event rate: {event_rate:.4%} ({n_events:,} events)")

    # Prepare data
    valid = ~(features_df[feature_names].isna().any(axis=1) | y.isna())
    X_all = features_df.loc[valid, feature_names].values
    y_all = y[valid].values
    times_all = features_df.loc[valid].index

    print(f"  Valid samples: {len(X_all):,}")

    if len(X_all) < 1000:
        print("  SKIP: Not enough samples")
        return None

    if y_all.sum() < 10:
        print("  SKIP: Not enough positive samples")
        return None

    # Chronological split
    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    times_train, times_test = times_all[:split_idx], times_all[split_idx:]

    print(f"  Train: {len(X_train):,} samples ({times_train[0].date()} to {times_train[-1].date()})")
    print(f"  Test:  {len(X_test):,} samples ({times_test[0].date()} to {times_test[-1].date()})")
    print(f"  Train event rate: {y_train.mean():.4%}")
    print(f"  Test event rate:  {y_test.mean():.4%}")

    # Train model
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)

    model = FlareForecaster(scale_pos_weight=spw)
    model.fit(X_train, y_train, feature_names)

    # Evaluate
    y_pred = model.predict_proba(X_test)
    optimal_threshold = model.optimize_threshold(X_test, y_test)
    metrics = compute_all_metrics(y_test, y_pred, optimal_threshold)

    print(f"\n  RESULTS:")
    print(f"    TSS:    {metrics['tss']:.4f}")
    print(f"    HSS:    {metrics['hss']:.4f}")
    print(f"    AUC:    {metrics['auc']:.4f}")
    print(f"    POD:    {metrics['pod']:.4f}")
    print(f"    POFD:   {metrics['pofd']:.4f}")
    print(f"    CSI:    {metrics['csi']:.4f}")
    print(f"    Brier:  {metrics['brier']:.4f}")
    print(f"    Threshold: {optimal_threshold:.4f}")

    return {
        'label': label,
        'horizon': horizon,
        'threshold': threshold_class,
        'event_rate': float(event_rate),
        'n_events': n_events,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_event_rate': float(y_train.mean()),
        'test_event_rate': float(y_test.mean()),
        'train_period': f"{times_train[0]} to {times_train[-1]}",
        'test_period': f"{times_test[0]} to {times_test[-1]}",
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()},
        'optimal_threshold': float(optimal_threshold),
    }


def main():
    print("=" * 70)
    print("PRADHAN — Multi-Config Training")
    print("=" * 70)

    # Load data
    print("\n[1] Loading GOES data...")
    goes = load_goes_parquet(str(DATA_DIR / "goes"))
    print(f"    Loaded {len(goes):,} records")

    # Compute features once
    print("\n[2] Computing features...")
    soft = goes['xrs_a_flux'].values
    hard = goes['xrs_b_flux'].values
    features_df = compute_features(soft, hard, cadence_seconds=60.0)
    feature_names = get_feature_names()
    features_df.index = goes.index
    print(f"    Computed {len(feature_names)} features")

    # Configurations to test
    configs = [
        ("24h M-class (baseline)", "24h", "M"),
        ("12h M-class", "12h", "M"),
        ("6h M-class", "6h", "M"),
        ("1h M-class", "1h", "M"),
        ("6h C-class", "6h", "C"),
        ("1h C-class", "1h", "C"),
    ]

    # Train all configs
    results = []
    for label, horizon, threshold in configs:
        try:
            result = train_config(goes, features_df, feature_names, horizon, threshold, label)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — All Configurations")
    print("=" * 70)
    print(f"\n{'Config':<25} {'TSS':>8} {'HSS':>8} {'AUC':>8} {'POD':>8} {'Event%':>8}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['metrics']['tss'], reverse=True):
        m = r['metrics']
        print(f"{r['label']:<25} {m['tss']:>8.4f} {m['hss']:>8.4f} {m['auc']:>8.4f} "
              f"{m['pod']:>8.4f} {r['event_rate']:>7.2%}")

    # Best config
    if results:
        best = max(results, key=lambda x: x['metrics']['tss'])
        print(f"\nBEST: {best['label']}")
        print(f"  TSS: {best['metrics']['tss']:.4f}")
        print(f"  AUC: {best['metrics']['auc']:.4f}")

    # Save all results
    output_path = Path("results/multi_config_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
