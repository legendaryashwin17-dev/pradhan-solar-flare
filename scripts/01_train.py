"""
PRADHAN Training Script — XGBoost with Chronological Split
==========================================================

Trains the XGBoost model on GOES data with proper temporal validation.

Usage:
    python scripts/01_train.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels, print_label_summary
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics, print_metrics_report


def main():
    print("=" * 70)
    print("PRADHAN — Flare Forecasting Model Training")
    print("=" * 70)

    # =================================================================
    # STEP 1: Load Data
    # =================================================================
    print("\n[1] Loading GOES X-ray data...")

    goes_path = str(DATA_DIR / "goes")

    try:
        goes = load_goes_parquet(goes_path)
        print(f"    Loaded {len(goes):,} records from historical data")
    except FileNotFoundError:
        print(f"    ERROR: GOES data not found at {goes_path}")
        print("    Please download GOES data first.")
        return

    print(f"    Time range: {goes.index.min()} to {goes.index.max()}")
    print(f"    Columns: {goes.columns.tolist()}")

    # =================================================================
    # STEP 2: Print Label Summary
    # =================================================================
    print("\n[2] Analyzing flare statistics...")
    print_label_summary(goes['xrs_b_flux'])

    # =================================================================
    # STEP 3: Compute Features
    # =================================================================
    print("\n[3] Computing 19 statistical proxy features...")

    soft = goes['xrs_a_flux'].values
    hard = goes['xrs_b_flux'].values

    df_features = compute_features(soft, hard, cadence_seconds=60.0)
    feature_names = get_feature_names()

    # Align index
    df_features.index = goes.index

    print(f"    Computed {len(feature_names)} features:")
    for name in feature_names:
        print(f"      - {name}")

    # =================================================================
    # STEP 4: Create Labels
    # =================================================================
    print("\n[4] Creating flare labels...")

    flux = goes['xrs_b_flux']
    y = create_flare_labels(flux, horizon='24h', threshold_class='M')

    print(f"    Horizon: 24 hours")
    print(f"    Threshold: M-class (>=10^-5 W/m^2)")
    print(f"    Event rate: {y.mean():.4%}")
    print(f"    Total events: {int(y.sum()):,}")

    # =================================================================
    # STEP 5: Prepare Data
    # =================================================================
    print("\n[5] Preparing training data...")

    # Remove NaN/invalid rows
    valid = ~(df_features[feature_names].isna().any(axis=1) | y.isna())
    X_all = df_features.loc[valid, feature_names].values
    y_all = y[valid].values
    times_all = df_features.loc[valid].index

    print(f"    Valid samples: {len(X_all):,}")
    print(f"    Positive samples: {int(y_all.sum()):,} ({y_all.mean():.4%})")

    # =================================================================
    # STEP 6: Chronological Split
    # =================================================================
    print("\n[6] Creating chronological validation split...")
    print("    CRITICAL: Training on earlier data, testing on later data")

    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    times_train, times_test = times_all[:split_idx], times_all[split_idx:]

    print(f"\n    Training set: {len(X_train):,} samples")
    print(f"    Training period: {times_train[0]} to {times_train[-1]}")
    print(f"    Training event rate: {y_train.mean():.4%}")
    print(f"\n    Test set: {len(X_test):,} samples")
    print(f"    Test period: {times_test[0]} to {times_test[-1]}")
    print(f"    Test event rate: {y_test.mean():.4%}")

    # =================================================================
    # STEP 7: Train Model
    # =================================================================
    print("\n[7] Training XGBoost model...")

    # Compute scale_pos_weight for class imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    print(f"    Class imbalance ratio: {spw:.1f}x")

    model = FlareForecaster(scale_pos_weight=spw)
    model.fit(X_train, y_train, feature_names)

    print("    Model trained successfully")

    # =================================================================
    # STEP 8: Evaluate
    # =================================================================
    print("\n[8] Evaluating model on test set...")

    y_pred = model.predict_proba(X_test)
    optimal_threshold = model.optimize_threshold(X_test, y_test)

    print(f"    Optimal decision threshold: {optimal_threshold:.4f}")

    metrics = compute_all_metrics(y_test, y_pred, optimal_threshold)
    print_metrics_report(metrics, "TEST SET RESULTS")

    # =================================================================
    # STEP 9: Ablation (Raw Flux vs All Features)
    # =================================================================
    print("\n[9] Ablation experiment: raw flux vs all features...")

    raw_idx = [feature_names.index('soft'), feature_names.index('hard')]
    X_train_raw = X_train[:, raw_idx]
    X_test_raw = X_test[:, raw_idx]

    model_raw = FlareForecaster(scale_pos_weight=spw)
    model_raw.fit(X_train_raw, y_train, ['soft', 'hard'])
    y_pred_raw = model_raw.predict_proba(X_test_raw)
    model_raw.optimize_threshold(X_test_raw, y_test)

    metrics_raw = compute_all_metrics(y_test, y_pred_raw, model_raw.threshold)

    print(f"\n    {'Metric':<20} {'Raw Flux':>12} {'All Features':>14} {'Delta':>10}")
    print(f"    {'-'*58}")
    for k in ['tss', 'hss', 'auc', 'brier']:
        delta = metrics[k] - metrics_raw[k]
        print(f"    {k:<20} {metrics_raw[k]:>12.4f} {metrics[k]:>14.4f} {delta:>+10.4f}")

    # =================================================================
    # STEP 10: Feature Importance
    # =================================================================
    print("\n[10] Feature importance analysis...")

    importance = model.get_feature_importance()
    print("\n    Top 10 most important features:")
    for i, row in importance.head(10).iterrows():
        print(f"      {i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")

    # =================================================================
    # STEP 11: Save Model and Results
    # =================================================================
    print("\n[11] Saving model and results...")

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Save model
    model.save("models/pradhan_forecaster")
    print("    Model saved to models/pradhan_forecaster")

    # Save results
    results = {
        'model': 'XGBoost',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()},
        'ablation_raw': {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in metrics_raw.items()},
        'feature_importance': {
            row['feature']: float(row['importance'])
            for _, row in importance.iterrows()
        },
        'data_info': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_event_rate': float(y_train.mean()),
            'test_event_rate': float(y_test.mean()),
            'train_period': f"{times_train[0]} to {times_train[-1]}",
            'test_period': f"{times_test[0]} to {times_test[-1]}",
        }
    }

    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("    Results saved to results/training_results.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model, metrics


if __name__ == "__main__":
    main()
