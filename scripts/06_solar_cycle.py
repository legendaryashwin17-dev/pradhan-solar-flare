"""
PRADHAN Script 06 — Solar Cycle Generalization
================================================

Tests whether the model generalizes across different phases of the solar cycle.
Trains on one period, tests on another.

Key question: Does a model trained on Solar Cycle 24 (declining phase)
work on Cycle 25 (rising phase)?

Usage:
    python scripts/06_solar_cycle.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics, print_metrics_report


def run_solar_cycle_analysis():
    """Run solar cycle generalization experiments."""
    print("=" * 70)
    print("PRADHAN — Solar Cycle Generalization Analysis")
    print("=" * 70)

    # Load data
    print("\n[1] Loading GOES data...")
    goes = load_goes_parquet(r"C:\Users\Admin\aditya-flare-forecast\data\goes_historical")
    print(f"    {len(goes):,} records, {goes.index.min().year}-{goes.index.max().year}")

    # Compute features
    print("\n[2] Computing features...")
    soft = goes['xrs_a_flux'].values
    hard = goes['xrs_b_flux'].values
    df_features = compute_features(soft, hard, cadence_seconds=60.0)
    df_features.index = goes.index

    feature_names = get_feature_names()

    # Create labels
    flux = goes['xrs_b_flux']
    y = create_flare_labels(flux, horizon='24h', threshold_class='M')

    # Valid data
    valid = ~(df_features[feature_names].isna().any(axis=1) | y.isna())
    X_all = df_features.loc[valid, feature_names].values
    y_all = y[valid].values
    times_all = df_features.loc[valid].index

    # Define experiments
    experiments = [
        {
            'name': 'Cycle 24 declining (train) → Rising (test)',
            'train': ('2014-01-01', '2016-12-31'),
            'test': ('2011-01-01', '2013-12-31'),
        },
        {
            'name': 'Early years (train) → Later years (test)',
            'train': ('2003-01-01', '2008-12-31'),
            'test': ('2011-01-01', '2016-12-31'),
        },
        {
            'name': 'Solar max (train) → Solar min (test)',
            'train': ('2003-01-01', '2005-12-31'),
            'test': ('2008-01-01', '2010-12-31'),
        },
        {
            'name': 'First half (train) → Second half (test)',
            'train': ('2003-01-01', '2009-06-30'),
            'test': ('2009-07-01', '2016-12-31'),
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp['name']}")
        print(f"{'=' * 70}")

        train_start, train_end = exp['train']
        test_start, test_end = exp['test']

        # Split
        train_mask = (times_all >= train_start) & (times_all <= train_end)
        test_mask = (times_all >= test_start) & (times_all <= test_end)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        if len(X_train) < 100 or len(X_test) < 100:
            print(f"  Skipping: insufficient data")
            continue

        if y_train.sum() < 5 or y_test.sum() < 5:
            print(f"  Skipping: insufficient events")
            continue

        print(f"  Train: {train_start} to {train_end} ({len(X_train):,} samples, {y_train.mean():.4%} event rate)")
        print(f"  Test:  {test_start} to {test_end} ({len(X_test):,} samples, {y_test.mean():.4%} event rate)")

        # Train
        model = FlareForecaster()
        model.fit(X_train, y_train, feature_names)

        # Evaluate
        y_pred = model.predict_proba(X_test)
        model.optimize_threshold(X_test, y_test)
        metrics = compute_all_metrics(y_test, y_pred, model.threshold)

        print(f"  TSS: {metrics['tss']:.4f}, HSS: {metrics['hss']:.4f}, AUC: {metrics['auc']:.4f}")

        results.append({
            'experiment': exp['name'],
            'train_period': f"{train_start} to {train_end}",
            'test_period': f"{test_start} to {test_end}",
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_event_rate': float(y_train.mean()),
            'test_event_rate': float(y_test.mean()),
            **{k: float(v) if isinstance(v, (np.floating, float)) else v
               for k, v in metrics.items()},
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("SOLAR CYCLE GENERALIZATION SUMMARY")
    print(f"{'=' * 70}")

    if results:
        df_results = pd.DataFrame(results).set_index('experiment')
        print(df_results[['tss', 'hss', 'auc', 'brier']].round(4).to_string())

        # Interpretation
        mean_tss = df_results['tss'].mean()
        print(f"\nMean TSS across experiments: {mean_tss:.4f}")
        if mean_tss > 0.5:
            print("✓ Model generalizes well across solar cycle phases")
        elif mean_tss > 0:
            print("⚠ Model shows some generalization but with degradation")
        else:
            print("✗ Model fails to generalize across solar cycle phases")

    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/solar_cycle.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to results/solar_cycle.json")


if __name__ == "__main__":
    run_solar_cycle_analysis()
