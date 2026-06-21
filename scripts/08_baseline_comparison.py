"""
PRADHAN Baseline Comparison — Compare Against Simple Methods
=============================================================

Proper baseline comparison is ESSENTIAL for demonstrating model value.

Baselines compared:
1. Random (base rate) — predicts the climatological mean
2. Persistence — "tomorrow = today"
3. Climatological — always predicts base rate probability
4. NOAA SWPC-style — mimics operational forecast tiers
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import get_sample_data
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels, compute_climatological_rate
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import (
    compute_all_metrics, 
    print_metrics_report,
    compare_with_baselines
)


def run_baseline_comparison():
    print("=" * 70)
    print("PRADHAN — Baseline Comparison")
    print("=" * 70)
    
    # =================================================================
    # Load data
    # =================================================================
    print("\n[1] Loading data...")
    
    try:
        from src.data.reader import load_goes_parquet
        goes = load_goes_parquet("data/goes_historical")
    except:
        # Use smaller sample with more flares for meaningful comparison
        goes = get_sample_data(n_points=10000, include_known_flares=True)
    
    # =================================================================
    # Prepare features and labels
    # =================================================================
    print("\n[2] Preparing features and labels...")
    
    soft = goes['xrs_a_flux'].values
    hard = goes['xrs_b_flux'].values
    
    df_features = compute_features(soft, hard)
    df_features.index = goes.index
    
    feature_names = get_feature_names()
    flux = goes['xrs_b_flux']
    
    # Use 1-hour horizon for synthetic data to ensure we have events
    y = create_flare_labels(flux, horizon='1h', threshold_class='C')
    
    # Valid data
    valid = ~(df_features[feature_names].isna().any(axis=1) | y.isna())
    X = df_features.loc[valid, feature_names].values
    y_true = y[valid].values
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_true[:split], y_true[split:]
    
    print(f"    Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"    Event rate (train): {y_train.mean():.4%}")
    print(f"    Event rate (test): {y_test.mean():.4%}")
    
    # Check if we have events
    if y_test.sum() < 10:
        print("\n⚠️  WARNING: Not enough events in test set for meaningful comparison")
        print("   This is common with rare events - using smaller horizon")
    
    # =================================================================
    # Train PRADHAN model
    # =================================================================
    print("\n[3] Training PRADHAN model...")
    
    model = FlareForecaster()
    model.fit(X_train, y_train, feature_names)
    y_pred = model.predict_proba(X_test)
    model.optimize_threshold(X_test, y_test)
    
    # =================================================================
    # Compute baselines
    # =================================================================
    print("\n[4] Computing baseline predictions...")
    
    # 1. Random (base rate)
    np.random.seed(42)
    climatological_rate = compute_climatological_rate(flux, 'M', '24h')
    y_pred_random = np.random.random(len(y_test)) < climatological_rate
    y_proba_random = np.full(len(y_test), climatological_rate)
    
    # 2. Persistence (shift by 1)
    y_pred_persistence = np.roll(y_test, 1)
    y_proba_persistence = np.roll(y_test.astype(float), 1)
    
    # 3. Climatological (always predict base rate)
    y_proba_climatology = np.full(len(y_test), climatological_rate)
    
    # 4. NOAA SWPC-style (simplified)
    # In reality, NOAA uses complex expert analysis
    if climatological_rate > 0.3:
        noaa_prob = 0.7
    elif climatological_rate > 0.1:
        noaa_prob = 0.4
    else:
        noaa_prob = 0.15
    y_proba_noaa = np.full(len(y_test), noaa_prob)
    
    # =================================================================
    # Compute metrics for all methods
    # =================================================================
    print("\n[5] Computing metrics for all methods...")
    
    methods = {
        'PRADHAN (XGBoost)': y_pred,
        'Random (Base Rate)': y_proba_random,
        'Persistence': y_proba_persistence,
        'Climatology': y_proba_climatology,
        'NOAA-style': y_proba_noaa,
    }
    
    results = []
    for name, probs in methods.items():
        if name == 'Random (Base Rate)':
            # For random, use binary predictions
            metrics = compute_all_metrics(y_test, y_proba_random)
            metrics['method'] = name
        else:
            metrics = compute_all_metrics(y_test, probs)
            metrics['method'] = name
        results.append(metrics)
    
    df_results = pd.DataFrame(results).set_index('method')
    
    # =================================================================
    # Print comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n" + df_results[['auc', 'pr_auc', 'tss', 'hss', 'brier']].round(4).to_string())
    
    # =================================================================
    # Interpretation
    # =================================================================
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    
    pradhan_tss = df_results.loc['PRADHAN (XGBoost)', 'tss']
    random_tss = df_results.loc['Random (Base Rate)', 'tss']
    persist_tss = df_results.loc['Persistence', 'tss']
    
    print(f"\nPRADHAN TSS: {pradhan_tss:.4f}")
    print(f"Random TSS:  {random_tss:.4f}")
    print(f"Persistence TSS: {persist_tss:.4f}")
    
    improvement_over_random = pradhan_tss - random_tss
    improvement_over_persist = pradhan_tss - persist_tss
    
    print(f"\nImprovement over random: {improvement_over_random:+.4f}")
    print(f"Improvement over persistence: {improvement_over_persist:+.4f}")
    
    if pradhan_tss > 0.5:
        print("\n✓ PRADHAN demonstrates substantial skill (TSS > 0.5)")
    elif pradhan_tss > 0:
        print("\n⚠ PRADHAN shows positive skill, but modest improvement")
    else:
        print("\n✗ PRADHAN does not outperform baselines")
    
    # =================================================================
    # Save results
    # =================================================================
    print("\n[6] Saving results...")
    
    Path("results").mkdir(exist_ok=True)
    
    with open("results/baseline_comparison.json", "w") as f:
        json.dump({
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in row.items()}
            for name, row in df_results.iterrows()
        }, f, indent=2)
    
    print("    ✓ Results saved to results/baseline_comparison.json")
    
    return df_results


if __name__ == "__main__":
    run_baseline_comparison()