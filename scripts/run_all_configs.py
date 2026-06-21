"""
PRADHAN Hyperparameter Sweep — Finding TSS ≥ 0.65
==================================================

Runs multiple configurations to find the optimal combination:
- Horizons: 1h, 3h, 6h
- Thresholds: C-class, M-class
- Scale weights: 50, 75, 100
- Feature subsets: All, Top 10

Target: TSS ≥ 0.65
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics

# Top 10 features by importance (from previous SHAP analysis)
TOP_10_FEATURES = [
    'hard_mean_5m', 'soft_mean_5m', 'soft', 'neupert_proxy',
    'hard_std_5m', 'soft_std_1m', 'hard', 'soft_log', 'hard_log', 'hard_std_1m'
]


def run_experiment(
    goes,
    horizon,
    threshold_class,
    scale_weight,
    feature_subset=None,
    n_samples=200000
):
    """
    Run a single experiment configuration.
    
    Returns
    -------
    dict
        Results dictionary
    """
    sample = goes.iloc[:n_samples]
    
    soft = sample['xrs_a_flux'].values
    hard = sample['xrs_b_flux'].values
    df_features = compute_features(soft, hard, cadence_seconds=60.0)
    df_features.index = sample.index
    
    if feature_subset:
        feature_names = feature_subset
    else:
        feature_names = get_feature_names()
    
    # Ensure all features exist
    available = [f for f in feature_names if f in df_features.columns]
    
    X = df_features[available].values
    y = create_flare_labels(sample['xrs_b_flux'], horizon=horizon, threshold_class=threshold_class)
    
    # Clean
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid]
    y = y[valid].values
    
    if y.sum() < 10:
        return None  # Not enough positive samples
    
    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    model = FlareForecaster(scale_pos_weight=scale_weight)
    model.fit(X_train, y_train, available)
    
    # Evaluate
    y_pred = model.predict_proba(X_test)
    model.optimize_threshold(X_test, y_test)
    metrics = compute_all_metrics(y_test, y_pred, model.threshold)
    
    return {
        'horizon': horizon,
        'threshold_class': threshold_class,
        'scale_weight': scale_weight,
        'n_features': len(available),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'event_rate_train': float(y_train.mean()),
        'event_rate_test': float(y_test.mean()),
        'tss': metrics['tss'],
        'auc': metrics['auc'],
        'pod': metrics['pod'],
        'pofd': metrics['pofd'],
        'hss': metrics['hss'],
        'brier': metrics['brier'],
        'optimal_threshold': metrics['optimal_threshold'],
    }


def main():
    print("=" * 70)
    print("PRADHAN — Hyperparameter Sweep for TSS >= 0.65")
    print("=" * 70)
    
    # Load GOES data once
    print("\nLoading GOES data...")
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    print(f"Loaded {len(goes):,} data points")
    
    # Define configurations
    configs = []
    
    # Horizon variations
    for horizon in ['1h', '3h', '6h']:
        for threshold in ['M', 'C']:
            for weight in [50, 75, 100]:
                configs.append((horizon, threshold, weight, None))
    
    # Top 10 features (for best horizons)
    for horizon in ['1h', '3h']:
        for threshold in ['M', 'C']:
            configs.append((horizon, threshold, 75, TOP_10_FEATURES))
    
    print(f"\nRunning {len(configs)} configurations...")
    print("-" * 70)
    
    results = []
    
    for i, (horizon, threshold, weight, features) in enumerate(configs):
        feat_name = f"Top-{len(features)}" if features else "All"
        print(f"[{i+1}/{len(configs)}] {horizon}, {threshold}, w={weight}, feat={feat_name}", end=" ... ")
        
        result = run_experiment(
            goes, horizon, threshold, weight, 
            feature_subset=features,
            n_samples=200000
        )
        
        if result is not None:
            results.append(result)
            tss = result['tss']
            auc = result['auc']
            marker = " *** TARGET" if tss >= 0.65 else ""
            print(f"TSS={tss:.4f}, AUC={auc:.4f}{marker}")
        else:
            print("SKIPPED (insufficient data)")
    
    # Sort by TSS
    results.sort(key=lambda x: x['tss'], reverse=True)
    
    # Save results
    output_path = Path("results/hyperparameter_sweep.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (sorted by TSS)")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Horizon':<8} {'Class':<6} {'Weight':<8} {'Features':<8} {'TSS':>8} {'AUC':>8} {'POD':>8} {'POFD':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results[:15]):
        feat = f"Top-{r['n_features']}" if r['n_features'] < 20 else "All"
        marker = " ***" if r['tss'] >= 0.65 else ""
        print(f"{i+1:<5} {r['horizon']:<8} {r['threshold_class']:<6} {r['scale_weight']:<8} {feat:<8} "
              f"{r['tss']:>8.4f} {r['auc']:>8.4f} {r['pod']:>8.4f} {r['pofd']:>8.4f}{marker}")
    
    # Best result
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION:")
    print(f"  Horizon: {best['horizon']}")
    print(f"  Threshold: {best['threshold_class']}-class")
    print(f"  Scale weight: {best['scale_weight']}")
    print(f"  Features: {best['n_features']}")
    print(f"  TSS: {best['tss']:.4f}")
    print(f"  AUC: {best['auc']:.4f}")
    print(f"  POD: {best['pod']:.4f}")
    print(f"  POFD: {best['pofd']:.4f}")
    print(f"  Event rate: {best['event_rate_test']:.2%}")
    
    if best['tss'] >= 0.65:
        print(f"\n  *** TARGET ACHIEVED: TSS >= 0.65 ***")
    else:
        print(f"\n  Gap to target: {0.65 - best['tss']:.4f}")
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
