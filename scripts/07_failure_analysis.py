"""
PRADHAN Failure Analysis — Statistical, Not Storytelling
=========================================================

Systematic analysis of model failures:
1. False Positives: what triggers false alarms?
2. False Negatives: what events are missed?
3. Temporal patterns: when do failures occur?
4. Feature distributions: what features mislead the model?

This is STATISTICAL analysis — no narratives about "why".
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster


def analyze_failures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    times: pd.DatetimeIndex = None,
    threshold: float = 0.5
):
    """
    Statistical failure analysis.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted probabilities
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    times : pd.DatetimeIndex, optional
        Timestamps for temporal analysis
    threshold : float
        Classification threshold
    """
    y_bin = (y_pred > threshold).astype(int)
    
    # Masks
    tp_mask = (y_true == 1) & (y_bin == 1)
    tn_mask = (y_true == 0) & (y_bin == 0)
    fp_mask = (y_true == 0) & (y_bin == 1)
    fn_mask = (y_true == 1) & (y_bin == 0)
    
    print("=" * 70)
    print("FAILURE ANALYSIS (Statistical)")
    print("=" * 70)
    
    # 1. Basic counts
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp_mask.sum():>8,}")
    print(f"  True Negatives:  {tn_mask.sum():>8,}")
    print(f"  False Positives: {fp_mask.sum():>8,}")
    print(f"  False Negatives: {fn_mask.sum():>8,}")
    
    # 2. Rates
    total = len(y_true)
    print(f"\nRates:")
    print(f"  False Positive Rate: {fp_mask.sum() / max(1, (fp_mask.sum() + tn_mask.sum())):.4f}")
    print(f"  False Negative Rate: {fn_mask.sum() / max(1, (fn_mask.sum() + tp_mask.sum())):.4f}")
    
    # 3. Feature analysis for false positives
    if fp_mask.sum() > 0 and X.shape[1] > 0:
        fp_features = X[fp_mask]
        all_features = X
        
        print(f"\n--- False Positive Feature Analysis ---")
        print(f"{'Feature':<25} {'FP Mean':>12} {'All Mean':>12} {'Ratio':>8}")
        print("-" * 60)
        
        fp_means = fp_features.mean(axis=0)
        all_means = all_features.mean(axis=0)
        
        # Find features with largest deviation
        ratios = np.abs(fp_means) / (np.abs(all_means) + 1e-10)
        top_features = np.argsort(ratios)[::-1][:10]
        
        for idx in top_features:
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"  {name:<23} {fp_means[idx]:>12.4f} {all_means[idx]:>12.4f} {ratios[idx]:>8.2f}")
    
    # 4. Feature analysis for false negatives
    if fn_mask.sum() > 0 and X.shape[1] > 0:
        fn_features = X[fn_mask]
        all_features = X
        
        print(f"\n--- False Negative Feature Analysis ---")
        print(f"{'Feature':<25} {'FN Mean':>12} {'All Mean':>12} {'Ratio':>8}")
        print("-" * 60)
        
        fn_means = fn_features.mean(axis=0)
        all_means = all_features.mean(axis=0)
        
        ratios = np.abs(fn_means) / (np.abs(all_means) + 1e-10)
        top_features = np.argsort(ratios)[:10]  # Lowest ratios (features that are too low)
        
        for idx in top_features:
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            print(f"  {name:<23} {fn_means[idx]:>12.4f} {all_means[idx]:>12.4f} {ratios[idx]:>8.2f}")
    
    # 5. Temporal analysis
    if times is not None and len(times) > 0:
        print(f"\n--- Temporal Distribution ---")
        
        if fp_mask.sum() > 0:
            fp_times = times[fp_mask]
            fp_hours = fp_times.hour
            print(f"\nFalse Positives by Hour:")
            for h in range(24):
                count = (fp_hours == h).sum()
                if count > 0:
                    bar = '█' * min(40, count // max(1, fp_mask.sum() // 40))
                    print(f"  {h:02d}:00  {count:>6,}  {bar}")
        
        if fn_mask.sum() > 0:
            fn_times = times[fn_mask]
            fn_hours = fn_times.hour
            print(f"\nFalse Negatives by Hour:")
            for h in range(24):
                count = (fn_hours == h).sum()
                if count > 0:
                    bar = '█' * min(40, count // max(1, fn_mask.sum() // 40))
                    print(f"  {h:02d}:00  {count:>6,}  {bar}")
    
    # 6. Prediction confidence distribution
    print(f"\n--- Prediction Confidence ---")
    print(f"\nTrue Positives (correct detections):")
    if tp_mask.sum() > 0:
        tp_proba = y_pred[tp_mask]
        print(f"  Mean: {tp_proba.mean():.4f}, Std: {tp_proba.std():.4f}")
        print(f"  Range: [{tp_proba.min():.4f}, {tp_proba.max():.4f}]")
    
    print(f"\nFalse Positives (false alarms):")
    if fp_mask.sum() > 0:
        fp_proba = y_pred[fp_mask]
        print(f"  Mean: {fp_proba.mean():.4f}, Std: {fp_proba.std():.4f}")
        print(f"  Range: [{fp_proba.min():.4f}, {fp_proba.max():.4f}]")
    
    print(f"\nFalse Negatives (missed events):")
    if fn_mask.sum() > 0:
        fn_proba = y_pred[fn_mask]
        print(f"  Mean: {fn_proba.mean():.4f}, Std: {fn_proba.std():.4f}")
        print(f"  Range: [{fn_proba.min():.4f}, {fn_proba.max():.4f}]")
    
    return {
        'n_tp': int(tp_mask.sum()),
        'n_tn': int(tn_mask.sum()),
        'n_fp': int(fp_mask.sum()),
        'n_fn': int(fn_mask.sum()),
    }


def run_failure_analysis():
    """Run full failure analysis on GOES data."""
    print("=" * 70)
    print("PRADHAN — Failure Analysis Pipeline")
    print("=" * 70)
    
    # Load data
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    sample = goes.iloc[:100000]
    
    # Features
    soft = sample['xrs_a_flux'].values
    hard = sample['xrs_b_flux'].values
    df_features = compute_features(soft, hard, cadence_seconds=60.0)
    df_features.index = sample.index
    feature_names = get_feature_names()
    
    # Labels
    flux = sample['xrs_b_flux']
    y = create_flare_labels(flux, horizon='6h', threshold_class='M')
    
    # Clean
    valid = ~df_features[feature_names].isna().any(axis=1) & ~y.isna()
    X = df_features.loc[valid, feature_names].values
    y_clean = y[valid].values
    times = sample.index[valid]
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
    times_test = times[split_idx:]
    
    # Train
    model = FlareForecaster(scale_pos_weight=50)
    model.fit(X_train, y_train, feature_names)
    
    # Predict
    y_pred = model.predict_proba(X_test)
    model.optimize_threshold(X_test, y_test)
    
    # Analyze
    results = analyze_failures(y_test, y_pred, X_test, feature_names, times_test, model.threshold)
    
    return results


if __name__ == "__main__":
    run_failure_analysis()
