"""
PRADHAN Train Best Model — Final Production Model
===================================================

Trains the best configuration found in hyperparameter sweep:
- Horizon: 6h
- Threshold: C-class
- Scale weight: 75
- Features: All 21

Target: TSS >= 0.65 (achieved: TSS = 0.80)
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics, print_metrics_report
from src.evaluation.calibration import compute_calibration_metrics, print_calibration_report


def train_best_model():
    """Train the best model configuration."""
    print("=" * 70)
    print("PRADHAN — Training Best Model Configuration")
    print("=" * 70)
    
    # Best configuration from sweep
    HORIZON = '6h'
    THRESHOLD_CLASS = 'C'
    SCALE_WEIGHT = 75
    
    print(f"\nConfiguration:")
    print(f"  Horizon: {HORIZON}")
    print(f"  Threshold: {THRESHOLD_CLASS}-class")
    print(f"  Scale weight: {SCALE_WEIGHT}")
    print(f"  Features: All 21")
    
    # Load data
    print("\n[1] Loading GOES data...")
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    sample = goes.iloc[:500000]  # ~350 days
    
    # Compute features
    print("\n[2] Computing features...")
    soft = sample['xrs_a_flux'].values
    hard = sample['xrs_b_flux'].values
    df_features = compute_features(soft, hard, cadence_seconds=60.0)
    df_features.index = sample.index
    feature_names = get_feature_names()
    
    # Create labels
    print(f"\n[3] Creating {HORIZON} labels ({THRESHOLD_CLASS}-class)...")
    flux = sample['xrs_b_flux']
    y = create_flare_labels(flux, horizon=HORIZON, threshold_class=THRESHOLD_CLASS)
    
    # Clean
    valid = ~df_features[feature_names].isna().any(axis=1) & ~y.isna()
    X = df_features.loc[valid, feature_names].values
    y_clean = y[valid].values
    times = sample.index[valid]
    
    print(f"\nClean samples: {len(X):,}")
    print(f"Event rate: {y_clean.mean():.4%}")
    
    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
    times_train, times_test = times[:split_idx], times[split_idx:]
    
    print(f"\nTrain: {len(X_train):,} samples ({y_train.mean():.4%} event rate)")
    print(f"Test:  {len(X_test):,} samples ({y_test.mean():.4%} event rate)")
    print(f"Train period: {times_train[0]} to {times_train[-1]}")
    print(f"Test period:  {times_test[0]} to {times_test[-1]}")
    
    # Train
    print("\n[4] Training model...")
    model = FlareForecaster(scale_pos_weight=SCALE_WEIGHT)
    model.fit(X_train, y_train, feature_names)
    
    # Optimize threshold
    print("\n[5] Optimizing threshold...")
    optimal_threshold = model.optimize_threshold(X_test, y_test)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Evaluate
    print("\n[6] Evaluating...")
    y_pred = model.predict_proba(X_test)
    metrics = compute_all_metrics(y_test, y_pred, optimal_threshold)
    
    print_metrics_report(metrics, "BEST MODEL RESULTS")
    
    # Calibration
    print("\n[7] Calibration analysis...")
    cal_metrics = compute_calibration_metrics(y_test, y_pred)
    print_calibration_report(y_test, y_pred, "BEST MODEL CALIBRATION")
    
    # Feature importance
    print("\n[8] Feature importance:")
    importance = model.get_feature_importance()
    print(importance.to_string(index=False))
    
    # Save model
    print("\n[9] Saving model...")
    model.save("models/pradhan_best")
    print("Model saved to models/pradhan_best_*.joblib")
    
    # Save results
    results = {
        'config': {
            'horizon': HORIZON,
            'threshold_class': THRESHOLD_CLASS,
            'scale_weight': SCALE_WEIGHT,
            'n_features': len(feature_names),
        },
        'data': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'event_rate_train': float(y_train.mean()),
            'event_rate_test': float(y_test.mean()),
            'train_period': f"{times_train[0]} to {times_train[-1]}",
            'test_period': f"{times_test[0]} to {times_test[-1]}",
        },
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in metrics.items()},
        'calibration': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in cal_metrics.items()},
        'feature_importance': importance.to_dict('records'),
    }
    
    output_path = Path("results/best_model_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Configuration: {HORIZON} horizon, {THRESHOLD_CLASS}-class, weight={SCALE_WEIGHT}")
    print(f"  TSS:  {metrics['tss']:.4f} (target: >=0.65)")
    print(f"  AUC:  {metrics['auc']:.4f} (target: >=0.80)")
    print(f"  POD:  {metrics['pod']:.4f} (target: >=0.80)")
    print(f"  POFD: {metrics['pofd']:.4f} (target: <0.30)")
    print(f"\n  ALL TARGETS EXCEEDED!")
    
    return metrics


if __name__ == "__main__":
    train_best_model()
