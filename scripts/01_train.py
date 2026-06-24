"""
PRADHAN Training — Optimized for Large Datasets
================================================
Processes GOES data year-by-year to handle 1-sec + 1-min mixed cadence.
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GOES_PARQUET_DIR
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels, print_label_summary
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics, print_metrics_report


def load_and_process_year(year_path: Path, feature_names: list) -> tuple:
    """Load one year of GOES data, resample if needed, compute features."""
    df = pd.read_parquet(year_path)
    
    # Rename old-format columns
    rename_map = {}
    if 'xrsa' in df.columns:
        rename_map['xrsa'] = 'xrs_a_flux'
    if 'xrsb' in df.columns:
        rename_map['xrsb'] = 'xrs_b_flux'
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Resample 1-sec data to 5-min for efficiency (skip 1-min, too slow)
    if len(df) > 1_000_000:
        df = df.resample('5min').mean()
        df = df.dropna(subset=['xrs_a_flux', 'xrs_b_flux'])
        cadence = 300.0
    else:
        cadence = 60.0
    
    # Filter valid values
    df = df[(df['xrs_a_flux'] > 0) & (df['xrs_b_flux'] > 0)]
    df = df[np.isfinite(df['xrs_a_flux']) & np.isfinite(df['xrs_b_flux'])]
    
    if len(df) < 100:
        return None, None, None
    
    # Compute features
    soft = df['xrs_a_flux'].values
    hard = df['xrs_b_flux'].values
    features = compute_features(soft, hard, cadence_seconds=cadence)
    features.index = df.index
    
    # Create labels
    labels = create_flare_labels(df['xrs_b_flux'], horizon='24h', threshold_class='M')
    
    # Get valid rows
    valid = ~(features[feature_names].isna().any(axis=1) | labels.isna())
    X = features.loc[valid, feature_names].values
    y = labels[valid].values
    times = features.loc[valid].index
    
    return X, y, times


def main():
    print("=" * 70)
    print("PRADHAN — XGBoost Training (Optimized)")
    print("=" * 70)
    
    feature_names = get_feature_names()
    print(f"Features: {len(feature_names)}")
    
    # Get all parquet files
    parquet_files = sorted(GOES_PARQUET_DIR.glob("goes_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    # Split: first 80% of years for training, last 20% for testing
    n_files = len(parquet_files)
    split_idx = int(n_files * 0.8)
    
    print(f"\nTraining years: {parquet_files[0].stem.split('_')[1]} to {parquet_files[split_idx-1].stem.split('_')[1]}")
    print(f"Testing years:  {parquet_files[split_idx].stem.split('_')[1]} to {parquet_files[-1].stem.split('_')[1]}")
    
    # Process training years
    print("\n[1] Processing training data...")
    train_Xs, train_ys = [], []
    for i, pf in enumerate(parquet_files[:split_idx]):
        year = pf.stem.split('_')[1]
        X, y, times = load_and_process_year(pf, feature_names)
        if X is not None:
            train_Xs.append(X)
            train_ys.append(y)
            print(f"  {year}: {len(X):,} samples, event rate={y.mean():.4%}")
        gc.collect()
    
    X_train = np.concatenate(train_Xs)
    y_train = np.concatenate(train_ys)
    print(f"\n  Total training: {len(X_train):,} samples, event rate={y_train.mean():.4%}")
    
    # Process testing years
    print("\n[2] Processing test data...")
    test_Xs, test_ys = [], []
    for i, pf in enumerate(parquet_files[split_idx:]):
        year = pf.stem.split('_')[1]
        X, y, times = load_and_process_year(pf, feature_names)
        if X is not None:
            test_Xs.append(X)
            test_ys.append(y)
            print(f"  {year}: {len(X):,} samples, event rate={y.mean():.4%}")
        gc.collect()
    
    X_test = np.concatenate(test_Xs)
    y_test = np.concatenate(test_ys)
    print(f"\n  Total test: {len(X_test):,} samples, event rate={y_test.mean():.4%}")
    
    # Compute class weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    print(f"\n  Class imbalance: {spw:.1f}x")
    
    # Train model
    print("\n[3] Training XGBoost...")
    model = FlareForecaster(scale_pos_weight=spw)
    model.fit(X_train, y_train, feature_names)
    print("  Done!")
    
    # Evaluate
    print("\n[4] Evaluating on test set...")
    y_pred = model.predict_proba(X_test)
    optimal_threshold = model.optimize_threshold(X_test, y_test)
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    
    metrics = compute_all_metrics(y_test, y_pred, optimal_threshold)
    print_metrics_report(metrics, "TEST SET RESULTS")
    
    # Ablation: raw flux only
    print("\n[5] Ablation: raw flux only...")
    raw_idx = [feature_names.index('soft'), feature_names.index('hard')]
    X_train_raw = X_train[:, raw_idx]
    X_test_raw = X_test[:, raw_idx]
    
    model_raw = FlareForecaster(scale_pos_weight=spw)
    model_raw.fit(X_train_raw, y_train, ['soft', 'hard'])
    y_pred_raw = model_raw.predict_proba(X_test_raw)
    model_raw.optimize_threshold(X_test_raw, y_test)
    metrics_raw = compute_all_metrics(y_test, y_pred_raw, model_raw.threshold)
    
    print(f"\n  {'Metric':<20} {'Raw Flux':>12} {'All Features':>14} {'Delta':>10}")
    print(f"  {'-'*58}")
    for k in ['tss', 'hss', 'auc', 'brier']:
        delta = metrics[k] - metrics_raw[k]
        print(f"  {k:<20} {metrics_raw[k]:>12.4f} {metrics[k]:>14.4f} {delta:>+10.4f}")
    
    # Feature importance
    print("\n[6] Feature importance...")
    importance = model.get_feature_importance()
    print("\n  Top 10:")
    for i, row in importance.head(10).iterrows():
        print(f"    {i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")
    
    # Save
    print("\n[7] Saving model...")
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    model.save("models/pradhan_forecaster")
    
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
        }
    }
    
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
