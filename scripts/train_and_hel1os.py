"""Quick test: load GOES data and train model."""
import sys
sys.path.insert(0, '.')

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics, print_metrics_report
import numpy as np
import json
from pathlib import Path

print("=" * 70)
print("PRADHAN — Train GOES Model + Apply to HEL1OS")
print("=" * 70)

# 1. Load GOES data
print("\n[1] Loading GOES data...")
goes = load_goes_parquet(r"C:\Users\Admin\aditya-flare-forecast\data\goes_historical")
print(f"    Loaded {len(goes):,} records")

# 2. Compute features
print("\n[2] Computing features...")
soft = goes['xrs_a_flux'].values
hard = goes['xrs_b_flux'].values
features_df = compute_features(soft, hard, cadence_seconds=60.0)
feature_names = get_feature_names()
features_df.index = goes.index

# 3. Create labels (1h C-class)
print("\n[3] Creating labels (1h C-class)...")
flux = goes['xrs_b_flux']
y = create_flare_labels(flux, horizon='1h', threshold_class='C')
print(f"    Event rate: {y.mean():.4%}")

# 4. Prepare data
valid = ~(features_df[feature_names].isna().any(axis=1) | y.isna())
X_all = features_df.loc[valid, feature_names].values
y_all = y[valid].values
times_all = features_df.loc[valid].index

# 5. Chronological split (80/20)
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
times_train, times_test = times_all[:split_idx], times_all[split_idx:]

print(f"\n[4] Training: {len(X_train):,} samples ({times_train[0]} to {times_train[-1]})")
print(f"    Test: {len(X_test):,} samples ({times_test[0]} to {times_test[-1]})")

# 6. Train
print("\n[5] Training XGBoost...")
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
spw = n_neg / max(n_pos, 1)
print(f"    scale_pos_weight: {spw:.1f}")

model = FlareForecaster(scale_pos_weight=spw)
model.fit(X_train, y_train, feature_names)

# 7. Evaluate
print("\n[6] Evaluating...")
y_pred = model.predict_proba(X_test)
optimal_threshold = model.optimize_threshold(X_test, y_test)
metrics = compute_all_metrics(y_test, y_pred, optimal_threshold)
print_metrics_report(metrics, "GOES MODEL (1h C-class)")

# 8. Save model
print("\n[7] Saving model...")
Path("models").mkdir(exist_ok=True)
model.save("models/pradhan_best")
print("    Saved to models/pradhan_best")

# Save results
results = {
    'config': '1h C-class',
    'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()},
    'optimal_threshold': float(optimal_threshold),
    'n_train': len(X_train),
    'n_test': len(X_test),
}
Path("results").mkdir(exist_ok=True)
with open("results/best_model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("    Results saved")

print("\n" + "=" * 70)
print("GOES MODEL TRAINED SUCCESSFULLY")
print("=" * 70)
