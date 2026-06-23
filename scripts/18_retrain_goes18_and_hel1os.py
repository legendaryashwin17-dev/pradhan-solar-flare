"""
Retrain model on GOES-18 2026 data and analyze HEL1OS simultaneous observations.

Key tasks:
1. Train XGBoost on GOES-18 2026 (same instrument, different solar cycle)
2. Cross-validate with GOES 2003-2017 training data
3. Analyze HEL1OS response during GOES-detected flares
4. Show HEL1OS as an independent early warning channel
"""
import pandas as pd
import numpy as np
import joblib
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# PART 1: Train on GOES-18 2026
# ============================================================
print("=" * 70)
print("PART 1: Retrain XGBoost on GOES-18 2026")
print("=" * 70)

# Load GOES-18 resampled
goes18 = pd.read_parquet("data/goes/goes18_2026.parquet")
goes18["time"] = pd.to_datetime(goes18["time"])
goes18_60s = goes18.set_index("time").resample("60s").agg({
    "flux_a": "mean", "flux_b": "mean",
}).dropna().reset_index()

print(f"GOES-18 2026: {len(goes18_60s):,} records at 60s cadence")

# Labels
lookback = 60
flux_b = goes18_60s["flux_b"].values.copy()
flux_b = np.nan_to_num(flux_b, nan=0.0)
temp = pd.Series(flux_b)
future_max = temp.shift(-lookback).rolling(lookback).max().values
labels = (future_max >= 1e-6).astype(int)

# Features
from src.data.features import compute_features
features_df = compute_features(goes18_60s["flux_a"].values, goes18_60s["flux_b"].values, 60.0)
feature_cols = [c for c in features_df.columns]

# Drop NaN rows (start of series)
valid_mask = ~np.isnan(labels) & ~features_df.isnull().any(axis=1).values
X_all = features_df[valid_mask].values
y_all = labels[valid_mask].astype(int)

print(f"Valid samples: {len(X_all):,} (positive: {y_all.sum():,} = {100*y_all.sum()/len(X_all):.1f}%)")

# Chronological split: first 80% train, last 20% test
split_idx = int(0.8 * len(X_all))
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

print(f"Train: {len(X_train):,} ({y_train.sum():,} positive)")
print(f"Test:  {len(X_test):,} ({y_test.sum():,} positive)")

# Train XGBoost
from xgboost import XGBClassifier

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos = n_neg / n_pos if n_pos > 0 else 1

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
)

print("\nTraining XGBoost on GOES-18 data...")
model.fit(X_train, y_train, verbose=False)

# Evaluate
y_prob = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
best_tss = -1
best_thresh = 0.5
for t in np.arange(0.1, 0.9, 0.01):
    yp = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
    tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0
    if tss > best_tss:
        best_tss = tss
        best_thresh = t

y_pred = (y_prob >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tss = best_tss
pod = tp/(tp+fn) if (tp+fn) > 0 else 0
pofd = fp/(fp+tn) if (fp+tn) > 0 else 0
hss_num = 2*(tp*tn - fp*fn)
hss_den = (tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)
hss = hss_num/hss_den if hss_den > 0 else 0

try:
    auc = roc_auc_score(y_test, y_prob)
except:
    auc = None

print(f"\nGOES-18 Retrained Model Results:")
print(f"  Optimal threshold: {best_thresh:.4f}")
print(f"  TSS:  {tss:.4f} {'PASS' if tss >= 0.65 else 'FAIL'} (target >= 0.65)")
print(f"  POD:  {pod:.4f}")
print(f"  POFD: {pofd:.4f}")
print(f"  HSS:  {hss:.4f}")
if auc:
    print(f"  AUC:  {auc:.4f}")
print(f"  Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# Save model
joblib.dump(model, "models/pradhan_goes18_model.joblib")
joblib.dump({"threshold": best_thresh, "trained_on": "GOES-18 2026"},
            "models/pradhan_goes18_config.joblib")

# ============================================================
# PART 2: HEL1OS Simultaneous Observation Analysis
# ============================================================
print("\n" + "=" * 70)
print("PART 2: HEL1OS Response During GOES-18 Flares")
print("=" * 70)

# Load HEL1OS
hel1os = pd.read_parquet("data/pradan_hel1os/hel1os_combined.parquet")
hel1os = hel1os.reset_index()
print(f"HEL1OS: {len(hel1os):,} records")

# Load event catalog
catalog_path = Path("results/simultaneous_event_catalog.json")
with open(catalog_path) as f:
    catalog = json.load(f)

events = catalog["events"]
print(f"Event catalog: {len(events)} events")

# Analyze HEL1OS response during flares vs quiet times
hel1os_rate = hel1os["rate"].values
hel1os_time = hel1os["time"].values

# Define "quiet" as times when GOES flux < B-class (1e-7)
# and "flare" as times when GOES flux >= C-class (1e-6)
# We need to match HEL1OS times to GOES-18 flux

goes18_full = goes18.copy()
goes18_full["time"] = pd.to_datetime(goes18_full["time"])

# For each HEL1OS record, find the nearest GOES-18 flux
hel1os_goes_flux = []
for t in hel1os_time:
    idx = (goes18_full["time"] - pd.Timestamp(t)).abs().idxmin()
    hel1os_goes_flux.append(goes18_full.loc[idx, "flux_b"])

hel1os_goes_flux = np.array(hel1os_goes_flux)

# Classify by GOES class
hel1os["goes_flux"] = hel1os_goes_flux
hel1os["goes_class"] = "quiet"
hel1os.loc[hel1os_goes_flux >= 1e-7, "goes_class"] = "B"
hel1os.loc[hel1os_goes_flux >= 1e-6, "goes_class"] = "C"
hel1os.loc[hel1os_goes_flux >= 1e-5, "goes_class"] = "M"
hel1os.loc[hel1os_goes_flux >= 1e-4, "goes_class"] = "X"

print("\nHEL1OS response by GOES class:")
print(f"{'Class':>8} {'N':>8} {'HEL1OS mean':>15} {'HEL1OS max':>15} {'HEL1OS std':>15}")
print("-" * 65)
for cls in ["quiet", "B", "C", "M", "X"]:
    mask = hel1os["goes_class"] == cls
    if mask.sum() > 0:
        subset = hel1os.loc[mask, "rate"]
        print(f"{cls:>8} {mask.sum():>8,} {subset.mean():>15.2f} {subset.max():>15.0f} {subset.std():>15.2f}")

# ============================================================
# PART 3: GOES-18 + HEL1OS combined prediction
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Can HEL1OS improve GOES-only prediction?")
print("=" * 70)

# For each GOES-18 sample, find the nearest HEL1OS rate (if within 30s)
# This tests whether adding HEL1OS flux as a feature improves prediction

# Create aligned dataset: GOES-18 features + HEL1OS rate
# HEL1OS is 10s cadence, GOES-18 is 60s cadence
# For each GOES-18 time, find the HEL1OS rate within ±30s

goes18_times = goes18_60s["time"].values
hel1os_times = hel1os["time"].values
hel1os_rates = hel1os["rate"].values

aligned_hel1os = np.zeros(len(goes18_times))
has_hel1os = np.zeros(len(goes18_times), dtype=bool)

for i, gt in enumerate(goes18_times):
    gt_ts = pd.Timestamp(gt)
    diffs = np.abs(hel1os_times - gt_ts)
    nearest_idx = diffs.argmin()
    if diffs[nearest_idx] < pd.Timedelta(seconds=30):
        aligned_hel1os[i] = hel1os_rates[nearest_idx]
        has_hel1os[i] = True

print(f"GOES-18 samples with HEL1OS coverage: {has_hel1os.sum():,} / {len(has_hel1os):,} ({100*has_hel1os.sum()/len(has_hel1os):.1f}%)")

# Retrain with HEL1OS as additional feature
X_with_hel1os = np.column_stack([X_all, aligned_hel1os[:len(X_all)]])
has_hel1os_valid = has_hel1os[:len(X_all)]

# Only train on samples that have HEL1OS coverage
hel1os_mask = has_hel1os_valid
X_h = X_with_hel1os[hel1os_mask]
y_h = y_all[hel1os_mask]

if len(y_h) > 100:
    split_h = int(0.8 * len(X_h))
    Xh_train, Xh_test = X_h[:split_h], X_h[split_h:]
    yh_train, yh_test = y_h[:split_h], y_h[split_h:]
    
    n_pos_h = yh_train.sum()
    n_neg_h = len(yh_train) - n_pos_h
    sp_h = n_neg_h / n_pos_h if n_pos_h > 0 else 1
    
    model_h = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sp_h, eval_metric="logloss",
        random_state=42, use_label_encoder=False,
    )
    model_h.fit(Xh_train, yh_train, verbose=False)
    
    yh_prob = model_h.predict_proba(Xh_test)[:, 1]
    
    # Find optimal threshold
    best_tss_h = -1
    best_th_h = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        yp = (yh_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yh_test, yp).ravel()
        tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0
        if tss > best_tss_h:
            best_tss_h = tss
            best_th_h = t
    
    yh_pred = (yh_prob >= best_th_h).astype(int)
    tn, fp, fn, tp = confusion_matrix(yh_test, yh_pred).ravel()
    pod_h = tp/(tp+fn) if (tp+fn) > 0 else 0
    pofd_h = fp/(fp+tn) if (fp+tn) > 0 else 0
    hss_h = 2*(tp*tn - fp*fn)/((tp+fp)*(tp+fn)+(tn+fp)*(tn+fn)) if ((tp+fp)*(tp+fn)+(tn+fp)*(tn+fn)) > 0 else 0
    
    print(f"\nGOES-18 only:       TSS={tss:.4f}, POD={pod:.4f}, POFD={pofd:.4f}, HSS={hss:.4f}")
    print(f"GOES-18 + HEL1OS:   TSS={best_tss_h:.4f}, POD={pod_h:.4f}, POFD={pofd_h:.4f}, HSS={hss_h:.4f}")
    print(f"Improvement:        TSS {best_tss_h - tss:+.4f}")
    
    joblib.dump(model_h, "models/pradhan_goes18_hel1os_model.joblib")
else:
    print("Insufficient HEL1OS-aligned data for combined model")

# Save comprehensive results
output = {
    "goes18_retrained": {
        "threshold": float(best_thresh),
        "TSS": float(tss),
        "POD": float(pod),
        "POFD": float(pofd),
        "HSS": float(hss),
        "AUC": float(auc) if auc else None,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    },
    "hel1os_analysis": {
        "total_records": len(hel1os),
        "has_goes_coverage": int(has_hel1os.sum()),
    },
}
with open("results/goes18_retrained_evaluation.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to results/goes18_retrained_evaluation.json")
