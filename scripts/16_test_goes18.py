"""
Test GOES-trained model on GOES-18 2026 data.
Resamples GOES-18 to 60s cadence to match training data.
"""
import xarray as xr
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Step 1: Save GOES-18 as parquet ---
print("=" * 70)
print("Step 1: Save GOES-18 2026 as parquet")
print("=" * 70)

goes_dir = Path("data/goes18_2026")
files = sorted(goes_dir.glob("dn_xrsf-l2-flx1s_g18_d*.nc"))
print(f"Loading {len(files)} GOES-18 files...")

dfs = []
for f in files:
    try:
        ds = xr.open_dataset(str(f))
        df = ds.to_dataframe().reset_index()
        df = df[df["quad_diode"] == 0][["time", "xrsa_flux", "xrsb_flux"]].copy()
        dfs.append(df)
    except:
        pass

goes18_df = pd.concat(dfs, ignore_index=True)
goes18_df["time"] = pd.to_datetime(goes18_df["time"])
goes18_df = goes18_df.sort_values("time").reset_index(drop=True)
goes18_df = goes18_df.rename(columns={"xrsa_flux": "flux_a", "xrsb_flux": "flux_b"})

parquet_path = "data/goes/goes18_2026.parquet"
goes18_df.to_parquet(parquet_path, index=False)
print(f"Saved: {parquet_path} ({len(goes18_df):,} records)")

# --- Step 2: Resample to 60s cadence ---
print("\n" + "=" * 70)
print("Step 2: Resample GOES-18 to 60s cadence")
print("=" * 70)

goes18_60s = goes18_df.set_index("time").resample("60s").agg({
    "flux_a": "mean",
    "flux_b": "mean",
}).dropna().reset_index()
print(f"After resampling: {len(goes18_60s):,} records")
print(f"Time range: {goes18_60s['time'].min()} to {goes18_60s['time'].max()}")

# --- Step 3: Create labels ---
print("\n" + "=" * 70)
print("Step 3: Create labels")
print("=" * 70)

# 1-hour ahead = 60 samples at 60s cadence
lookback = 60
flux_b = goes18_60s["flux_b"].values.copy()
flux_b = np.nan_to_num(flux_b, nan=0.0)
temp = pd.Series(flux_b)
future_max = temp.shift(-lookback).rolling(lookback).max().values
labels = (future_max >= 1e-6).astype(int)

n_positive = labels[~np.isnan(future_max)].sum() if not np.all(np.isnan(future_max)) else 0
n_total = len(labels) - lookback
print(f"Total samples: {len(goes18_60s):,}")
print(f"Positive (C-class within 1h): {labels.sum():,} ({100*labels.sum()/n_total:.1f}%)")

# --- Step 4: Compute features ---
print("\n" + "=" * 70)
print("Step 4: Compute features")
print("=" * 70)

from src.data.features import compute_features

soft = goes18_60s["flux_a"].values
hard = goes18_60s["flux_b"].values
features_df = compute_features(soft, hard, cadence_seconds=60.0)
features_df["time"] = goes18_60s["time"].values
features_df["label_1h_C"] = labels

feature_cols = [c for c in features_df.columns if c != "time" and c != "label_1h_C"]
print(f"Features: {len(feature_cols)} columns, {len(features_df)} rows")

# --- Step 5: Evaluate model ---
print("\n" + "=" * 70)
print("Step 5: Evaluate GOES-trained model on GOES-18 2026")
print("=" * 70)

model = joblib.load("models/pradhan_best_model.joblib")
config = joblib.load("models/pradhan_best_config.joblib")
threshold = config["threshold"]

X = features_df[feature_cols].values
y_true = features_df["label_1h_C"].values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

valid = ~np.isnan(y_true)
y_true_v = y_true[valid].astype(int)
y_pred_v = y_pred[valid]
y_prob_v = y_prob[valid]

tn, fp, fn, tp = confusion_matrix(y_true_v, y_pred_v).ravel()
tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0
pofd = fp/(fp+tn) if (fp+tn) > 0 else 0
pod = tp/(tp+fn) if (tp+fn) > 0 else 0
hss = 2*(tp*tn - fp*fn)/((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) if ((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) > 0 else 0
precision = precision_score(y_true_v, y_pred_v, zero_division=0)
recall = recall_score(y_true_v, y_pred_v, zero_division=0)
f1 = f1_score(y_true_v, y_pred_v, zero_division=0)

try:
    auc = roc_auc_score(y_true_v, y_prob_v)
except:
    auc = None

print(f"\nResults:")
print(f"  Samples:    {len(y_true_v):,} (positive: {y_true_v.sum():,} = {100*y_true_v.sum()/len(y_true_v):.1f}%)")
print(f"  Threshold:  {threshold:.4f}")
print(f"  TSS:        {tss:.4f} {'PASS' if tss >= 0.65 else 'FAIL'} (target >= 0.65)")
print(f"  POD:        {pod:.4f}")
print(f"  POFD:       {pofd:.4f}")
print(f"  HSS:        {hss:.4f}")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")
print(f"  F1:         {f1:.4f}")
if auc:
    print(f"  AUC:        {auc:.4f}")
print(f"  Confusion:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# Compare with GOES-2017 test set
print("\n" + "=" * 70)
print("Comparison with GOES-2017 test set")
print("=" * 70)

ref = {"TSS": 0.7931, "POD": 0.8438, "POFD": 0.0556, "HSS": 0.7969}
curr = {"TSS": tss, "POD": pod, "POFD": pofd, "HSS": hss}
print(f"\n{'Metric':>8} {'GOES-2017':>12} {'GOES-18':>12} {'Delta':>10}")
print("-" * 45)
for k in ref:
    d = curr[k] - ref[k]
    print(f"{k:>8} {ref[k]:>12.4f} {curr[k]:>12.4f} {d:>+10.4f}")

# Save results
output = {
    "goes18_evaluation": {
        "total_samples": int(len(y_true_v)),
        "positive_samples": int(y_true_v.sum()),
        "threshold": float(threshold),
        "TSS": float(tss), "POD": float(pod), "POFD": float(pofd),
        "HSS": float(hss), "precision": float(precision),
        "recall": float(recall), "F1": float(f1),
        "AUC": float(auc) if auc else None,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    },
    "goes2017_reference": ref,
    "generalization_gap": {k: float(curr[k] - ref[k]) for k in ref},
}
with open("results/goes18_evaluation.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to results/goes18_evaluation.json")
