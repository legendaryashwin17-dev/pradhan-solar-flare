# PRADHAN Multi-Input Pipeline — Execution Plan

## Overview

Build a multi-input solar flare forecasting system using:
- **GOES XRS** (X-ray flux, 21 statistical features)
- **SHARP magnetic features** (7 parameters from SDO/HMI)
- **HEL1OS** (hard X-ray light curves from Aditya-L1)

**Model:** XGBoost with early fusion  
**Outputs:** P(flare in 1h), P(flare in 6h), Predicted class (B/C/M/X)  
**Validation:** Active-region split, Solar-cycle split, TSS, HSS, ROC-AUC, PR-AUC, Reliability diagram

---

## Phase 1: NOAA SWPC Flare Catalogue (with AR numbers)

**Goal:** Get a complete flare event list linking each flare to its source active region.

**Source:** NCEI — `https://www.ncei.noaa.gov/data/goes-xray-flare-daily-event-list/`  
**Alternative:** HEK via sunpy (code already exists in `src/data/ingestion.py` but doesn't extract `ar_noaanum`)

**Script:** `scripts/20_fetch_flare_catalogue.py`

**Steps:**
1. Download NOAA SWPC daily event lists (CSV format, one per year)
2. Parse into DataFrame: `start_time, peak_time, end_time, goes_class, peak_flux, ar_number`
3. Save to `data/noaa_flare_catalogue.csv`

**Output:** `data/noaa_flare_catalogue.csv` (~10,000 events, 2003-2025)

---

## Phase 2: SHARP Magnetic Feature Data

**Goal:** Download SHARP parameters for all active regions that produced flares.

**Source:** JSOC (Joint Science Operations Center) via `drms` Python library  
**Key parameters:** USFLUX, TOTUSJH, TOTUSJZ, TOTPOT, R_VALUE, SAVNCPP, MEANPOT

**Script:** `scripts/21_download_sharp.py`

**Steps:**
1. Install `drms` package
2. For each AR in the flare catalogue, query SHARP data at 12-minute cadence
3. Extract the 7 key magnetic parameters
4. Align timestamps with GOES XRS data
5. Save to `data/sharp_features.parquet`

**Challenge:** JSOC rate limits — need to batch queries efficiently  
**Fallback:** If JSOC is slow, use pre-computed SHARP summaries from SWPC

**Output:** `data/sharp_features.parquet` (~500K rows, 7 features)

---

## Phase 3: HEL1OS Data Loading

**Goal:** Load all 105 HEL1OS lightcurve observations into a combined parquet.

**Existing code:** `scripts/05_load_hel1os.py` and `scripts/13_load_hel1os.py`  
**Data location:** `C:\Users\Admin\.mavis\sessions\...\pradhan-solar-flare-repo\data\pradan_hel1os\` (105 FITS files)

**Script:** `scripts/22_load_hel1os_all.py`

**Steps:**
1. Scan all 105 HEL1OS observation directories
2. Extract CdTe lightcurves (8-150 keV, highest energy channel)
3. Resample to 1-minute cadence (matching GOES)
4. Compute features: flux, derivatives, rolling stats (similar to GOES features but fewer channels)
5. Save to `data/hel1os_combined.parquet`

**HEL1OS features (estimated ~8):**
- `hel1os_flux` (raw count rate)
- `hel1os_log` (log10 flux)
- `hel1os_dflux` (derivative)
- `hel1os_mean_5m` (5-min rolling mean)
- `hel1os_std_5m` (5-min rolling std)
- `hel1os_peak_15m` (15-min max)
- `hel1os_flux_ratio` (hard/soft if dual-channel available)
- `hel1os_quality` (data quality flag)

**Output:** `data/hel1os_combined.parquet`

---

## Phase 4: Multi-Input Feature Pipeline

**Goal:** Combine GOES + SHARP + HEL1OS features into a single feature vector per timestep.

**Script:** `src/data/multi_features.py`

**Feature Groups:**

### Group A — Magnetic (SHARP, 7 features)
| Feature | Description | Physics |
|---------|-------------|---------|
| `usflux` | Total unsigned magnetic flux | Energy storage |
| `totusjh` | Total unsigned current helicity | Twist/shear |
| `totusjz` | Total unsigned vertical current | Current systems |
| `totpot` | Total photospheric magnetic free energy | Available energy |
| `r_value` | R-value (flux cancellation proxy) | Reconnection trigger |
| `savncpp` | Absolute value of net current per polarity | Current imbalance |
| `meanpot` | Mean photospheric magnetic free energy density | Energy density |

### Group B — X-ray (GOES, 21 features)
- Existing features from `src/data/features.py` (soft, hard, ratios, derivatives, rolling stats)

### Group C — Hard X-ray (HEL1OS, ~8 features)
- New features from Phase 3

**Total feature count: ~36 features**

### Alignment Strategy:
- All data resampled to 1-minute cadence
- SHARP data interpolated to 1-minute timestamps
- HEL1OS data resampled to 1-minute
- Drop rows where SHARP or HEL1OS data is missing (valid only during Aditya-L1 observations)

**Output:** `data/multi_input_features.parquet`

---

## Phase 5: Label Engineering

**Goal:** Create binary labels for flare occurrence at 1h and 6h horizons.

**Existing code:** `src/data/labels.py` (already supports multiple horizons)

**Script:** `src/data/multi_labels.py`

**Steps:**
1. Use NOAA SWPC catalogue for ground-truth flare events
2. For each timestamp, check if a flare occurs within the next 1h / 6h
3. Create labels: `flare_1h` (binary), `flare_6h` (binary), `flare_class_1h` (B/C/M/X)
4. Merge with AR numbers for validation splitting

**Output:** `data/multi_input_labels.parquet`

---

## Phase 6: Train/Test Split (AR-based + Solar-cycle)

**Goal:** Implement proper validation splits that prevent data leakage.

**Script:** `src/data/splits.py`

### Split Strategy:

**Primary split (AR-based holdout):**
- Train: ARs that produced flares before 2020
- Test: ARs that produced flares after 2020
- This ensures the model generalizes to unseen active regions

**Secondary split (solar-cycle):**
- Train: Solar cycle 24 (2010-2019)
- Test: Solar cycle 25 (2020-2025)
- This tests generalization across solar cycles

**Combined split:**
- Train: ARs from cycle 24
- Test: ARs from cycle 25 (unseen regions + unseen cycle)

---

## Phase 7: Model Training (Early Fusion XGBoost)

**Goal:** Train XGBoost on concatenated GOES + SHARP + HEL1OS features.

**Script:** `scripts/23_train_multi_input.py`

**Model config:**
```python
XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    scale_pos_weight=auto,  # adjusted per class
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric='logloss',
    early_stopping_rounds=50,
)
```

**Training pipeline:**
1. Load multi-input features + labels
2. Split by AR numbers (Phase 6)
3. Train with early stopping on validation set
4. Optimize threshold for TSS
5. Save model + config

**Output:** `models/pradhan_multi_input_model.joblib`

---

## Phase 8: Ablation Experiments

**Goal:** Quantify contribution of each data source.

**Script:** `scripts/24_ablation.py`

### Experiments:
| Exp | Features | Question |
|-----|----------|----------|
| 1 | GOES only (21 features) | Baseline — what can X-rays alone do? |
| 2 | GOES + HEL1OS (29 features) | Does hard X-ray add value? |
| 3 | GOES + SHARP (28 features) | Does magnetic data add value? |
| 4 | GOES + SHARP + HEL1OS (36 features) | Full model — combined benefit |

### Metrics per experiment:
- TSS, HSS, ROC-AUC, PR-AUC
- Reliability diagram data
- Feature importance ranking

**Output:** `results/ablation_results.json`

---

## Phase 9: Evaluation & Visualization

**Goal:** Generate publication-quality evaluation plots.

**Script:** `scripts/25_evaluate.py`

**Plots:**
1. Reliability diagram (calibration) — predicted vs observed probability
2. ROC curve with AUC
3. PR curve with AUC
4. TSS comparison across ablation experiments
5. Feature importance (SHARP vs GOES vs HEL1OS)
6. Confusion matrix
7. Event rate vs TSS scatter
8. Solar-cycle performance comparison

---

## Phase 10: Dashboard Integration

**Goal:** Update the Next.js dashboard with multi-input results.

**Updates needed:**
- Add SHARP magnetic feature visualization
- Add HEL1OS light curve display
- Show ablation experiment results
- Update model metrics with new multi-input performance
- Add AR-based validation results

---

## File Structure (New)

```
data/
├── goes/                          (existing, 23 parquet files)
├── pradan_solexs/                 (existing)
├── pradan_hel1os/                 (NEW — copy from pradhan-solar-flare-repo)
├── noaa_flare_catalogue.csv       (NEW — Phase 1)
├── sharp_features.parquet         (NEW — Phase 2)
├── hel1os_combined.parquet        (NEW — Phase 3)
├── multi_input_features.parquet   (NEW — Phase 4)
└── multi_input_labels.parquet     (NEW — Phase 5)

scripts/
├── 20_fetch_flare_catalogue.py    (NEW)
├── 21_download_sharp.py           (NEW)
├── 22_load_hel1os_all.py          (NEW)
├── 23_train_multi_input.py        (NEW)
├── 24_ablation.py                 (NEW)
└── 25_evaluate.py                 (NEW)

src/data/
├── multi_features.py              (NEW — feature fusion)
├── multi_labels.py                (NEW — label engineering)
└── splits.py                      (NEW — AR/solar-cycle splits)

results/
├── ablation_results.json          (NEW)
└── multi_input_results.json       (NEW)

models/
└── pradhan_multi_input_model.joblib  (NEW)
```

---

## Execution Order

| Phase | Script | Dependencies | Est. Time |
|-------|--------|--------------|-----------|
| 1 | `20_fetch_flare_catalogue.py` | sunpy, internet | 30 min |
| 2 | `21_download_sharp.py` | drms, JSOC access | 2-4 hours |
| 3 | `22_load_hel1os_all.py` | astropy | 15 min |
| 4 | `multi_features.py` | Phases 1-3 | 1 hour |
| 5 | `multi_labels.py` | Phase 1 | 30 min |
| 6 | `splits.py` | Phase 1 | 15 min |
| 7 | `23_train_multi_input.py` | Phases 4-6 | 30 min |
| 8 | `24_ablation.py` | Phase 7 | 1 hour |
| 9 | `25_evaluate.py` | Phase 8 | 1 hour |
| 10 | Dashboard updates | Phase 9 | 2 hours |

**Total estimated time: 8-12 hours**

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| JSOC rate limits for SHARP | Batch queries, use cache, fallback to SWPC summaries |
| HEL1OS data quality | Quality flags already computed, filter low-quality segments |
| SHARP-GOES timestamp misalignment | Interpolate SHARP to 1-minute cadence |
| Class imbalance (X-class ~1%) | Focal loss, SMOTE, scale_pos_weight |
| Data leakage (AR in both train/test) | Strict AR-based splitting |
