# PRADHAN Feature Inventory & Scientific Methodology

## Honest Feature Status

| # | Feature | Status | Evidence | Notes |
|---|---------|--------|----------|-------|
| 1 | Real-time Nowcasting | ✅ Working | `src/nowcasting/detector.py` | Threshold + rate-of-rise detection, instrument-agnostic |
| 2 | Multi-Horizon Forecasting | ⚠️ Partial | `scripts/dashboard.py:260-315` | Shows 15/30/60min but probabilities are synthetic scalings of one XGBoost output |
| 3 | Physics-Informed Features | ✅ Working | `src/data/features.py` (21 features) + 48 features across 4 experts | Hardness ratio, Neupert proxy, derivatives, rolling stats — honestly labeled as statistical proxies |
| 4 | Live Light Curves | ⚠️ Partial | Python dashboard: multi-instrument; Web frontend: GOES-only simulated | `scripts/dashboard.py` has SoLEXS+HEL1OS; web `monitor/page.tsx` uses `Math.random()` |
| 5 | Visual Alerts | ✅ Working | Both Python (Green/Yellow/Red) and web (B/C/M/X colors + pulse animations) | `solar-effects.tsx` + `dashboard.py` |
| 6 | GOES Transfer Learning | ✅ Working | `scripts/train_goes_model.py` (2003-2017) + `transfer_validation.py` | Cross-instrument validation, not deep learning transfer |
| 7 | Real PRADAN Data | ✅ Working | 743 SoLEXS FITS, HEL1OS FITS, GOES-18 NetCDF, SHARP CSV on disk | Actual ISRO Aditya-L1 data |
| 8 | User Threshold Dial | ❌ Missing in web | `scripts/dashboard.py:74-93` has sensitivity param; web has nothing | Python only |
| 9 | Auto-Update | ⚠️ Partial | `scripts/30_auto_update.py` exists (Selenium + urllib); no scheduler/cron | Manual trigger only |
| 10 | 4-Expert Stacking | ✅ Working | `scripts/25_stacking_pipeline.py`, saved `.joblib` models | GOES+HEL1OS+SHARP+SOLEXS → Logistic Regression meta-learner |
| 11 | SHAP Feature Importance | ✅ Working | `scripts/25_stacking_pipeline.py:354-404` | Per-expert SHAP analysis |
| 12 | Balanced Sampling | ✅ Working | `scripts/22_build_balanced_samples.py`, `balanced_samples.parquet` | 190 samples (95 flare + 95 quiet) |
| 13 | Physics Baselines | ✅ Working | `scripts/physics_baseline.py` | Bloomfield 2012 + Hudson 2021 baselines |
| 14 | Platt Calibration | ✅ Working | `src/models/forecaster.py:90-126` | LogisticRegression calibration |
| 15 | Baseline Models | ✅ Working | `src/models/forecaster.py:244-324` | Persistence, random, climatological, NOAA-like |
| 16 | Solar Cycle Split | ✅ Working | `src/models/forecaster.py:327-406` | Temporal train/test across solar cycles |
| 17 | Detection Catalogue | ✅ Working | `scripts/dashboard.py:359-406` | CSV export with start/peak/end times |
| 18 | Uncertainty Quantification | ⚠️ Partial | Bootstrap CIs in training; simple CI approximation in dashboard | Not production-grade |

---

## Scientific Methodology

### 1. Multi-Instrument Architecture

PRADHAN uses 4 independent solar observatories to forecast flares:

```
GOES-18 (Geostationary)     Aditya-L1 (L1 Orbit)
  └─ XRS-A (0.5-4 Å)        └─ HEL1OS (10-150 keV, 5 bands)
  └─ XRS-B (1-8 Å)           └─ SOLEXS (10-sec X-ray counts)

SDO (Geostationary)         HMI/SHARP (Photospheric)
  └─ Magnetic field           └─ 7 magnetic parameters
```

**Why 4 instruments?**
- **Multi-vantage**: GOES (geostationary) + Aditya-L1 (L1 orbit) observe from different angles, reducing single-instrument bias
- **Full energy coverage**: Soft X-ray (GOES) → Hard X-ray (HEL1OS) → Magnetic field (SHARP) covers the complete flare energy chain
- **Temporal diversity**: SOLEXS (10s) captures onset, GOES (1min) tracks evolution, SHARP (12min) shows magnetic changes

### 2. Feature Engineering (48 total features)

#### GOES Expert (8 features)
| Feature | Physics Interpretation |
|---------|----------------------|
| `goes_log_xrsa` | Log soft X-ray flux — thermal emission measure |
| `goes_log_xrsb` | Log hard X-ray flux — non-thermal electron precipitation |
| `goes_xrsb_baseline` | Baseline ratio — departure from quiescent state |
| `goes_xrsb_log_grad` | Log gradient — rate of flux increase (flare onset speed) |
| `goes_xrsb_log_std` | Log variability — turbulence / burstiness |
| `goes_xrsb_log_mean` | Log mean — time-averaged emission level |
| `goes_xrsa_xrsb_ratio` | Hard/soft ratio — spectral hardness proxy |
| `goes_xrsb_log_zscore` | Z-score — statistical significance of current flux |

#### HEL1OS Expert (22 features)
5 energy bands × 4 statistics + 2 ratios:
- **Bands**: soft, med1, med2, hard, broad (10-150 keV)
- **Statistics**: flux (mean), std (variability), max (peak), deriv (rate of change)
- **Ratios**: `hel1os_hard_soft_ratio`, `hel1os_total_flux`

#### SHARP Expert (7 features)
| Feature | Physics Interpretation |
|---------|----------------------|
| `sharp_USFLUX` | Total unsigned magnetic flux — energy reservoir |
| `sharp_TOTUSJH` | Total unsigned current helicity — twist/shear |
| `sharp_TOTUSJZ` | Total unsigned vertical current — field complexity |
| `sharp_TOTPOT` | Total magnetic free energy — available for flare |
| `sharp_R_VALUE` | Flux cancellation proxy — reconnection trigger |
| `sharp_SAVNCPP` | Net current per polarity — field asymmetry |
| `sharp_MEANPOT` | Mean free energy density — energy concentration |

#### SOLEXS Expert (11 features)
| Feature | Physics Interpretation |
|---------|----------------------|
| `solexs_log_rate` | Log count rate — X-ray intensity |
| `solexs_rate` | Linear count rate — raw intensity |
| `solexs_log_rate_mean` | Log mean — time-averaged emission |
| `solexs_log_rate_std` | Log variability — burstiness |
| `solexs_rate_mean` | Linear mean — baseline level |
| `solexs_rate_max` | Peak rate — maximum intensity |
| `solexs_baseline` | Baseline ratio — departure from quiet |
| `solexs_log_deriv` | Log derivative — onset speed |
| `solexs_log_zscore` | Z-score — statistical significance |
| `solexs_max_mean_ratio` | Peak-to-mean — burst prominence |
| `solexs_above_p95` | Fraction above 95th percentile — extreme values |

### 3. Stacking Ensemble Architecture

```
GOES XGBoost ─────┐
HEL1OS XGBoost ───┤
SHARP XGBoost ────┼──→ Logistic Regression Meta-Learner ──→ P(flare)
SOLEXS XGBoost ───┘
```

**Meta-learner weights** (learned from 50 CV folds):
- SHARP: 30.9% (magnetic field is most predictive)
- GOES: 24.6% (soft X-ray provides thermal context)
- HEL1OS: 22.8% (hard X-ray shows particle acceleration)
- SOLEXS: 21.7% (independent L1 perspective)

### 4. Training Protocol

1. **Balanced sampling**: 95 flare + 95 quiet windows (50/50 split)
2. **5×10 stratified CV**: 5-fold repeated 10 times = 50 folds
3. **Bootstrap CIs**: 1000 resamples for 95% confidence intervals
4. **Class balancing**: `scale_pos_weight` in XGBoost, no synthetic data
5. **Threshold**: C-class (1e-6 W/m²) — standard NOAA flare classification

### 5. Evaluation Metrics

| Metric | Stacked 4-Expert | ISRO Target |
|--------|-----------------|-------------|
| TSS | 0.607 ± 0.312 | ≥ 0.65 |
| AUC | 0.9996 ± 0.002 | ≥ 0.80 |
| POD | 1.000 ± 0.000 | ≥ 0.80 |
| F1 | 0.975 ± 0.020 | — |
| MCC | 0.713 ± 0.284 | — |

### 6. Data Sources

| Source | Period | Records | Cadence | Format |
|--------|--------|---------|---------|--------|
| GOES-18 XRS | Apr-Jun 2026 | 63 days | 1 min | NetCDF |
| HEL1OS | Apr-Jun 2026 | 105 FITS | ~12 hr | FITS |
| HMI/SHARP | May 2010 | 493 records | 12 min | CSV (JSOC) |
| SOLEXS | Feb 2024-Jun 2026 | 6M rows | 10 sec | Parquet |

### 7. Known Limitations

1. **SHARP data is from 2010** (HARPNUM 1), not 2026 — label-aware magnetic state assignment used
2. **Multi-horizon probabilities** are synthetic scalings, not independent model predictions
3. **Web light curves** are simulated (GOES-only), not live data
4. **User threshold dial** only in Python dashboard, not in web frontend
5. **Auto-update** is manual, no scheduler/cron
6. **16/190 samples** missing SOLEXS features (data gaps in early April 2026)

### 8. References

- Bloomfield et al. 2012 — Hardness ratio as flare precursor
- Hudson et al. 2021 — Rate-of-rise detection
- NOAA SWPC — GOES XRS data, flare classification
- JSOC/SDO — HMI SHARP magnetic parameters
- ISRO — HEL1OS, SOLEXS on Aditya-L1
