# PRADHAN — Solar Flare Nowcasting System

**P**redictive **R**eal-time **A**nalysis of **D**ata from **H**eliospheric **A**ditya-**N**avigation

## 🏆 Results Summary

| Metric | ISRO Target | PRADHAN Achieved | Status |
|--------|-------------|------------------|--------|
| **TSS** | ≥ 0.65 | **0.7931** | ✅ Exceeds |
| **AUC-ROC** | ≥ 0.80 | **0.9611** | ✅ Exceeds |
| **POD** | ≥ 0.80 | **0.8438** | ✅ Exceeds |
| **HSS** | - | **0.7969** | ✅ Excellent |

**Best Configuration:** 1-hour horizon, C-class threshold (event rate: 23.6%)

## What's New (v1.0)

- 🎯 **6 configuration comparisons** (1h/6h/24h × C/M-class)
- 📊 **14+ publication-quality plots** generated
- 🚀 **Beautiful Streamlit dashboard** with custom dark UI
- 📈 **SHAP analysis** for model interpretability
- 🔄 **NetCDF→Parquet converter** for GOES data
- 🗜️ **SoLEXS extraction** (743 light curve files from ZIPs)

## Scientific Honesty

PRADHAN uses **statistical features** derived from X-ray light curves for flare forecasting. These are NOT physics-based parameters.

### What We Use (Statistical Proxies):
- Derivatives, ratios, variances of time series
- Correlate empirically with flaring activity
- Validated through literature (Bloomfield et al. 2012)

### What We Don't Use (True Physics):
- Magnetic free energy (requires vector magnetograms)
- Shear angles (requires vector magnetograms)
- R-value, WLSG parameters (require magnetograms)

### Label Definitions:
- Labels derived from X-ray flux thresholds (NOAA classification)
- NOT fully independent from feature channels
- Acknowledged limitation: label-feature circularity risk

## Project Structure

```
PRADHAN/
├── src/
│   ├── data/
│   │   ├── reader.py          # GOES parquet + SoLEXS FITS loading
│   │   ├── features.py        # 19 statistical proxy features
│   │   ├── labels.py          # NOAA-compliant flare labels
│   │   └── active_regions.py  # AR tracking (synthetic demo)
│   ├── models/
│   │   ├── forecaster.py      # XGBoost model + baselines
│   │   └── ensemble.py        # Multi-model ensemble
│   ├── nowcasting/
│   │   └── detector.py        # Threshold-based real-time detection
│   └── evaluation/
│       ├── metrics.py         # TSS, HSS, AUC, Brier
│       └── calibration.py     # Reliability analysis
├── scripts/
│   ├── 01_train.py            # Train on GOES data
│   ├── 02_train_multi_config.py  # Train 6 configs for comparison
│   ├── 03_train_best.py       # Train and save best model
│   ├── 03_extract_solexs.py   # Extract SoLEXS from ZIPs
│   ├── 04_load_solexs.py      # Load extracted FITS to parquet
│   ├── convert_nc_to_parquet.py  # NetCDF→Parquet converter
│   ├── plot_all.py            # Generate all comparison plots
│   ├── shap_analysis.py       # SHAP model interpretability
│   ├── dashboard_v2.py        # Beautiful Streamlit dashboard
│   ├── colab_train.ipynb      # Google Colab training notebook
│   └── monitor_goes.py        # Download progress monitor
├── results/
│   ├── best_model_results.json    # Best model metrics
│   ├── multi_config_results.json  # All 6 config comparison
│   └── training_results.json      # Training logs
├── models/
│   ├── pradhan_best_model.joblib      # Best trained model
│   └── pradhan_best_config.joblib     # Best model config
└── data/
    ├── goes/                  # GOES parquet files (2003-2024)
    ├── pradan_solexs/         # SoLEXS extracted light curves
    └── pradan_hel1os/         # HEL1OS light curves
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract SoLEXS data from downloaded ZIPs
python scripts/03_extract_solexs.py

# Load extracted FITS into parquet
python scripts/04_load_solexs.py

# Convert GOES NetCDF to parquet (if needed)
python scripts/convert_nc_to_parquet.py

# Train all configurations and find best model
python scripts/03_train_best.py

# Generate comparison plots
python scripts/plot_all.py

# Run SHAP analysis
python scripts/shap_analysis.py

# Run the beautiful dashboard
streamlit run scripts/dashboard_v2.py
```

## Google Colab Training

1. Open `scripts/colab_train.ipynb` in Colab
2. Upload GOES parquet files (or use synthetic data for testing)
3. Run all cells to train and evaluate
4. Download trained model

## Data Sources

### GOES (Training Data)
- Source: NOAA NCEI
- Period: 2003-2016 (14 years)
- Resolution: 1-minute cadence
- Columns: XRS-A (0.5-4A), XRS-B (1-8A)

### Aditya-L1 SoLEXS (Deployment Target)
- Source: PRADAN portal (pradan.issdc.gov.in)
- Instrument: Soft X-ray Spectrometer (2-22 keV)
- Resolution: 1-second cadence
- Aperture: SDD2 (flare-optimized)

### Aditya-L1 HEL1OS (Future)
- Source: PRADAN portal
- Instrument: High Energy L1 Observatory (8-150 keV)
- Resolution: 1-second cadence

## Key Features (19 Statistical Proxies)

1. `soft`, `hard` — Raw flux channels
2. `soft_log`, `hard_log` — Log10-transformed flux
3. `hard_soft_ratio` — Hard/soft ratio
4. `dsoft`, `dhard` — First derivatives (rate of change)
5. `soft_mean_1m`, `hard_mean_1m` — 1-minute rolling means
6. `soft_mean_5m`, `hard_mean_5m` — 5-minute rolling means
7. `soft_std_1m`, `hard_std_1m` — 1-minute rolling std
8. `soft_std_5m`, `hard_std_5m` — 5-minute rolling std
9. `soft_hard_corr` — Pearson correlation
10. `xcorr` — Cross-correlation at lag-1
11. `dhard_soft_ratio` — Derivative ratio
12. `ddsoft` — Second derivative (acceleration)

## Evaluation Metrics

- **TSS** (True Skill Statistic): Primary metric, not affected by base rate
- **HSS** (Heidke Skill Score): Secondary skill metric
- **AUC-ROC**: Discrimination ability
- **PR-AUC**: Precision-recall area (better for imbalanced data)
- **Brier Score**: Probability calibration quality
- **Bootstrap CI**: Uncertainty quantification

## Limitations

1. **Label circularity**: Labels derived from same X-ray channels as features
2. **No magnetogram features**: No AR complexity measures available
3. **Transfer learning**: GOES-trained model applied to Aditya-L1 (different instrument response)
4. **Single-instrument validation**: Not yet validated against independent data sources
5. **Solar cycle generalization**: Performance may degrade across cycle phases

## References

- Bloomfield et al. (2012) — Statistical features for flare forecasting
- Williams et al. (2025) — XGBoost for solar flare prediction (TSS=0.804)
- Woodcock & Jolliffe (2008) — Metric selection for geophysical forecasting
- NOAA Space Weather Prediction Center — Flare classification scales
