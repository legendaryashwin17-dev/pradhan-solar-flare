"""
PRADHAN Configuration — Central Path Management
================================================
All data paths, output directories, and project settings in one place.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
NOTEBOOKS_DIR = WORKSPACE / "notebooks"

# GOES data source (external)
GOES_PARQUET_DIR = Path(r"C:\Users\Admin\aditya-flare-forecast\data\goes_historical")
GOES_NETCDF_DIR = GOES_PARQUET_DIR  # Same directory, .nc files alongside .parquet

# SoLEXS data source (external)
SOLEXS_ZIP_DIR = Path(r"C:\Users\Admin\Downloads")
SOLEXS_EXTRACT_DIR = DATA_DIR / "pradan_solexs" / "extracted"

# ── GOES Configuration ─────────────────────────────────────────────────
GOES_COLUMNS = {
    "xrsa": "xrs_a_flux",      # XRS-A (soft channel, 0.05-0.4 nm)
    "xrsb": "xrs_b_flux",      # XRS-B (hard channel, 0.1-0.8 nm)
    "xrsa_quality": "xrs_a_quality",
    "xrsb_quality": "xrs_b_quality",
}

# NOAA flare classification thresholds (W/m^2)
FLUX_THRESHOLDS = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}

# ── Feature Engineering ────────────────────────────────────────────────
FEATURE_CONFIG = {
    "window_sizes": [10, 30, 60],       # minutes
    "derivative_orders": [1, 2],         # first and second derivatives
    "cross_correlation_lags": [5, 15, 30],  # minutes
}

# ── Model Configuration ────────────────────────────────────────────────
MODEL_CONFIG = {
    "target_col": "xrs_b_flux",
    "label_threshold": FLUX_THRESHOLDS["C"],   # C-class and above
    "forecast_horizon_hours": 24,
    "test_fraction": 0.2,
    "random_state": 42,
}

# ── Visualization ──────────────────────────────────────────────────────
VIZ_CONFIG = {
    "figure_size": (14, 8),
    "dpi": 150,
    "style": "seaborn-v0_8-darkgrid",
    "color_flux_a": "#2196F3",   # blue
    "color_flux_b": "#F44336",   # red
    "color_threshold": "#FF9800", # orange
}

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
