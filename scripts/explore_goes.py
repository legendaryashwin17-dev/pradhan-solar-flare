"""
PRADHAN Quick Start — Data Exploration
=======================================
Run this script to verify data loading and see basic statistics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, GOES_PARQUET_DIR, FLUX_THRESHOLDS

import pandas as pd
import numpy as np


def explore_goes():
    """Load and explore GOES parquet data."""
    print("=" * 60)
    print("PRADHAN — GOES Data Exploration")
    print("=" * 60)

    # Load data
    goes_dir = DATA_DIR / "goes"
    parquet_files = sorted(goes_dir.glob("*.parquet"))
    print(f"\nFound {len(parquet_files)} parquet files")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)
        print(f"  {f.name}: {len(df):,} rows")

    goes = pd.concat(dfs, ignore_index=False)
    goes = goes.sort_index()

    # Rename columns (only if not already present)
    rename_map = {}
    if "xrsa" in goes.columns and "xrs_a_flux" not in goes.columns:
        rename_map["xrsa"] = "xrs_a_flux"
    if "xrsb" in goes.columns and "xrs_b_flux" not in goes.columns:
        rename_map["xrsb"] = "xrs_b_flux"
    if rename_map:
        goes = goes.rename(columns=rename_map)

    # Drop duplicate columns if any
    goes = goes.loc[:, ~goes.columns.duplicated()]

    print(f"\nTotal records: {len(goes):,}")
    print(f"Time range:    {goes.index.min()} to {goes.index.max()}")
    print(f"Columns:       {goes.columns.tolist()}")
    print(f"\nXRS-B (hard) flux stats:")
    flux_min = float(goes["xrs_b_flux"].min())
    flux_max = float(goes["xrs_b_flux"].max())
    flux_mean = float(goes["xrs_b_flux"].mean())
    print(f"  Min:  {flux_min:.2e} W/m2")
    print(f"  Max:  {flux_max:.2e} W/m2")
    print(f"  Mean: {flux_mean:.2e} W/m2")

    # Flare classification
    print(f"\nFlare classification (XRS-B flux):")
    for cls, threshold in sorted(FLUX_THRESHOLDS.items(), key=lambda x: x[1]):
        count = (goes["xrs_b_flux"] >= threshold).sum()
        pct = 100 * count / len(goes)
        print(f"  {cls}: {count:>8,} samples ({pct:.2f}%)")

    # Quality check
    if "xrs_b_quality" in goes.columns:
        bad = (goes["xrs_b_quality"] > 0).sum()
        print(f"\nQuality flags > 0: {bad:,} ({100*bad/len(goes):.2f}%)")

    # Gap analysis
    time_diff = goes.index.to_series().diff()
    big_gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
    print(f"\nGaps > 1 hour: {len(big_gaps)}")
    if len(big_gaps) > 0:
        print(f"  Largest gap: {big_gaps.max()}")

    print(f"\n{'=' * 60}")
    print("Data exploration complete!")
    return goes


if __name__ == "__main__":
    goes = explore_goes()
