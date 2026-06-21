"""
PRADHAN Script 04 — Load SoLEXS FITS into Parquet
==================================================

Loads extracted SoLEXS .lc FITS files and combines into a single
parquet file for training and analysis.

Usage:
    python scripts/04_load_solexs.py

Input:  data/pradan_solexs/extracted/*.lc
Output: data/pradan_solexs/solexs_combined.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_solexs_lc


def load_all_solexs(
    extracted_dir: str = "data/pradan_solexs/extracted",
    output_parquet: str = "data/pradan_solexs/solexs_combined.parquet",
    resample_to: str = "10s",
):
    """
    Load all extracted SoLEXS .lc files and save as parquet.

    Parameters
    ----------
    extracted_dir : str
        Directory containing extracted .lc files
    output_parquet : str
        Output parquet file path
    resample_to : str
        Resample cadence (default 10s for manageable size)
    """
    ext_path = Path(extracted_dir)

    if not ext_path.exists():
        raise FileNotFoundError(
            f"Extracted directory not found: {extracted_dir}\n"
            f"Run 03_extract_solexs.py first."
        )

    lc_files = sorted(ext_path.glob("*.lc"))

    if not lc_files:
        raise FileNotFoundError(f"No .lc files found in {extracted_dir}")

    print(f"Found {len(lc_files)} SoLEXS light curve files")

    dfs = []
    loaded = 0
    errors = 0

    for lc_file in lc_files:
        try:
            df = load_solexs_lc(str(lc_file))
            df['source'] = lc_file.stem
            dfs.append(df)
            loaded += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: {lc_file.name}: {e}")

    if not dfs:
        raise ValueError("No light curves could be loaded")

    print(f"Loaded {loaded} files ({errors} errors)")

    # Combine
    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()

    # Remove exact duplicates
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"Combined: {len(combined):,} data points")
    print(f"Time range: {combined.index.min()} to {combined.index.max()}")

    # Resample to reduce size
    if resample_to:
        print(f"Resampling to {resample_to} cadence...")
        numeric_cols = ['rate', 'error']
        combined = combined[numeric_cols].resample(resample_to).mean()
        combined = combined.dropna(subset=['rate'])
        print(f"After resample: {len(combined):,} data points")

    # Save
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_parquet)
    print(f"Saved to {output_parquet}")

    # Stats
    print(f"\nStatistics:")
    print(f"  Rate range: {combined['rate'].min():.2f} to {combined['rate'].max():.2f}")
    print(f"  Mean rate: {combined['rate'].mean():.2f}")
    print(f"  File size: {Path(output_parquet).stat().st_size / 1e6:.1f} MB")

    return combined


if __name__ == "__main__":
    load_all_solexs()
