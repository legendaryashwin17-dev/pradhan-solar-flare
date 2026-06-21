"""
PRADHAN Script 05 — Load HEL1OS FITS into Parquet
==================================================

Loads HEL1OS FITS files from the extracted data and combines into
a single parquet file for training and analysis.

HEL1OS (High Energy L1 Object Spectrometer) is on Aditya-L1
and observes X-rays in the 8-150 keV range.

Usage:
    python scripts/05_load_hel1os.py

Input:  data/pradan_hel1os/**/*.fits
Output: data/pradan_hel1os/hel1os_combined.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from astropy.io import fits as pyfits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not available, cannot read FITS files")


def load_hel1os_fits(filepath: str) -> pd.DataFrame:
    """
    Load a single HEL1OS FITS file.
    
    Parameters
    ----------
    filepath : str
        Path to .fits file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, rate, error
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("astropy required for FITS reading")
    
    with pyfits.open(filepath) as hdul:
        # Get data from first extension
        if len(hdul) > 1:
            data = hdul[1].data
        else:
            data = hdul[0].data
        
        # Extract columns (HEL1OS format varies, adapt as needed)
        # Common column names: TIME, RATE, ERROR, COUNTS
        col_names = [col.name.lower() for col in data.columns]
        
        # Build DataFrame
        df = pd.DataFrame()
        
        # Try to find time column
        time_cols = ['time', 'tmid', 't_start', 't']
        for tc in time_cols:
            if tc in col_names:
                df['time'] = data[tc.upper()]
                break
        
        # Try to find rate/count column
        rate_cols = ['rate', 'count_rate', 'counts', 'cnts']
        for rc in rate_cols:
            if rc in col_names:
                df['rate'] = data[rc.upper()]
                break
        
        # Try to find error column
        error_cols = ['error', 'rate_err', 'count_err', 'stat_err']
        for ec in error_cols:
            if ec in col_names:
                df['error'] = data[ec.upper()]
                break
        
        if 'error' not in df.columns:
            # Compute Poisson error from rate if available
            if 'rate' in df.columns:
                df['error'] = np.sqrt(np.abs(df['rate']))
            else:
                df['error'] = 0
        
        # Convert time to datetime index if needed
        # HEL1OS times are often in seconds since mission start
        # Need to convert to absolute time
        if 'time' in df.columns:
            # Try to parse as datetime
            try:
                df['time'] = pd.to_datetime(df['time'])
                df.index = df['time']
                df = df.drop(columns=['time'])
            except:
                # If can't parse, use integer index
                df.index = df['time']
                df = df.drop(columns=['time'])
        
        df['source'] = Path(filepath).stem
    
    return df


def load_all_hel1os(
    hel1os_dir: str = "data/pradan_hel1os",
    output_parquet: str = "data/pradan_hel1os/hel1os_combined.parquet",
    resample_to: str = "10s",
):
    """
    Load all HEL1OS FITS files and save as parquet.
    
    Parameters
    ----------
    hel1os_dir : str
        Directory containing HEL1OS FITS files
    output_parquet : str
        Output parquet file path
    resample_to : str, optional
        Resample cadence (default 10s for manageable size)
    """
    hel1os_path = Path(hel1os_dir)
    
    if not hel1os_path.exists():
        raise FileNotFoundError(f"HEL1OS directory not found: {hel1os_dir}")
    
    # Find all FITS files
    fits_files = list(hel1os_path.glob("**/*.fits"))
    
    if not fits_files:
        raise FileNotFoundError(f"No .fits files found in {hel1os_dir}")
    
    print(f"Found {len(fits_files)} HEL1OS FITS files")
    
    dfs = []
    loaded = 0
    errors = 0
    
    for fits_file in fits_files:
        try:
            df = load_hel1os_fits(str(fits_file))
            if not df.empty:
                dfs.append(df)
                loaded += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: {fits_file.name}: {e}")
    
    if not dfs:
        raise ValueError("No HEL1OS data could be loaded")
    
    print(f"Loaded {loaded} files ({errors} errors)")
    
    # Combine
    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()
    
    # Remove exact duplicates
    combined = combined[~combined.index.duplicated(keep='first')]
    
    print(f"Combined: {len(combined):,} data points")
    if len(combined) > 0:
        print(f"Time range: {combined.index.min()} to {combined.index.max()}")
    
    # Resample to reduce size
    if resample_to and len(combined) > 0:
        print(f"Resampling to {resample_to} cadence...")
        numeric_cols = ['rate', 'error']
        available_cols = [c for c in numeric_cols if c in combined.columns]
        combined = combined[available_cols].resample(resample_to).mean()
        combined = combined.dropna(subset=['rate'])
        print(f"After resample: {len(combined):,} data points")
    
    # Save
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_parquet)
    print(f"Saved to {output_parquet}")
    
    # Stats
    if 'rate' in combined.columns:
        print(f"\nStatistics:")
        print(f"  Rate range: {combined['rate'].min():.2f} to {combined['rate'].max():.2f}")
        print(f"  Mean rate: {combined['rate'].mean():.2f}")
    
    print(f"  File size: {Path(output_parquet).stat().st_size / 1e6:.1f} MB")
    
    return combined


if __name__ == "__main__":
    load_all_hel1os()
