"""
PRADHAN Data Reader — GOES & Aditya-L1 Data
=============================================

Handles loading GOES X-ray data from parquet files and
Aditya-L1 SoLEXS/HEL1OS data from FITS light curves.

Data Sources:
- GOES: NOAA NCEI parquet files (xrsa, xrsb columns, 1-min cadence)
- SoLEXS: PRADAN portal FITS light curves (2-22 keV, 1s cadence)
- HEL1OS: PRADAN portal FITS light curves (8-150 keV, 1s cadence)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta


# GOES X-ray flux thresholds for NOAA classification
FLUX_THRESHOLDS = {
    'A': 1e-8,
    'B': 1e-7,
    'C': 1e-6,
    'M': 1e-5,
    'X': 1e-4,
}


def load_goes_parquet(data_dir: str) -> pd.DataFrame:
    """
    Load GOES data from parquet files.

    The actual parquet files have columns: xrsa, xrsb, xrsa_quality, xrsb_quality
    We rename them to xrs_a_flux, xrs_b_flux for consistency.

    Parameters
    ----------
    data_dir : str
        Directory containing parquet files

    Returns
    -------
    pd.DataFrame
        DataFrame with columns xrs_a_flux, xrs_b_flux, xrs_a_quality, xrs_b_quality
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(list(data_path.glob("*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)

    goes = pd.concat(dfs, ignore_index=False)
    goes = goes.sort_index()

    # Rename columns from raw parquet names to standard names
    rename_map = {}
    if 'xrsa' in goes.columns:
        rename_map['xrsa'] = 'xrs_a_flux'
    if 'xrsb' in goes.columns:
        rename_map['xrsb'] = 'xrs_b_flux'
    if 'xrsa_quality' in goes.columns:
        rename_map['xrsa_quality'] = 'xrs_a_quality'
    if 'xrsb_quality' in goes.columns:
        rename_map['xrsb_quality'] = 'xrs_b_quality'

    goes = goes.rename(columns=rename_map)

    # Drop duplicate columns if any
    goes = goes.loc[:, ~goes.columns.duplicated()]

    # Validate required columns
    required_cols = ['xrs_a_flux', 'xrs_b_flux']
    for col in required_cols:
        if col not in goes.columns:
            raise ValueError(f"Missing required column: {col}")

    return goes


def load_solexs_lc(lc_path: str) -> pd.DataFrame:
    """
    Load a single SoLEXS FITS light curve file.

    Parameters
    ----------
    lc_path : str
        Path to .lc FITS file

    Returns
    -------
    pd.DataFrame
        DataFrame with time index and columns: rate, error
    """
    from astropy.io import fits

    with fits.open(lc_path) as hdul:
        data = hdul[1].data
        header = hdul[1].header

        # SoLEXS light curves have columns: TIME, COUNTS (or Time, Rate, Error)
        # Time is in seconds since epoch (usually 2000-01-01T00:00:00)
        col_names = [name.upper() for name in data.dtype.names]
        
        # Find time column
        if 'TIME' in col_names:
            time_col = data['TIME']
        elif 'Time' in data.dtype.names:
            time_col = data['Time']
        else:
            time_col = data.columns[0]
        
        # Find rate/counts column
        if 'RATE' in col_names:
            rate_col = data['RATE']
        elif 'COUNTS' in col_names:
            rate_col = data['COUNTS']
        elif 'Rate' in data.dtype.names:
            rate_col = data['Rate']
        else:
            rate_col = data.columns[1]
        
        # Find error column (optional)
        error_col = None
        if 'ERROR' in col_names:
            error_col = data['ERROR']
        elif 'Error' in data.dtype.names:
            error_col = data['Error']
        elif len(data.dtype.names) > 2:
            error_col = data.columns[2]

        # Try to get epoch from header
        if 'MJDREFI' in header and 'MJDREFF' in header:
            mjd_ref = header['MJDREFI'] + header['MJDREFF']
            from astropy.time import Time
            t_ref = Time(mjd_ref, format='mjd')
            times = t_ref + time_col * (1.0 / 86400.0)  # seconds to days
            times_pd = pd.DatetimeIndex(times.to_datetime())
        elif 'TSTART' in header and 'DATE-OBS' in header:
            t_start = pd.Timestamp(header['DATE-OBS'])
            times_pd = t_start + pd.to_timedelta(time_col, unit='s')
        else:
            # Fallback: assume seconds from 2024-01-01
            t_start = pd.Timestamp('2024-01-01')
            times_pd = t_start + pd.to_timedelta(time_col, unit='s')

        df = pd.DataFrame({
            'rate': rate_col.astype(float),
            'error': error_col.astype(float) if error_col is not None else np.nan,
        }, index=times_pd)

        df.index.name = 'time'

    return df


def load_solexs_directory(
    data_dir: str,
    instrument: str = 'SDD2',
    resample_to: str = '10s'
) -> pd.DataFrame:
    """
    Load all SoLEXS light curves from extracted directory structure.

    Expected structure:
        data_dir/
            AL1_SLX_L1_YYYYMMDD_v1.0/
                SDD2/
                    AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc

    Parameters
    ----------
    data_dir : str
        Root directory containing extracted SoLEXS files
    instrument : str
        'SDD2' (flare-optimized, recommended) or 'SDD1' (quiet Sun)
    resample_to : str
        Resample cadence (default '10s' for efficiency)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all loaded light curves
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all .lc files matching the instrument
    lc_files = sorted(data_path.glob(f"*/*/{instrument}/*.lc"))

    if not lc_files:
        # Try alternative naming patterns
        lc_files = sorted(data_path.glob(f"**/{instrument}/*.lc"))

    if not lc_files:
        raise FileNotFoundError(
            f"No .lc files found for instrument {instrument} in {data_dir}\n"
            f"Run 03_extract_solexs.py first to extract from ZIP files."
        )

    print(f"Found {len(lc_files)} light curve files for {instrument}")

    dfs = []
    for lc_file in lc_files:
        try:
            df = load_solexs_lc(str(lc_file))
            # Add source info
            df['source_file'] = lc_file.name
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {lc_file.name}: {e}")
            continue

    if not dfs:
        raise ValueError("No light curves could be loaded")

    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()

    # Remove duplicates (keep first)
    combined = combined[~combined.index.duplicated(keep='first')]

    # Resample to reduce size
    if resample_to:
        combined = combined.resample(resample_to).mean()
        combined = combined.dropna(subset=['rate'])

    return combined


def load_hel1os_directory(
    data_dir: str,
    resample_to: str = '1s'
) -> pd.DataFrame:
    """
    Load all HEL1OS light curves from extracted directory structure.

    Expected structure:
        data_dir/
            YYYY/
                MM/
                    DD/
                        HLS_YYYYMMDD_HHMMSS_*sec_lev1_V111/
                            cdte/
                                lightcurve_cdte1.fits

    Parameters
    ----------
    data_dir : str
        Root directory containing extracted HEL1OS files
    resample_to : str
        Resample cadence

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with rate, error, livetime
    """
    from astropy.io import fits

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all HEL1OS FITS files
    fits_files = sorted(data_path.glob("**/lightcurve_cdte1.fits"))

    if not fits_files:
        raise FileNotFoundError(
            f"No HEL1OS FITS files found in {data_dir}\n"
            f"Run download script first."
        )

    print(f"Found {len(fits_files)} HEL1OS light curve files")

    dfs = []
    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                data = hdul[1].data
                header = hdul[1].header

                time_col = data['Time'] if 'Time' in data.dtype.names else data['time']
                rate_col = data['Rate'] if 'Rate' in data.dtype.names else data['rate']

                # Build times
                if 'MJDREFI' in header:
                    mjd_ref = header['MJDREFI'] + header.get('MJDREFF', 0)
                    from astropy.time import Time
                    t_ref = Time(mjd_ref, format='mjd')
                    times = t_ref + time_col * (1.0 / 86400.0)
                    times_pd = pd.DatetimeIndex(times.to_datetime())
                else:
                    t_start = pd.Timestamp(header.get('DATE-OBS', '2024-01-01'))
                    times_pd = t_start + pd.to_timedelta(time_col, unit='s')

                df = pd.DataFrame({
                    'rate': rate_col.astype(float),
                    'error': (data['Error'] if 'Error' in data.dtype.names
                              else data.get('error', np.nan)).astype(float),
                }, index=times_pd)

                df['livetime'] = header.get('LIVETIME', np.nan)
                df['source_file'] = fits_file.name

            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {fits_file.name}: {e}")
            continue

    if not dfs:
        raise ValueError("No HEL1OS light curves could be loaded")

    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    if resample_to:
        combined = combined.resample(resample_to).mean()
        combined = combined.dropna(subset=['rate'])

    return combined


def validate_data(df: pd.DataFrame, source: str = 'goes') -> pd.DataFrame:
    """
    Validate and clean data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data
    source : str
        Data source ('goes', 'solexs', 'hel1os')

    Returns
    -------
    pd.DataFrame
        Validated data
    """
    df_clean = df.copy()

    flux_col = 'xrs_b_flux' if source == 'goes' else 'rate'

    if flux_col in df_clean.columns:
        # Remove negative values
        df_clean = df_clean[df_clean[flux_col] > 0]

        # Remove extreme outliers (fill values)
        upper_bound = 1e-2 if source == 'goes' else df_clean[flux_col].quantile(0.999)
        df_clean = df_clean[df_clean[flux_col] < upper_bound]

    # Detect and log gaps
    time_diff = df_clean.index.to_series().diff()
    max_gap = timedelta(hours=1)
    gaps = time_diff[time_diff > max_gap]

    if len(gaps) > 0:
        print(f"  Warning: Found {len(gaps)} gaps larger than {max_gap}")

    return df_clean


if __name__ == "__main__":
    print("PRADHAN Data Reader Test")
    print("=" * 50)

    # Test GOES loading
    try:
        goes = load_goes_parquet(r"C:\Users\Admin\aditya-flare-forecast\data\goes_historical")
        print(f"Loaded {len(goes):,} GOES records")
        print(f"  Columns: {goes.columns.tolist()}")
        print(f"  Time range: {goes.index.min()} to {goes.index.max()}")
        print(f"  XRS-B flux range: {goes['xrs_b_flux'].min():.2e} to {goes['xrs_b_flux'].max():.2e}")
    except Exception as e:
        print(f"GOES loading failed: {e}")
