#!/usr/bin/env python3
"""
PRADHAN Historical GOES XRS Data Downloader
=============================================

Builds a multi-year GOES XRS dataset for cross-solar-cycle evaluation.

Data sources:
  1. GOES-18 NetCDF (data/goes18_2026/) — 2026, 1-sec cadence
  2. GOES parquet (data/goes/goes18_2026.parquet) — 2026, pre-processed
  3. GOES 2024 parquet (data/goes/goes_2024.parquet) — 2024
  4. NOAA SWPC API — last 7 days, 1-min cadence

Solar cycle coverage:
  - Solar Cycle 25: 2024-2026 (GOES-18/16/17)
  - Solar Cycle 24 peak: ~2014 (needs separate download)
  - Solar Cycle 23 peak: ~2001 (needs separate download)

For a hackathon, we work with what we have:
  - 2 months of 1-sec GOES-18 (April-June 2026)
  - 2 days of GOES 2024 (June 2024)
  - Last 7 days real-time (NOAA API)

Usage:
    python scripts/33_download_historical_goes.py           # Build combined dataset
    python scripts/33_download_historical_goes.py --extract  # Extract features too
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
RAW_DIR = os.path.join(WORKSPACE, 'data', 'goes18_2026')
PARQUET_2026 = os.path.join(WORKSPACE, 'data', 'goes', 'goes18_2026.parquet')
PARQUET_2024 = os.path.join(WORKSPACE, 'data', 'goes', 'goes_2024.parquet')
OUTPUT_DIR = os.path.join(WORKSPACE, 'data', 'historical')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_goes18_2026_from_parquet():
    """Load pre-processed GOES-18 2026 parquet."""
    if not os.path.exists(PARQUET_2026):
        print(f'  Not found: {PARQUET_2026}')
        return None
    
    print(f'  Loading {PARQUET_2026}...')
    df = pd.read_parquet(PARQUET_2026)
    
    # Fix index — current file has integer index, needs datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.set_index('time').sort_index()
    
    # Rename columns to standard
    df = df.rename(columns={'flux_a': 'xrsa', 'flux_b': 'xrsb'})
    
    # Remove bad values
    df = df[(df['xrsa'] > 0) & (df['xrsb'] > 0)]
    
    print(f'  Loaded: {len(df):,} records')
    print(f'  Range: {df.index[0]} to {df.index[-1]}')
    return df


def load_goes_2024():
    """Load GOES 2024 parquet."""
    if not os.path.exists(PARQUET_2024):
        print(f'  Not found: {PARQUET_2024}')
        return None
    
    print(f'  Loading {PARQUET_2024}...')
    df = pd.read_parquet(PARQUET_2024)
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.set_index('time').sort_index()
    
    # Remove bad values
    df = df[(df['xrsa'] > 0) & (df['xrsb'] > 0)]
    
    print(f'  Loaded: {len(df):,} records')
    print(f'  Range: {df.index[0]} to {df.index[-1]}')
    return df


def load_goes18_from_netcdf():
    """Load GOES-18 NetCDF files directly."""
    nc_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.nc')))
    if not nc_files:
        print(f'  No NetCDF files in {RAW_DIR}')
        return None
    
    print(f'  Found {len(nc_files)} NetCDF files')
    
    try:
        import xarray as xr
    except ImportError:
        print('  xarray not installed, skipping NetCDF')
        return None
    
    print(f'  Loading NetCDF files (this may take a while)...')
    ds = xr.open_mfdataset(nc_files, combine='by_coords')
    
    # Extract XRS channels
    # GOES-18 NetCDF variables: xrayflux_a (0.5-4A), xrayflux_b (1-8A)
    var_names = list(ds.data_vars)
    print(f'  Variables: {var_names[:10]}')
    
    # Try common variable names
    xrsa_var = None
    xrsb_var = None
    for v in var_names:
        vl = v.lower()
        if 'flux' in vl and ('a' in vl or '0.5' in vl or 'soft' in vl):
            xrsa_var = v
        if 'flux' in vl and ('b' in vl or '1' in vl or 'hard' in vl):
            xrsb_var = v
    
    if xrsa_var is None or xrsb_var is None:
        # Fallback: use first two flux variables
        flux_vars = [v for v in var_names if 'flux' in v.lower()]
        if len(flux_vars) >= 2:
            xrsa_var, xrsb_var = flux_vars[0], flux_vars[1]
        else:
            print(f'  Cannot identify XRS channels from: {var_names}')
            return None
    
    print(f'  XRSA: {xrsa_var}, XRSB: {xrsb_var}')
    
    xrsa = ds[xrsa_var].to_pandas()
    xrsb = ds[xrsb_var].to_pandas()
    
    df = pd.DataFrame({'xrsa': xrsa, 'xrsb': xrsb})
    df = df.dropna()
    df = df[(df['xrsa'] > 0) & (df['xrsb'] > 0)]
    df = df.sort_index()
    
    print(f'  Loaded: {len(df):,} records')
    print(f'  Range: {df.index[0]} to {df.index[-1]}')
    return df


def downsample_to_1min(df):
    """Downsample high-cadence data to 1-minute median."""
    print(f'  Downsampling {len(df):,} records to 1-minute cadence...')
    df_1min = df.resample('1min').median()
    df_1min = df_1min.dropna()
    print(f'  After downsampling: {len(df_1min):,} records')
    return df_1min


def extract_cross_cycle_features(goes_df, window_hours=24):
    """
    Extract features at 1-hour intervals for cross-cycle evaluation.
    
    Returns DataFrame with features at each hour.
    """
    print(f'\nExtracting features at 1-hour intervals...')
    print(f'  Total time range: {goes_df.index[0]} to {goes_df.index[-1]}')
    
    # Get all hours
    start = goes_df.index[0]
    end = goes_df.index[-1]
    hours = pd.date_range(start, end, freq='1h')
    
    records = []
    for i, t in enumerate(hours):
        # Get window
        t_start = t - pd.Timedelta(hours=window_hours)
        window = goes_df[(goes_df.index >= t_start) & (goes_df.index <= t)]
        
        if len(window) < 10:
            continue
        
        xrsb_now = float(window['xrsb'].iloc[-1])
        xrsa_now = float(window['xrsa'].iloc[-1])
        
        if xrsb_now <= 0 or xrsa_now <= 0:
            continue
        
        # Extract 8 GOES features
        feats = {'time': t}
        feats['goes_log_xrsa'] = float(np.log10(xrsa_now))
        feats['goes_log_xrsb'] = float(np.log10(xrsb_now))
        
        w24 = goes_df[(goes_df.index >= t_start) & (goes_df.index <= t)]
        if len(w24) > 0 and w24['xrsb'].min() > 0:
            feats['goes_xrsb_baseline'] = float(xrsb_now / w24['xrsb'].min())
        else:
            feats['goes_xrsb_baseline'] = 1.0
        
        if len(window) > 1:
            dt = (window.index[-1] - window.index[0]).total_seconds() / 3600.0
            if dt > 0:
                log_start = np.log10(max(window['xrsb'].iloc[0], 1e-12))
                log_end = np.log10(max(window['xrsb'].iloc[-1], 1e-12))
                feats['goes_xrsb_log_grad'] = float((log_end - log_start) / dt)
            else:
                feats['goes_xrsb_log_grad'] = 0.0
        else:
            feats['goes_xrsb_log_grad'] = 0.0
        
        w3 = goes_df[(goes_df.index >= t - pd.Timedelta(hours=3)) & (goes_df.index <= t)]
        if len(w3) > 10 and (w3['xrsb'] > 0).all():
            log_xrsb_3h = np.log10(w3['xrsb'].clip(lower=1e-12))
            feats['goes_xrsb_log_std'] = float(log_xrsb_3h.std())
            feats['goes_xrsb_log_mean'] = float(log_xrsb_3h.mean())
        else:
            feats['goes_xrsb_log_std'] = 0.0
            feats['goes_xrsb_log_mean'] = float(np.log10(xrsb_now))
        
        feats['goes_xrsa_xrsb_ratio'] = float(xrsb_now / xrsa_now) if xrsa_now > 0 else 0.0
        
        if len(w24) > 10 and w24['xrsb'].std() > 0:
            log_24h = np.log10(w24['xrsb'].clip(lower=1e-12))
            feats['goes_xrsb_log_zscore'] = float(
                (np.log10(xrsb_now) - log_24h.mean()) / log_24h.std()
            )
        else:
            feats['goes_xrsb_log_zscore'] = 0.0
        
        # Label: was there an M+ flare in the next 24h?
        next_24h = goes_df[(goes_df.index > t) & (goes_df.index <= t + pd.Timedelta(hours=24))]
        if len(next_24h) > 0:
            max_flux = next_24h['xrsb'].max()
            feats['label_mflare'] = 1 if max_flux >= 1e-5 else 0
            feats['label_cflare'] = 1 if max_flux >= 1e-6 else 0
            feats['max_flux_24h'] = float(max_flux)
        else:
            feats['label_mflare'] = 0
            feats['label_cflare'] = 0
            feats['max_flux_24h'] = float(xrsb_now)
        
        records.append(feats)
        
        if (i + 1) % 500 == 0:
            print(f'  Processed {i + 1}/{len(hours)} hours...')
    
    result = pd.DataFrame(records)
    result = result.set_index('time')
    print(f'  Extracted {len(result)} hourly feature vectors')
    return result


def main():
    parser = argparse.ArgumentParser(description='Build historical GOES dataset')
    parser.add_argument('--extract', action='store_true',
                       help='Also extract features for ML')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Downsample to N-minute cadence (default: 1)')
    args = parser.parse_args()
    
    print('=' * 60)
    print('PRADHAN Historical GOES XRS Data Builder')
    print('=' * 60)
    
    # Load all available data
    print('\n[1/4] Loading GOES-18 2026 (parquet)...')
    goes_2026 = load_goes18_2026_from_parquet()
    
    print('\n[2/4] Loading GOES 2024...')
    goes_2024 = load_goes_2024()
    
    print('\n[3/4] Loading GOES-18 from NetCDF (if parquet missing)...')
    goes_nc = None
    if goes_2026 is None:
        goes_nc = load_goes18_from_netcdf()
    
    # Use best available GOES-18 data
    goes_18 = goes_2026 if goes_2026 is not None else goes_nc
    
    # Combine all data
    print('\n[4/4] Combining datasets...')
    all_data = []
    if goes_2024 is not None:
        goes_2024['source'] = 'goes_2024'
        all_data.append(goes_2024)
        print(f'  GOES 2024: {len(goes_2024):,} records')
    if goes_18 is not None:
        goes_18['source'] = 'goes_18_2026'
        all_data.append(goes_18)
        print(f'  GOES-18 2026: {len(goes_18):,} records')
    
    if not all_data:
        print('\nNo data available!')
        return
    
    # Downsample each source independently before combining
    if args.downsample > 1:
        parts = []
        if goes_2024 is not None:
            p = goes_2024.drop(columns=['source'], errors='ignore').resample(f'{args.downsample}min').median().dropna()
            parts.append(p)
            print(f'  GOES 2024 after {args.downsample}-min: {len(p):,} records')
        if goes_18 is not None:
            p = goes_18.drop(columns=['source'], errors='ignore').resample(f'{args.downsample}min').median().dropna()
            parts.append(p)
            print(f'  GOES-18 2026 after {args.downsample}-min: {len(p):,} records')
        combined = pd.concat(parts).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
    else:
        combined = pd.concat(all_data).sort_index()
        combined = combined.drop(columns=['source'], errors='ignore')
        combined = combined[~combined.index.duplicated(keep='last')]
    
    print(f'\nCombined dataset: {len(combined):,} records')
    print(f'Time range: {combined.index[0]} to {combined.index[-1]}')
    
    # Save combined raw data
    out_path = os.path.join(OUTPUT_DIR, 'goes_combined_raw.parquet')
    combined.to_parquet(out_path)
    print(f'\nSaved: {out_path}')
    
    # Extract features
    if args.extract:
        features = extract_cross_cycle_features(combined)
        feat_path = os.path.join(OUTPUT_DIR, 'goes_cross_cycle_features.parquet')
        features.to_parquet(feat_path)
        print(f'\nSaved features: {feat_path}')
        
        # Summary statistics
        print(f'\n{"=" * 60}')
        print('FEATURE SUMMARY')
        print(f'{"=" * 60}')
        flare_mask_m = features['label_mflare'] == 1
        flare_mask_c = features['label_cflare'] == 1
        print(f'Total hourly samples: {len(features)}')
        print(f'M+ flare hours: {flare_mask_m.sum()} ({flare_mask_m.mean()*100:.1f}%)')
        print(f'C+ flare hours: {flare_mask_c.sum()} ({flare_mask_c.mean()*100:.1f}%)')
        print(f'Quiet hours: {(~flare_mask_c).sum()}')
        print(f'\nFeature columns: {[c for c in features.columns if c.startswith("goes_")]}')
        print(f'\nDate range: {features.index[0]} to {features.index[-1]}')
    
    # Summary
    print(f'\n{"=" * 60}')
    print('HISTORICAL DATASET SUMMARY')
    print(f'{"=" * 60}')
    print(f'Total records: {len(combined):,}')
    print(f'Date range: {combined.index[0]} to {combined.index[-1]}')
    
    # Flare statistics
    c_count = ((combined['xrsb'] >= 1e-6) & (combined['xrsb'] < 1e-5)).sum()
    m_count = ((combined['xrsb'] >= 1e-5) & (combined['xrsb'] < 1e-4)).sum()
    x_count = (combined['xrsb'] >= 1e-4).sum()
    print(f'B/A-class minutes: {len(combined) - c_count - m_count - x_count:,}')
    print(f'C-class minutes: {c_count:,}')
    print(f'M-class minutes: {m_count:,}')
    print(f'X-class minutes: {x_count:,}')


if __name__ == '__main__':
    main()
