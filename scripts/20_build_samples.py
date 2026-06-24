"""
PRADHAN Multi-Input Pipeline — Step 1: Sliding Window Sampler

Extracts features from HEL1OS + GOES-18 at overlapping timestamps.
Creates labeled samples for flare forecasting.

Each HEL1OS FITS file (~12h) produces multiple samples:
- Feature window: T-1h to T (1-hour lookback)
- Label window: T to T+6h (6-hour forecast horizon)
- Slide: every 1 hour across the 12h observation → ~6 samples per file

Total: 105 files × 6 samples = ~630 labeled samples
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
RAW_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\raw'
PROCESSED_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\processed\samples'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# HEL1OS Feature Extraction
# ============================================================

def extract_hel1os_features(fits_path, feature_time):
    """
    Extract HEL1OS features from a FITS file at a specific time.
    
    Features (per energy band):
    - flux: mean count rate in the 1-hour window ending at feature_time
    - flux_std: standard deviation of count rate
    - flux_max: maximum count rate
    - flux_deriv: average derivative (dF/dt) over the window
    - flux_ratio: ratio of hard (20-60 keV) to soft (5-20 keV) flux
    
    Returns dict of features or None if time not in file.
    """
    try:
        hdul = fits.open(fits_path)
        hdr = hdul[0].header
        
        # Get observation start time
        mjd_start = hdr.get('MJDSTART', None)
        if mjd_start is None:
            hdul.close()
            return None
        
        # Energy bands: HDU 1-5 (5-20, 20-30, 30-40, 40-60, 1.8-90 keV)
        band_names = ['soft', 'med1', 'med2', 'hard', 'broad']
        features = {}
        
        for i, band_name in enumerate(band_names):
            hdu_idx = i + 1
            if hdu_idx >= len(hdul):
                continue
                
            data = hdul[hdu_idx].data
            if data is None or len(data) == 0:
                continue
            
            mjd = data['MJD']
            ctr = data['CTR']
            
            # Convert feature_time (datetime) to MJD
            from astropy.time import Time
            t_feature = Time(feature_time).mjd
            
            # Find 1-hour window: [T-1h, T]
            t_start = t_feature - (1.0 / 24.0)  # 1 hour in days
            mask = (mjd >= t_start) & (mjd <= t_feature)
            
            if mask.sum() < 10:  # Need at least 10 data points
                continue
            
            window_flux = ctr[mask]
            
            features[f'hel1os_{band_name}_flux'] = np.mean(window_flux)
            features[f'hel1os_{band_name}_std'] = np.std(window_flux)
            features[f'hel1os_{band_name}_max'] = np.max(window_flux)
            
            # Derivative
            if len(window_flux) > 1:
                dt = (mjd[mask][-1] - mjd[mask][0]) * 86400  # seconds
                if dt > 0:
                    features[f'hel1os_{band_name}_deriv'] = (window_flux[-1] - window_flux[0]) / dt
                else:
                    features[f'hel1os_{band_name}_deriv'] = 0.0
            
        hdul.close()
        
        # Compute hard/soft ratio
        if 'hel1os_hard_flux' in features and 'hel1os_soft_flux' in features:
            if features['hel1os_soft_flux'] > 0:
                features['hel1os_hard_soft_ratio'] = features['hel1os_hard_flux'] / features['hel1os_soft_flux']
            else:
                features['hel1os_hard_soft_ratio'] = 0.0
        
        # PCA-like: total energy across all bands
        total_flux = sum(features.get(f'hel1os_{b}_flux', 0) for b in band_names)
        features['hel1os_total_flux'] = total_flux
        
        return features if features else None
        
    except Exception as e:
        return None


# ============================================================
# GOES Feature Extraction
# ============================================================

def load_goes_data(goes_dir):
    """Load all GOES-18 NetCDF files into a single DataFrame."""
    import xarray as xr
    
    files = sorted([f for f in os.listdir(goes_dir) if f.endswith('.nc')])
    print(f'Loading {len(files)} GOES-18 files...')
    
    dfs = []
    for f in files:
        try:
            fp = os.path.join(goes_dir, f)
            ds = xr.open_dataset(fp)
            
            df = pd.DataFrame({
                'xrsa': ds['xrsa_flux'].values,
                'xrsb': ds['xrsb_flux'].values,
            }, index=pd.to_datetime(ds.time.values))
            
            ds.close()
            dfs.append(df)
        except:
            continue
    
    if not dfs:
        return None
    
    goes = pd.concat(dfs)
    goes = goes.sort_index()
    goes = goes[~goes.index.duplicated(keep='first')]
    
    print(f'GOES loaded: {len(goes)} rows, {goes.index[0]} to {goes.index[-1]}')
    return goes


def extract_goes_features(goes_df, feature_time, lookback_hours=1):
    """
    Extract GOES features at a specific time.
    
    Features:
    - xrsa_flux: current 0.5-4Å flux
    - xrsb_flux: current 1-8Å flux
    - xrsa_60min_grad: gradient over last 60 minutes
    - xrsb_60min_grad: gradient over last 60 minutes
    - xrsb_3hr_std: standard deviation over last 3 hours
    - xrsb_baseline_ratio: current flux / min flux over last 24h
    """
    if goes_df is None or len(goes_df) == 0:
        return None
    
    try:
        # Get data in lookback window
        t_start = feature_time - timedelta(hours=lookback_hours)
        t_3hr = feature_time - timedelta(hours=3)
        t_24hr = feature_time - timedelta(hours=24)
        
        window = goes_df[(goes_df.index >= t_start) & (goes_df.index <= feature_time)]
        window_3hr = goes_df[(goes_df.index >= t_3hr) & (goes_df.index <= feature_time)]
        window_24hr = goes_df[(goes_df.index >= t_24hr) & (goes_df.index <= feature_time)]
        
        if len(window) < 10:
            return None
        
        features = {}
        
        # Current flux
        features['goes_xrsa'] = window['xrsa'].iloc[-1]
        features['goes_xrsb'] = window['xrsb'].iloc[-1]
        
        # 60-minute gradient
        if len(window) > 1:
            dt_sec = (window.index[-1] - window.index[0]).total_seconds()
            if dt_sec > 0:
                features['goes_xrsa_60min_grad'] = (window['xrsa'].iloc[-1] - window['xrsa'].iloc[0]) / dt_sec
                features['goes_xrsb_60min_grad'] = (window['xrsb'].iloc[-1] - window['xrsb'].iloc[0]) / dt_sec
            else:
                features['goes_xrsa_60min_grad'] = 0.0
                features['goes_xrsb_60min_grad'] = 0.0
        
        # 3-hour std
        if len(window_3hr) > 10:
            features['goes_xrsb_3hr_std'] = window_3hr['xrsb'].std()
        else:
            features['goes_xrsb_3hr_std'] = 0.0
        
        # Baseline ratio (current / 24h min)
        if len(window_24hr) > 0 and window_24hr['xrsb'].min() > 0:
            features['goes_xrsb_baseline'] = features['goes_xrsb'] / window_24hr['xrsb'].min()
        else:
            features['goes_xrsb_baseline'] = 1.0
        
        return features
        
    except Exception as e:
        return None


# ============================================================
# Label Extraction
# ============================================================

def extract_label(goes_df, label_start, label_end):
    """
    Extract flare label from GOES data.
    
    Label = 1 if max xrsb_flux in [label_start, label_end] >= C-class threshold (1e-6)
    """
    if goes_df is None:
        return 0
    
    window = goes_df[(goes_df.index >= label_start) & (goes_df.index <= label_end)]
    
    if len(window) == 0:
        return 0
    
    max_flux = window['xrsb'].max()
    
    # C-class threshold: 1e-6 W/m²
    return 1 if max_flux >= 1e-6 else 0


# ============================================================
# Main Pipeline
# ============================================================

def build_samples():
    """Build all samples from HEL1OS + GOES data."""
    
    # Load GOES data
    goes_dir = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\pradhan-solar-flare-repo\data\goes18_2026'
    goes_df = load_goes_data(goes_dir)
    
    if goes_df is None:
        print("ERROR: Could not load GOES data")
        return
    
    # Find all HEL1OS FITS files
    hel1os_dir = os.path.join(RAW_DIR, 'hel1os')
    fits_files = []
    for root, dirs, files in os.walk(hel1os_dir):
        for f in files:
            if f.endswith('.fits') and 'cdte1' in f:
                fits_files.append(os.path.join(root, f))
    
    print(f'Found {len(fits_files)} HEL1OS FITS files')
    
    samples = []
    sample_id = 0
    
    for i, fits_path in enumerate(fits_files):
        if (i + 1) % 20 == 0:
            print(f'  Processing file {i+1}/{len(fits_files)}...')
        
        try:
            # Get observation time range
            hdul = fits.open(fits_path)
            hdr = hdul[0].header
            from astropy.time import Time
            mjd_start = hdr['MJDSTART']
            mjd_stop = hdr['MJDSTOP']
            t_start = Time(mjd_start, format='mjd').to_datetime()
            t_stop = Time(mjd_stop, format='mjd').to_datetime()
            hdul.close()
            
            # Extract AR number from path (if available)
            # HLS_YYYYMMDD_... format
            dirname = os.path.basename(os.path.dirname(os.path.dirname(fits_path)))
            ar_number = dirname  # Use observation ID as group identifier
            
            # Slide window across the observation
            # Each sample: features at T, label from T to T+6h
            # Window: T-1h to T for features, T to T+6h for label
            
            obs_duration_hours = (t_stop - t_start).total_seconds() / 3600
            slide_interval = 1.0  # 1 hour between samples
            forecast_horizon = 6.0  # 6 hours ahead
            feature_lookback = 1.0  # 1 hour lookback
            
            # Calculate valid start times
            # Need: feature_lookback before start, forecast_horizon after end
            valid_start = t_start + timedelta(hours=feature_lookback)
            valid_end = t_stop - timedelta(hours=forecast_horizon)
            
            if valid_start >= valid_end:
                continue
            
            # Generate time windows
            current_time = valid_start
            while current_time <= valid_end:
                feature_time = current_time
                label_start = current_time
                label_end = current_time + timedelta(hours=forecast_horizon)
                
                # Extract features
                hel1os_feats = extract_hel1os_features(fits_path, feature_time)
                goes_feats = extract_goes_features(goes_df, feature_time)
                
                if hel1os_feats is None or goes_feats is None:
                    current_time += timedelta(hours=slide_interval)
                    continue
                
                # Extract label
                label = extract_label(goes_df, label_start, label_end)
                
                # Combine all features
                sample = {
                    'sample_id': sample_id,
                    'ar_number': ar_number,
                    'feature_time': feature_time,
                    'label_time_start': label_start,
                    'label_time_end': label_end,
                    'label': label,
                    'fits_file': os.path.basename(fits_path),
                }
                sample.update(hel1os_feats)
                sample.update(goes_feats)
                
                samples.append(sample)
                sample_id += 1
                
                current_time += timedelta(hours=slide_interval)
                
        except Exception as e:
            continue
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Save
    output_path = os.path.join(PROCESSED_DIR, 'hel1os_goes_samples.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f'\n{"="*60}')
    print(f'SAMPLES BUILT')
    print(f'{"="*60}')
    print(f'Total samples: {len(df)}')
    print(f'Positive (flare): {df["label"].sum()} ({df["label"].mean()*100:.1f}%)')
    print(f'Negative (no flare): {(df["label"]==0).sum()} ({(1-df["label"].mean())*100:.1f}%)')
    print(f'Unique ARs: {df["ar_number"].nunique()}')
    print(f'Features: {[c for c in df.columns if c.startswith("hel1os_") or c.startswith("goes_")]}')
    print(f'\nSaved to: {output_path}')
    
    return df


if __name__ == '__main__':
    df = build_samples()
