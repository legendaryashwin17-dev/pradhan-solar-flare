"""Debug: test feature extraction on one sample"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timedelta
import xarray as xr

# Load GOES
goes_dir = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\pradhan-solar-flare-repo\data\goes18_2026'
files = sorted([f for f in os.listdir(goes_dir) if f.endswith('.nc')])
dfs = []
for f in files[:2]:
    fp = os.path.join(goes_dir, f)
    ds = xr.open_dataset(fp)
    df = pd.DataFrame({
        'xrsa': ds['xrsa_flux'].values,
        'xrsb': ds['xrsb_flux'].values,
    }, index=pd.to_datetime(ds.time.values))
    ds.close()
    dfs.append(df)
goes = pd.concat(dfs)
print(f'GOES range: {goes.index[0]} to {goes.index[-1]}')

# Load first HEL1OS
hel1os_dir = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\raw\hel1os'
fits_files = []
for root, dirs, files in os.walk(hel1os_dir):
    for f in files:
        if f.endswith('.fits') and 'cdte1' in f:
            fits_files.append(os.path.join(root, f))
    if len(fits_files) >= 1:
        break

fp = fits_files[0]
hdul = fits.open(fp)
hdr = hdul[0].header
mjd_start = hdr['MJDSTART']
t_start = Time(mjd_start, format='mjd').to_datetime()
t_stop = Time(hdr['MJDSTOP'], format='mjd').to_datetime()
print(f'HEL1OS: {t_start} to {t_stop}')

# Test at mid-point
feature_time = t_start + timedelta(hours=6)
print(f'Feature time: {feature_time}')

# GOES window
t_start_go = feature_time - timedelta(hours=1)
window = goes[(goes.index >= t_start_go) & (goes.index <= feature_time)]
print(f'GOES window: {len(window)} rows')

# HEL1OS windows
band_names = ['soft', 'med1', 'med2', 'hard', 'broad']
t_feature_mjd = Time(feature_time).mjd
t_start_mjd = t_feature_mjd - (1.0 / 24.0)

for i, band_name in enumerate(band_names):
    hdu_idx = i + 1
    data = hdul[hdu_idx].data
    mjd = data['MJD']
    ctr = data['CTR']
    mask = (mjd >= t_start_mjd) & (mjd <= t_feature_mjd)
    if mask.sum() > 0:
        print(f'  {band_name}: {mask.sum()} points, flux={ctr[mask].mean():.2e}')
    else:
        print(f'  {band_name}: 0 points! MJD range in file: {mjd.min():.6f} to {mjd.max():.6f}')
        print(f'    Looking for MJD range: {t_start_mjd:.6f} to {t_feature_mjd:.6f}')

hdul.close()
