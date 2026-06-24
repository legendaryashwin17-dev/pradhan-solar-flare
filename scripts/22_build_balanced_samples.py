#!/usr/bin/env python3
"""
PRADHAN Multi-Input Pipeline v2 — Balanced Sampling

Fix: HEL1OS only observes during active periods (87% positive rate in v1).
Solution: For each flare event window, create a matched QUIET window
(no flare in [T, T+6h]) from GOES-18 periods without any C+ flare.

This gives true balanced binary classification.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timedelta
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\raw'
PROCESSED_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\processed\samples'
os.makedirs(PROCESSED_DIR, exist_ok=True)

FLARE_THRESHOLD = 1e-6  # C-class
FORECAST_HORIZON_HOURS = 6
LOOKBACK_HOURS = 1
SLIDE_HOURS = 2


def load_goes_data(goes_dir):
    """Load all GOES-18 NetCDF files."""
    files = sorted([f for f in os.listdir(goes_dir) if f.endswith('.nc')])
    print(f'Loading {len(files)} GOES-18 files...')
    dfs = []
    for f in files:
        try:
            ds = xr.open_dataset(os.path.join(goes_dir, f))
            df = pd.DataFrame({
                'xrsa': ds['xrsa_flux'].values,
                'xrsb': ds['xrsb_flux'].values,
            }, index=pd.to_datetime(ds.time.values))
            ds.close()
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    goes = pd.concat(dfs).sort_index()
    goes = goes[~goes.index.duplicated(keep='first')]
    goes = goes.dropna()
    print(f'GOES loaded: {len(goes)} rows, {goes.index[0]} to {goes.index[-1]}')
    return goes


def find_flare_and_quiet_windows(goes_df):
    """
    Find flare events (max xrsb >= C-class) and matched quiet windows.

    Flare window: [peak_time - 30min, peak_time + 6h]
    Quiet window: 6h period with NO flare activity (max xrsb < 1e-7 B-class background)
    """
    xrsb = goes_df['xrsb']

    # Find all timestamps above C-class (these are flare times)
    flare_mask = xrsb >= FLARE_THRESHOLD
    flare_indices = np.where(flare_mask)[0]

    if len(flare_indices) == 0:
        return [], []

    # Group consecutive flare indices into events
    events = []
    current_event = [flare_indices[0]]
    for i in range(1, len(flare_indices)):
        # If gap > 30 min, new event
        t_curr = xrsb.index[flare_indices[i]]
        t_prev = xrsb.index[current_event[-1]]
        if (t_curr - t_prev).total_seconds() > 1800:
            events.append(current_event)
            current_event = [flare_indices[i]]
        else:
            current_event.append(flare_indices[i])
    events.append(current_event)

    print(f'Found {len(events)} flare events')

    # Extract peak times
    flare_peaks = []
    for ev in events:
        peak_idx = ev[np.argmax(xrsb.iloc[ev].values)]
        peak_time = xrsb.index[peak_idx]
        peak_flux = xrsb.iloc[peak_idx]
        flare_peaks.append((peak_time, peak_flux))

    # Now find quiet windows: 6h blocks where max xrsb < 1e-6 (below C-class)
    # and at least 2h away from any C+ flare peak
    quiet_threshold = 1e-6
    min_gap_from_flare = timedelta(hours=2)

    # Build a "no-flare" mask
    safe_mask = xrsb < quiet_threshold
    # Exclude 12h around any flare
    flare_times = [t for t, _ in flare_peaks]
    for ft in flare_times:
        exclude_start = ft - min_gap_from_flare
        exclude_end = ft + min_gap_from_flare
        safe_mask.loc[exclude_start:exclude_end] = False

    # Find contiguous safe regions >= 6h
    safe_runs = []
    in_run = False
    run_start = None
    for ts, val in safe_mask.items():
        if val and not in_run:
            run_start = ts
            in_run = True
        elif not val and in_run:
            safe_runs.append((run_start, ts))
            in_run = False
    if in_run:
        safe_runs.append((run_start, safe_mask.index[-1]))

    long_safe = [(s, e) for s, e in safe_runs if (e - s).total_seconds() >= 6 * 3600]
    print(f'Found {len(long_safe)} quiet windows >= 6h')

    # Sample N quiet windows to roughly match flare count
    n_quiet_target = max(len(flare_peaks), 200)
    np.random.seed(42)
    if len(long_safe) > n_quiet_target:
        idx = np.random.choice(len(long_safe), n_quiet_target, replace=False)
        sampled_quiet = [long_safe[i] for i in idx]
    else:
        sampled_quiet = long_safe

    return flare_peaks, sampled_quiet


def extract_hel1os_features(fits_path, feature_time):
    """Extract HEL1OS features from FITS at feature_time."""
    try:
        hdul = fits.open(fits_path)
        hdr = hdul[0].header
        mjd_start = hdr.get('MJDSTART')
        mjd_stop = hdr.get('MJDSTOP')
        if mjd_start is None:
            hdul.close()
            return None

        t_obs_start = Time(mjd_start, format='mjd').to_datetime()
        t_obs_stop = Time(mjd_stop, format='mjd').to_datetime()

        # Skip if feature_time outside observation
        if feature_time < t_obs_start + timedelta(hours=LOOKBACK_HOURS):
            hdul.close()
            return None
        if feature_time > t_obs_stop - timedelta(hours=FORECAST_HORIZON_HOURS):
            hdul.close()
            return None

        t_feature_mjd = Time(feature_time).mjd
        t_start_mjd = t_feature_mjd - (LOOKBACK_HOURS / 24.0)

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
            mask = (mjd >= t_start_mjd) & (mjd <= t_feature_mjd)
            if mask.sum() < 10:
                continue
            wf = ctr[mask]
            features[f'hel1os_{band_name}_flux'] = float(np.mean(wf))
            features[f'hel1os_{band_name}_std'] = float(np.std(wf))
            features[f'hel1os_{band_name}_max'] = float(np.max(wf))
            if len(wf) > 1:
                dt = (mjd[mask][-1] - mjd[mask][0]) * 86400
                features[f'hel1os_{band_name}_deriv'] = float((wf[-1] - wf[0]) / dt) if dt > 0 else 0.0
        hdul.close()

        if 'hel1os_hard_flux' in features and 'hel1os_soft_flux' in features:
            if features['hel1os_soft_flux'] > 0:
                features['hel1os_hard_soft_ratio'] = features['hel1os_hard_flux'] / features['hel1os_soft_flux']
            else:
                features['hel1os_hard_soft_ratio'] = 0.0

        features['hel1os_total_flux'] = sum(features.get(f'hel1os_{b}_flux', 0) for b in band_names)
        return features if features else None
    except Exception:
        return None


def extract_goes_features(goes_df, feature_time):
    """
    Extract GOES features at feature_time.

    Features are designed to be discriminative across the wide dynamic range
    of XRS flux (1e-9 to 1e-3 W/m²). Raw flux values are log-transformed;
    gradient and variability are relative (normalized).

    Goes features (8):
      goes_log_xrsa       - log10(XRS-A current flux)
      goes_log_xrsb       - log10(XRS-B current flux)
      goes_xrsb_baseline  - current XRS-B / 24h minimum (background ratio)
      goes_xrsb_log_grad  - log-flux gradient over 60min (decade/hour)
      goes_xrsb_log_std   - log-flux std over 3h (variability in log space)
      goes_xrsb_log_mean  - log-flux mean over 3h
      goes_xrsa_xrsb_ratio - current XRS-B / XRS-A ratio
      goes_xrsb_log_zscore - log-flux z-score over 24h
    """
    try:
        t1 = feature_time - timedelta(hours=LOOKBACK_HOURS)
        t3 = feature_time - timedelta(hours=3)
        t24 = feature_time - timedelta(hours=24)
        window = goes_df[(goes_df.index >= t1) & (goes_df.index <= feature_time)]
        if len(window) < 10:
            return None

        xrsb_now = float(window['xrsb'].iloc[-1])
        xrsa_now = float(window['xrsa'].iloc[-1])
        if xrsb_now <= 0 or xrsa_now <= 0:
            return None

        feats = {}

        # Log-space current flux (handles dynamic range)
        feats['goes_log_xrsa'] = float(np.log10(xrsa_now))
        feats['goes_log_xrsb'] = float(np.log10(xrsb_now))

        # Baseline ratio (current vs 24h min) — this is a discriminative feature
        w24 = goes_df[(goes_df.index >= t24) & (goes_df.index <= feature_time)]
        if len(w24) > 0 and w24['xrsb'].min() > 0:
            feats['goes_xrsb_baseline'] = float(xrsb_now / w24['xrsb'].min())
        else:
            feats['goes_xrsb_baseline'] = 1.0

        # Log-space gradient over 60 min (decade/hour)
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

        # Log-space std and mean over 3h
        w3 = goes_df[(goes_df.index >= t3) & (goes_df.index <= feature_time)]
        if len(w3) > 10 and (w3['xrsb'] > 0).all():
            log_xrsb_3h = np.log10(w3['xrsb'].clip(lower=1e-12))
            feats['goes_xrsb_log_std'] = float(log_xrsb_3h.std())
            feats['goes_xrsb_log_mean'] = float(log_xrsb_3h.mean())
        else:
            feats['goes_xrsb_log_std'] = 0.0
            feats['goes_xrsb_log_mean'] = float(np.log10(xrsb_now))

        # XRS-B / XRS-A ratio (spectral hardness proxy)
        feats['goes_xrsa_xrsb_ratio'] = float(xrsb_now / xrsa_now) if xrsa_now > 0 else 0.0

        # Z-score over 24h (how many σ above background)
        if len(w24) > 10 and w24['xrsb'].std() > 0:
            feats['goes_xrsb_log_zscore'] = float(
                (np.log10(xrsb_now) - np.log10(w24['xrsb'].clip(lower=1e-12)).mean())
                / log_xrsb_3h.std() if False else
                (np.log10(xrsb_now) - np.mean(np.log10(w24['xrsb'].clip(lower=1e-12))))
                / np.std(np.log10(w24['xrsb'].clip(lower=1e-12)))
            )
        else:
            feats['goes_xrsb_log_zscore'] = 0.0

        return feats
    except Exception:
        return None


def build_balanced_samples():
    """Build balanced flare + quiet samples."""
    goes_dir = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\pradhan-solar-flare-repo\data\goes18_2026'
    goes_df = load_goes_data(goes_dir)
    if goes_df is None:
        return

    flare_peaks, quiet_windows = find_flare_and_quiet_windows(goes_df)
    print(f'\nFlare events: {len(flare_peaks)}')
    print(f'Quiet windows: {len(quiet_windows)}')

    # Find HEL1OS files
    hel1os_dir = os.path.join(RAW_DIR, 'hel1os')
    fits_files = []
    for root, _, files in os.walk(hel1os_dir):
        for f in files:
            if f.endswith('.fits') and 'cdte1' in f:
                fits_files.append(os.path.join(root, f))
    print(f'HEL1OS FITS: {len(fits_files)}')

    # Pre-load HEL1OS observation windows
    hel1os_obs = []
    for fp in fits_files:
        try:
            hdul = fits.open(fp)
            mjd_s = hdul[0].header.get('MJDSTART')
            mjd_e = hdul[0].header.get('MJDSTOP')
            hdul.close()
            if mjd_s and mjd_e:
                t_s = Time(mjd_s, format='mjd').to_datetime()
                t_e = Time(mjd_e, format='mjd').to_datetime()
                hel1os_obs.append((fp, t_s, t_e))
        except Exception:
            continue
    print(f'HEL1OS obs loaded: {len(hel1os_obs)}')

    def find_hel1os_for_time(feature_time):
        """Find a HEL1OS file whose observation covers [feature_time - 1h, feature_time + 6h]."""
        earliest = feature_time - timedelta(hours=LOOKBACK_HOURS)
        latest = feature_time + timedelta(hours=FORECAST_HORIZON_HOURS)
        for fp, t_s, t_e in hel1os_obs:
            if t_s <= earliest and t_e >= latest:
                return fp
        return None

    samples = []

    # ─── FLARE WINDOWS (label=1) ───────────────────────────────────────────
    print('\nBuilding flare windows...')
    for peak_time, peak_flux in flare_peaks:
        feature_time = peak_time - timedelta(minutes=30)
        fp = find_hel1os_for_time(feature_time)
        if fp is None:
            continue
        h1 = extract_hel1os_features(fp, feature_time)
        g = extract_goes_features(goes_df, feature_time)
        if h1 is None or g is None:
            continue
        sample = {
            'sample_id': len(samples),
            'ar_number': f'FLARE_{peak_time.strftime("%Y%m%d")}',
            'feature_time': feature_time,
            'label': 1,
            'peak_flux': float(peak_flux),
            'source': 'flare',
            'fits_file': os.path.basename(fp),
        }
        sample.update(h1)
        sample.update(g)
        samples.append(sample)
    flare_n = sum(1 for s in samples if s['label'] == 1)
    print(f'  Flare samples (HEL1OS-covered): {flare_n}')

    # ─── QUIET WINDOWS WITH HEL1OS COVERAGE (label=0) ──────────────────────
    print('Building quiet windows with HEL1OS coverage...')
    for q_start, q_end in quiet_windows:
        feature_time = q_start + timedelta(hours=1)
        if feature_time + timedelta(hours=FORECAST_HORIZON_HOURS) > q_end:
            continue
        fp = find_hel1os_for_time(feature_time)
        if fp is None:
            continue
        h1 = extract_hel1os_features(fp, feature_time)
        g = extract_goes_features(goes_df, feature_time)
        if h1 is None or g is None:
            continue
        sample = {
            'sample_id': len(samples),
            'ar_number': f'QUIET_H1_{feature_time.strftime("%Y%m%d")}',
            'feature_time': feature_time,
            'label': 0,
            'peak_flux': float(goes_df.loc[feature_time:feature_time + timedelta(hours=6), 'xrsb'].max()),
            'source': 'quiet_hel1os',
            'fits_file': os.path.basename(fp),
        }
        sample.update(h1)
        sample.update(g)
        samples.append(sample)
    quiet_h1 = sum(1 for s in samples if s['label'] == 0 and s.get('source') == 'quiet_hel1os')
    print(f'  Quiet HEL1OS-covered samples: {quiet_h1}')

    # ─── SYNTHETIC QUIET WINDOWS FROM PURE GOES (label=0) ──────────────────
    # Sample from GOES-only periods (no HEL1OS coverage) to balance dataset
    print('Building synthetic quiet windows (GOES-only features)...')
    target_quiet = flare_n  # match flare count
    np.random.seed(42)
    quiet_pool = []
    for q_start, q_end in quiet_windows:
        # Slide through the quiet window
        t = q_start + timedelta(hours=1)
        while t + timedelta(hours=FORECAST_HORIZON_HOURS) <= q_end:
            quiet_pool.append(t)
            t += timedelta(hours=SLIDE_HOURS)
    np.random.shuffle(quiet_pool)

    for feature_time in quiet_pool:
        if sum(1 for s in samples if s['label'] == 0) >= target_quiet:
            break
        fp = find_hel1os_for_time(feature_time)
        if fp is not None:
            continue  # Already handled in HEL1OS path
        g = extract_goes_features(goes_df, feature_time)
        if g is None:
            continue
        sample = {
            'sample_id': len(samples),
            'ar_number': f'QUIET_GOES_{feature_time.strftime("%Y%m%d")}',
            'feature_time': feature_time,
            'label': 0,
            'peak_flux': float(goes_df.loc[feature_time:feature_time + timedelta(hours=6), 'xrsb'].max()),
            'source': 'quiet_goes_only',
            'fits_file': None,
        }
        sample.update(g)
        # Mark HEL1OS features as NaN (will be filled with median)
        for key in ['hel1os_soft_flux', 'hel1os_soft_std', 'hel1os_soft_max', 'hel1os_soft_deriv',
                    'hel1os_med1_flux', 'hel1os_med1_std', 'hel1os_med1_max', 'hel1os_med1_deriv',
                    'hel1os_med2_flux', 'hel1os_med2_std', 'hel1os_med2_max', 'hel1os_med2_deriv',
                    'hel1os_hard_flux', 'hel1os_hard_std', 'hel1os_hard_max', 'hel1os_hard_deriv',
                    'hel1os_broad_flux', 'hel1os_broad_std', 'hel1os_broad_max', 'hel1os_broad_deriv',
                    'hel1os_hard_soft_ratio', 'hel1os_total_flux']:
            sample[key] = np.nan
        samples.append(sample)
    quiet_total = sum(1 for s in samples if s['label'] == 0)
    print(f'  Total quiet samples: {quiet_total}')

    # ─── BALANCE: bootstrap-resample quiet to match flare count ─────────────
    flare_count = sum(1 for s in samples if s['label'] == 1)
    if quiet_total < flare_count:
        quiet_existing = [s for s in samples if s['label'] == 0]
        np.random.seed(42)
        needed = flare_count - quiet_total
        print(f'  Bootstrapping {needed} quiet samples to balance...')
        for i in range(needed):
            base = dict(np.random.choice(quiet_existing))
            new_sample = {}
            for k, v in base.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ['sample_id', 'label']:
                    try:
                        fv = float(v)
                        if np.isnan(fv):
                            new_sample[k] = v
                        else:
                            new_sample[k] = fv * (1 + np.random.normal(0, 0.05))
                    except (ValueError, TypeError):
                        new_sample[k] = v
                else:
                    new_sample[k] = v
            new_sample['sample_id'] = len(samples)
            new_sample['source'] = str(base.get('source', 'quiet')) + '_aug'
            samples.append(new_sample)

    df = pd.DataFrame(samples)
    out = os.path.join(PROCESSED_DIR, 'balanced_samples.parquet')
    df.to_parquet(out, index=False)

    print(f'\n{"="*60}')
    print('BALANCED SAMPLES BUILT')
    print(f'{"="*60}')
    print(f'Total: {len(df)}')
    print(f'Flare (1): {(df["label"]==1).sum()} ({(df["label"]==1).mean()*100:.1f}%)')
    print(f'Quiet (0): {(df["label"]==0).sum()} ({(df["label"]==0).mean()*100:.1f}%)')
    print(f'Features: {len([c for c in df.columns if c.startswith(("hel1os_", "goes_"))])}')
    print(f'Saved to: {out}')
    return df


if __name__ == '__main__':
    build_balanced_samples()