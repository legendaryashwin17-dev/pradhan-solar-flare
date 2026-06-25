#!/usr/bin/env python3
"""
PRADHAN Real-Time GOES XRS Data Fetcher
========================================

Fetches real-time GOES-18 X-ray flux data from NOAA SWPC API.
Replaces simulated data on the web dashboard.

Data source: NOAA Space Weather Prediction Center
API: https://services.swpc.noaa.gov/json/goes/primary/
Resolution: 1-minute cadence
Channels: XRS-A (0.5-4 A), XRS-B (1-8 A)

Usage:
    python scripts/32_fetch_realtime_goes.py              # Fetch last 7 days
    python scripts/32_fetch_realtime_goes.py --hours 24   # Fetch last 24 hours
    python scripts/32_fetch_realtime_goes.py --json       # Output JSON for web
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
OUTPUT_DIR = os.path.join(WORKSPACE, 'data', 'realtime')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NOAA SWPC GOES XRS API endpoints
GOES_API_7DAY = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json'
GOES_API_1DAY = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json'

# NOAA flare event list for ground truth
NOAA_FLARES_API = 'https://services.swpc.noaa.gov/json/goes/primary/xray-events.json'


def fetch_json(url, timeout=30):
    """Fetch JSON from NOAA API."""
    req = Request(url, headers={'User-Agent': 'PRADHAN/2.0'})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except URLError as e:
        print(f'  API error: {e}')
        return None
    except Exception as e:
        print(f'  Error: {e}')
        return None


def fetch_goes_xrs(hours=168):
    """
    Fetch GOES XRS data from NOAA SWPC.
    
    Parameters
    ----------
    hours : int
        How many hours of data to fetch (default 168 = 7 days)
    
    Returns
    -------
    pd.DataFrame with columns: time, xrsa, xrsb, quality
    """
    url = GOES_API_7DAY if hours > 24 else GOES_API_1DAY
    print(f'Fetching GOES XRS data from NOAA SWPC...')
    print(f'  URL: {url}')
    
    data = fetch_json(url)
    if data is None or len(data) == 0:
        print('  No data returned from API')
        return None
    
    print(f'  Raw records: {len(data)}')
    
    # NOAA API returns separate rows for each energy band:
    #   0.05-0.4nm = XRS-A (soft), 0.1-0.8nm = XRS-B (hard)
    # Pivot to get xrsa and xrsb columns
    records = []
    for item in data:
        try:
            t = datetime.fromisoformat(item['time_tag'].replace('Z', '+00:00'))
            energy = item.get('energy', '')
            flux = float(item.get('flux', np.nan))
            records.append({'time': t, 'energy': energy, 'flux': flux})
        except (ValueError, KeyError):
            continue
    
    if not records:
        print('  No valid records parsed')
        return None
    
    raw_df = pd.DataFrame(records)
    
    # Pivot: rows=time, columns=energy, values=flux
    pivot = raw_df.pivot_table(index='time', columns='energy', values='flux', aggfunc='first')
    pivot.columns = ['xrsa' if '0.05' in str(c) else 'xrsb' for c in pivot.columns]
    pivot.index.name = 'time'
    df = pivot.sort_index()
    
    # Filter to requested time range
    cutoff = df.index.max() - pd.Timedelta(hours=hours)
    df = df[df.index >= cutoff]
    
    # Remove NaN and negative values
    df = df.dropna(subset=['xrsa', 'xrsb'])
    df = df[(df['xrsa'] > 0) & (df['xrsb'] > 0)]
    
    print(f'  Valid records: {len(df)}')
    print(f'  Time range: {df.index[0]} to {df.index[-1]}')
    print(f'  XRS-B range: {df["xrsb"].min():.2e} to {df["xrsb"].max():.2e}')
    
    return df


def classify_flux(flux):
    """Classify flux into NOAA class."""
    if flux >= 1e-4: return 'X'
    if flux >= 1e-5: return 'M'
    if flux >= 1e-6: return 'C'
    if flux >= 1e-7: return 'B'
    return 'A'


def extract_features(goes_df, feature_time, lookback_hours=1):
    """
    Extract 8 GOES features at feature_time.
    Same logic as scripts/22_build_balanced_samples.py.
    """
    from datetime import timedelta
    
    t1 = feature_time - timedelta(hours=lookback_hours)
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
    feats['goes_log_xrsa'] = float(np.log10(xrsa_now))
    feats['goes_log_xrsb'] = float(np.log10(xrsb_now))
    
    w24 = goes_df[(goes_df.index >= t24) & (goes_df.index <= feature_time)]
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
    
    w3 = goes_df[(goes_df.index >= t3) & (goes_df.index <= feature_time)]
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
    
    return feats


def fetch_noaa_flare_events():
    """Fetch NOAA official flare event list for validation."""
    print('\nFetching NOAA flare event list...')
    data = fetch_json(NOAA_FLARES_API)
    if data is None:
        return None
    
    records = []
    for item in data:
        try:
            records.append({
                'event_time': item.get('begin_time', item.get('peak_time', '')),
                'peak_time': item.get('peak_time', ''),
                'end_time': item.get('end_time', ''),
                'class': item.get('class', ''),
                'region': item.get('region', ''),
                'source_location': item.get('source_location', ''),
            })
        except (ValueError, KeyError):
            continue
    
    df = pd.DataFrame(records)
    print(f'  Found {len(df)} flare events')
    return df


def build_realtime_snapshot(goes_df, output_json=True):
    """
    Build a real-time snapshot for the web dashboard.
    
    Returns the latest data point and recent light curve.
    """
    if goes_df is None or len(goes_df) == 0:
        return None
    
    latest = goes_df.iloc[-1]
    current_flux = float(latest['xrsb'])
    current_class = classify_flux(current_flux)
    
    # Last 24h for light curve
    cutoff = goes_df.index.max() - pd.Timedelta(hours=24)
    recent = goes_df[goes_df.index >= cutoff]
    
    # Compute features at latest time
    features = extract_features(goes_df, goes_df.index[-1])
    
    snapshot = {
        'timestamp': goes_df.index[-1].isoformat(),
        'current_flux': current_flux,
        'current_flux_xrsa': float(latest['xrsa']),
        'current_class': current_class,
        'data_points': len(goes_df),
        'time_range': {
            'start': goes_df.index[0].isoformat(),
            'end': goes_df.index[-1].isoformat(),
        },
        'light_curve_24h': {
            'times': [t.isoformat() for t in recent.index[::5]],  # Downsample
            'xrsb': [float(v) for v in recent['xrsb'].values[::5]],
            'xrsa': [float(v) for v in recent['xrsa'].values[::5]],
        },
        'features': features,
        'flare_class': current_class,
        'source': 'NOAA GOES-18 XRS (real-time)',
    }
    
    if output_json:
        out_path = os.path.join(OUTPUT_DIR, 'goes_realtime.json')
        with open(out_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        print(f'\nSaved real-time snapshot: {out_path}')
    
    return snapshot


def main():
    parser = argparse.ArgumentParser(description='Fetch real-time GOES XRS data')
    parser.add_argument('--hours', type=int, default=168,
                       help='Hours of data to fetch (default: 168 = 7 days)')
    parser.add_argument('--save-parquet', action='store_true',
                       help='Save as parquet for historical use')
    parser.add_argument('--json', action='store_true',
                       help='Output JSON snapshot for web dashboard')
    args = parser.parse_args()
    
    print('=' * 60)
    print('PRADHAN Real-Time GOES XRS Fetcher')
    print('=' * 60)
    
    # Fetch data
    goes_df = fetch_goes_xrs(hours=args.hours)
    if goes_df is None:
        print('\nFailed to fetch GOES data')
        return
    
    # Save parquet
    if args.save_parquet:
        out_path = os.path.join(OUTPUT_DIR, f'goes_xrs_{args.hours}h.parquet')
        goes_df.to_parquet(out_path)
        print(f'\nSaved parquet: {out_path}')
    
    # Build snapshot
    snapshot = build_realtime_snapshot(goes_df, output_json=True)
    
    # Fetch flare events
    flare_events = fetch_noaa_flare_events()
    if flare_events is not None and len(flare_events) > 0:
        events_path = os.path.join(OUTPUT_DIR, 'noaa_flare_events.json')
        flare_events.to_json(events_path, orient='records', indent=2)
        print(f'Saved flare events: {events_path}')
    
    # Summary
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'Data points: {len(goes_df):,}')
    print(f'Time range: {goes_df.index[0]} to {goes_df.index[-1]}')
    print(f'Current flux: {snapshot["current_flux"]:.2e} W/m² ({snapshot["current_class"]}-class)')
    
    # Count flares in dataset
    flare_mask = goes_df['xrsb'] >= 1e-6
    c_count = ((goes_df['xrsb'] >= 1e-6) & (goes_df['xrsb'] < 1e-5)).sum()
    m_count = ((goes_df['xrsb'] >= 1e-5) & (goes_df['xrsb'] < 1e-4)).sum()
    x_count = (goes_df['xrsb'] >= 1e-4).sum()
    print(f'C-class minutes: {c_count:,}')
    print(f'M-class minutes: {m_count:,}')
    print(f'X-class minutes: {x_count:,}')


if __name__ == '__main__':
    main()
