#!/usr/bin/env python3
"""
SHARP Magnetic Feature Downloader for PRADHAN

Downloads HMI SHARP parameters from JSOC for active regions present in
the GOES-18 flare catalog (2026).

JSOC requires a registered email. Set JSOC_EMAIL env var.

7 Golden parameters:
  USFLUX  - Total unsigned magnetic flux
  TOTUSJH - Total unsigned current helicity (twist/shear)
  TOTUSJZ - Total unsigned vertical current
  TOTPOT  - Total photospheric magnetic free energy
  R_VALUE - R-value (flux cancellation proxy)
  SAVNCPP - Absolute value of net current per polarity
  MEANPOT - Mean photospheric magnetic free energy density
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import drms
    HAS_DRMS = True
except ImportError:
    HAS_DRMS = False

# Output paths
WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
OUTPUT_DIR = os.path.join(WORKSPACE, 'data', 'raw', 'sharp')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_arp_harp_mapping():
    """
    Get NOAA AR -> HARPNUM mapping for 2026 active regions.
    JSOC SHARP uses HARPNUM (not NOAA AR). We need to query the
    hmi.sharp_cea_720s series to find HARPNUMs active during 2026-04..2026-06.

    Without an AR catalog, we query a few sample AR numbers and look up
    HARPNUMs by date range.
    """
    # From context: Solar Cycle 25 ARs are in 13900-14000 range during 2026
    # We'll query a small batch first to verify
    return list(range(13900, 14001))


def query_sharp_for_arp(harp_num, start_date, end_date, client):
    """
    Query JSOC for SHARP parameters for a specific HARPNUM.

    Returns DataFrame with columns: T_REC, USFLUX, TOTUSJH, TOTUSJZ, TOTPOT,
    R_VALUE, SAVNCPP, MEANPOT, plus quality flags.
    """
    series = 'hmi.sharp_cea_720s'
    keys = 'USFLUX,TOTUSJH,TOTUSJZ,TOTPOT,R_VALUE,SAVNCPP,MEANPOT'

    try:
        query_str = f'{series}[{harp_num}][{start_date.strftime("%Y.%m.%d")}_TAI-{end_date.strftime("%Y.%m.%d")}_TAI]{{{keys}}}'
        result = client.query(query_str)
        if len(result) == 0:
            return None
        result['HARPNUM'] = harp_num
        return result
    except Exception as e:
        # Some HARPs don't exist - silently skip
        return None


def main():
    if not HAS_DRMS:
        print('ERROR: drms not installed. pip install drms')
        sys.exit(1)

    email = os.getenv('JSOC_EMAIL')
    if not email:
        print('ERROR: JSOC_EMAIL env var not set')
        print('Set with: $env:JSOC_EMAIL = "your.email@domain.com"')
        sys.exit(1)

    print(f'Connecting to JSOC as {email}...')
    try:
        client = drms.Client(email=email)
        print('JSOC auth OK')
    except Exception as e:
        print(f'JSOC auth failed: {e}')
        sys.exit(1)

    # Date range matching our GOES-18 + HEL1OS coverage
    start = datetime(2026, 4, 1)
    end = datetime(2026, 6, 30)

    arp_list = get_arp_harp_mapping()
    print(f'\nQuerying {len(arp_list)} candidate HARPNUMs from {start} to {end}...')

    all_records = []
    success_count = 0

    for i, harp in enumerate(arp_list):
        if (i + 1) % 10 == 0:
            print(f'  Progress: {i+1}/{len(arp_list)} ({success_count} successful)')

        df = query_sharp_for_arp(harp, start, end, client)
        if df is not None and len(df) > 0:
            all_records.append(df)
            success_count += 1

        # Rate limit: JSOC allows ~10 queries/sec
        time.sleep(0.15)

    print(f'\nTotal HARPs with data: {success_count}')

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        print(f'Total SHARP records: {len(combined)}')
        print(f'Columns: {list(combined.columns)}')
        print(f'\nFirst 3 records:')
        print(combined.head(3))

        # Drop rows with NaN in critical features
        critical = ['USFLUX', 'TOTUSJH', 'TOTUSJZ', 'TOTPOT', 'R_VALUE', 'SAVNCPP', 'MEANPOT']
        before = len(combined)
        combined = combined.dropna(subset=critical)
        print(f'\nDropped {before - len(combined)} rows with NaN in critical features')
        print(f'Final: {len(combined)} records')

        # Save
        out_path = os.path.join(OUTPUT_DIR, 'sharp_real.csv')
        combined.to_csv(out_path, index=False)
        print(f'\nSaved: {out_path}')

        # Summary statistics
        print(f'\nSummary:')
        print(combined[critical].describe())
    else:
        print('No SHARP records retrieved from JSOC')


if __name__ == '__main__':
    main()