"""Analyze HEL1OS data and test feature extraction."""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.data.features import compute_features, get_feature_names

# Load HEL1OS
hel1os = pd.read_parquet('data/pradan_hel1os/hel1os_combined.parquet')
print(f'HEL1OS CdTe: {len(hel1os):,} records')
print(f'Time range: {hel1os.index.min()} to {hel1os.index.max()}')
print(f'Rate: min={hel1os["rate"].min():.0f}, max={hel1os["rate"].max():.0f}, mean={hel1os["rate"].mean():.2f}')
print(f'Zeros: {(hel1os["rate"]==0).sum():,}/{len(hel1os):,} ({(hel1os["rate"]==0).mean()*100:.1f}%)')

# Identify flare-like periods (rate > mean + 2*std)
threshold = hel1os["rate"].mean() + 2 * hel1os["rate"].std()
flare_mask = hel1os["rate"] > threshold
n_flare = flare_mask.sum()
print(f'\nFlare-like periods (rate > {threshold:.0f}): {n_flare:,} ({n_flare/len(hel1os)*100:.2f}%)')

# Show top flare times
if n_flare > 0:
    flare_data = hel1os[flare_mask].sort_values('rate', ascending=False)
    print('Top 10 flare-like moments:')
    for t, r in flare_data.head(10).iterrows():
        print(f'  {t}: {r["rate"]:.0f} counts')

# Compute features on normalized data
rate = hel1os['rate'].values.astype(float)
eps = 1e-12

# Z-score normalization
mean_r = rate[rate > 0].mean() if (rate > 0).any() else 1
std_r = rate[rate > 0].std() if (rate > 0).any() else 1
rate_norm = np.where(rate > 0, (rate - mean_r) / (std_r + eps), 0)
rate_norm = np.maximum(rate_norm, eps)

# Use as both channels
features_df = compute_features(rate_norm, rate_norm, cadence_seconds=10.0)
feature_names = get_feature_names()
features_df.index = hel1os.index

valid_mask = ~features_df[feature_names].isna().any(axis=1)
print(f'\nValid samples: {valid_mask.sum():,}/{len(hel1os):,}')

# Show feature statistics for flare vs quiet periods
if n_flare > 0:
    print('\nFeature comparison (flare vs quiet):')
    for fn in ['soft_log', 'hard_log', 'hard_soft_ratio', 'dsoft', 'dhard']:
        if fn in features_df.columns:
            flare_vals = features_df.loc[flare_mask & valid_mask, fn]
            quiet_vals = features_df.loc[~flare_mask & valid_mask, fn]
            if len(flare_vals) > 0 and len(quiet_vals) > 0:
                print(f'  {fn}: flare_mean={flare_vals.mean():.4f}, quiet_mean={quiet_vals.mean():.4f}')

# Save summary
summary = {
    'n_records': len(hel1os),
    'time_range': f'{hel1os.index.min()} to {hel1os.index.max()}',
    'rate_stats': {
        'min': float(hel1os['rate'].min()),
        'max': float(hel1os['rate'].max()),
        'mean': float(hel1os['rate'].mean()),
        'std': float(hel1os['rate'].std()),
    },
    'n_flare_like': int(n_flare),
    'flare_threshold': float(threshold),
    'n_valid_features': int(valid_mask.sum()),
}
Path('results').mkdir(exist_ok=True)
with open('results/hel1os_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print('\nSaved results/hel1os_analysis.json')
