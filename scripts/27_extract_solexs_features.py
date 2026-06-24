"""Extract SOLEXS features for each balanced sample's time window.

SOLEXS is a single-channel X-ray detector on ISRO Aditya-L1.
It provides 10-second cadence count rates (Feb 2024 - Jun 2026).

Feature extraction mirrors GOES log-space approach:
  - Log-space current rate, mean, std
  - Baseline ratio (current / median)
  - Temporal derivative
  - Z-score within window
  - Max rate, max-to-mean ratio
"""

import os
import numpy as np
import pandas as pd

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
SOLEXS_PARQUET = os.path.join(WORKSPACE, 'data', 'pradan_solexs', 'solexs_combined.parquet')
SAMPLES_PARQUET = os.path.join(WORKSPACE, 'data', 'processed', 'samples', 'balanced_samples.parquet')
OUT_PARQUET = os.path.join(WORKSPACE, 'data', 'processed', 'samples', 'solexs_features.parquet')

WINDOW_HOURS = 6

def extract_solexs_features(solexs_df, feature_time, window_hours=6):
    """Extract SOLEXS features for a single time window centered on feature_time."""
    t_start = feature_time - pd.Timedelta(hours=window_hours)
    t_end = feature_time
    window = solexs_df.loc[t_start:t_end]

    if len(window) < 100:
        return None

    rate = window['rate'].values.astype(float)
    log_rate = np.log10(np.clip(rate, 1e-10, None))

    features = {}

    # Current (end of window) values
    features['solexs_log_rate'] = float(log_rate[-1])
    features['solexs_rate'] = float(rate[-1])

    # Window statistics
    features['solexs_log_rate_mean'] = float(np.mean(log_rate))
    features['solexs_log_rate_std'] = float(np.std(log_rate))
    features['solexs_rate_mean'] = float(np.mean(rate))
    features['solexs_rate_max'] = float(np.max(rate))

    # Baseline ratio (current / median)
    median_rate = np.median(rate)
    features['solexs_baseline'] = float(rate[-1] / max(median_rate, 1e-10))

    # Temporal derivative (last 10 min = 60 samples)
    deriv_window = min(60, len(log_rate) - 1)
    if deriv_window > 1:
        features['solexs_log_deriv'] = float(
            (log_rate[-1] - log_rate[-deriv_window]) / deriv_window
        )
    else:
        features['solexs_log_deriv'] = 0.0

    # Z-score within window
    if np.std(log_rate) > 1e-10:
        features['solexs_log_zscore'] = float(
            (log_rate[-1] - np.mean(log_rate)) / np.std(log_rate)
        )
    else:
        features['solexs_log_zscore'] = 0.0

    # Max-to-mean ratio
    features['solexs_max_mean_ratio'] = float(np.max(rate) / max(np.mean(rate), 1e-10))

    # Fraction of time above threshold (flare indicator)
    threshold = np.percentile(rate, 95)
    features['solexs_above_p95'] = float(np.mean(rate > threshold))

    return features


def main():
    print('Loading SOLEXS data...')
    solexs = pd.read_parquet(SOLEXS_PARQUET)
    print(f'  {len(solexs)} records, {solexs.index.min()} to {solexs.index.max()}')

    print('Loading balanced samples...')
    samples = pd.read_parquet(SAMPLES_PARQUET)
    print(f'  {len(samples)} samples')

    feature_rows = []
    failed = 0

    for i, row in samples.iterrows():
        ft = row['feature_time']
        feats = extract_solexs_features(solexs, ft, WINDOW_HOURS)
        if feats is not None:
            feats['sample_id'] = row['sample_id']
            feats['feature_time'] = ft
            feats['label'] = row['label']
            feature_rows.append(feats)
        else:
            failed += 1

    print(f'\nExtracted features for {len(feature_rows)} samples ({failed} failed)')

    feat_df = pd.DataFrame(feature_rows)
    print(f'\nSOLEXS feature columns:')
    solexs_cols = [c for c in feat_df.columns if c.startswith('solexs_')]
    print(f'  {solexs_cols}')

    print(f'\nFeature statistics:')
    for c in solexs_cols:
        vals = feat_df[c].dropna()
        print(f'  {c:<25s}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
              f'min={vals.min():.4f}, max={vals.max():.4f}')

    # Merge with original samples
    merged = samples.merge(feat_df[['sample_id'] + solexs_cols], on='sample_id', how='left')
    print(f'\nMerged dataset: {merged.shape}')
    print(f'SOLEXS features non-null: {merged[solexs_cols[0]].notna().sum()} / {len(merged)}')

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    merged.to_parquet(OUT_PARQUET, index=False)
    print(f'\nSaved: {OUT_PARQUET}')

    return feat_df


if __name__ == '__main__':
    main()
