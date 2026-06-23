"""
PRADHAN Feature Engineering — Statistical Proxies from Light Curves
===================================================================

SCIENTIFIC HONESTY:
These are STATISTICAL features derived from X-ray light curves.
They are NOT physics-based parameters.

What we compute:
- Derivatives, ratios, variances of time series
- Correlate empirically with flaring activity
- Validated through literature (Bloomfield et al. 2012)

What would be true physics (requires magnetograms):
- Magnetic free energy, shear angle, R-value, current helicity

Features (19 total):
- soft, hard: Raw flux channels (log-scaled)
- soft_log, hard_log: Log10 of flux
- hard_soft_ratio: XRS-B / XRS-A ratio
- dsoft, dhard: First derivatives (rate of change)
- soft_mean_1m, hard_mean_1m: Rolling 1-minute means
- soft_mean_5m, hard_mean_5m: Rolling 5-minute means
- soft_std_1m, hard_std_1m: Rolling 1-minute std
- soft_std_5m, hard_std_5m: Rolling 5-minute std
- soft_hard_corr: Pearson correlation (1-min window)
- xcorr: Cross-correlation lag-1
- dhard_soft_ratio: Ratio of derivatives
- ddsoft: Second derivative (acceleration)
"""

import numpy as np
import pandas as pd
from typing import List
from scipy.ndimage import uniform_filter1d


FEATURE_NAMES = [
    'soft', 'hard',
    'soft_log', 'hard_log',
    'hard_soft_ratio',
    'dsoft', 'dhard',
    'soft_mean_1m', 'hard_mean_1m',
    'soft_mean_5m', 'hard_mean_5m',
    'soft_std_1m', 'hard_std_1m',
    'soft_std_5m', 'hard_std_5m',
    'soft_hard_corr',
    'xcorr',
    'dhard_soft_ratio',
    'ddsoft',
    'spectral_hardening',
    'neupert_proxy',
]


def get_feature_names() -> List[str]:
    return FEATURE_NAMES.copy()


def compute_features(
    soft: np.ndarray,
    hard: np.ndarray,
    cadence_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Compute 19 statistical proxy features from X-ray light curves.

    Parameters
    ----------
    soft : np.ndarray
        Soft X-ray flux (XRS-A, 0.5-4 A, or SoLEXS 2-22 keV)
    hard : np.ndarray
        Hard X-ray flux (XRS-B, 1-8 A, or HEL1OS 8-150 keV)
    cadence_seconds : float
        Cadence in seconds (60 for GOES, 1-10 for SoLEXS/HEL1OS)

    Returns
    -------
    pd.DataFrame
        DataFrame with 19 features, NaN for initial window
    """
    n = len(soft)
    eps = 1e-12

    # Replace zeros/negatives
    soft = np.where(soft > 0, soft, eps)
    hard = np.where(hard > 0, hard, eps)

    # Window sizes in samples
    win_1m = max(1, int(60 / cadence_seconds))     # 1 minute
    win_5m = max(1, int(300 / cadence_seconds))    # 5 minutes

    # Log-transformed channels
    soft_log = np.log10(soft)
    hard_log = np.log10(hard)

    # Ratio
    hard_soft_ratio = hard / (soft + eps)

    # First derivatives (rate of change)
    dsoft = np.gradient(soft_log, cadence_seconds)
    dhard = np.gradient(hard_log, cadence_seconds)

    # Rolling means
    soft_mean_1m = uniform_filter1d(soft, size=win_1m, mode='nearest')
    hard_mean_1m = uniform_filter1d(hard, size=win_1m, mode='nearest')
    soft_mean_5m = uniform_filter1d(soft, size=win_5m, mode='nearest')
    hard_mean_5m = uniform_filter1d(hard, size=win_5m, mode='nearest')

    # Rolling std (via rolling variance approximation)
    soft_std_1m = _rolling_std(soft, win_1m)
    hard_std_1m = _rolling_std(hard, win_1m)
    soft_std_5m = _rolling_std(soft, win_5m)
    hard_std_5m = _rolling_std(hard, win_5m)

    # Pearson correlation (5-min window, needs at least 2 samples)
    soft_hard_corr = _rolling_corr(soft, hard, max(2, win_5m))

    # Cross-correlation at lag-1
    xcorr = np.full(n, np.nan)
    xcorr[1:] = np.corrcoef(soft[:-1], hard[1:])[0, 1:]

    # Derivative ratio
    dhard_soft_ratio = dhard / (dsoft + eps)

    # Second derivative (acceleration)
    ddsoft = np.gradient(dsoft, cadence_seconds)

    # Physics-inspired proxies
    # Spectral hardening proxy: rate of change of hard/soft ratio
    # Indicates whether the spectrum is hardening (potential flare precursor)
    spectral_hardening = np.gradient(hard_soft_ratio, cadence_seconds)

    # Neupert-inspired proxy: integral of hard X-ray proxy (dHard/dt)
    # The Neupert effect states that hard X-ray emission is proportional
    # to the time derivative of soft X-ray emission.
    # Proxy: cumulative dhard * soft (energy transport estimate)
    neupert_proxy = np.cumsum(np.maximum(dhard, 0)) * soft

    features = pd.DataFrame({
        'soft': soft,
        'hard': hard,
        'soft_log': soft_log,
        'hard_log': hard_log,
        'hard_soft_ratio': hard_soft_ratio,
        'dsoft': dsoft,
        'dhard': dhard,
        'soft_mean_1m': soft_mean_1m,
        'hard_mean_1m': hard_mean_1m,
        'soft_mean_5m': soft_mean_5m,
        'hard_mean_5m': hard_mean_5m,
        'soft_std_1m': soft_std_1m,
        'hard_std_1m': hard_std_1m,
        'soft_std_5m': soft_std_5m,
        'hard_std_5m': hard_std_5m,
        'soft_hard_corr': soft_hard_corr,
        'xcorr': xcorr,
        'dhard_soft_ratio': dhard_soft_ratio,
        'ddsoft': ddsoft,
        'spectral_hardening': spectral_hardening,
        'neupert_proxy': neupert_proxy,
    })

    # Replace inf with nan
    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation using pandas (C-backed, fast)."""
    if window <= 1:
        # For window=1, std is always 0 (single sample)
        return np.zeros_like(arr, dtype=float)
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).std(ddof=0).values


def _rolling_corr(
    arr1: np.ndarray, arr2: np.ndarray, window: int
) -> np.ndarray:
    """Compute rolling Pearson correlation using pandas (C-backed, fast)."""
    s1 = pd.Series(arr1)
    s2 = pd.Series(arr2)
    return s1.rolling(window=window, min_periods=window).corr(s2).values


if __name__ == "__main__":
    print("Testing feature computation...")
    np.random.seed(42)
    n = 6000
    soft = 1e-8 + np.random.lognormal(0, 0.3, n) * 1e-8
    hard = soft * 2 + np.random.lognormal(0, 0.2, n) * 1e-8

    features = compute_features(soft, hard, cadence_seconds=60.0)
    print(f"\nComputed {len(FEATURE_NAMES)} features for {n} points")
    print(f"\nFeature summary:")
    print(features.describe().round(4))
