"""
PRADHAN Label Creation — Proper Flare Definitions
==================================================

This module creates proper flare labels following NOAA conventions.

CRITICAL: Proper label definition is essential for meaningful results.
Using arbitrary thresholds undermines scientific validity.

NOAA Flare Classification:
- A-class: 10^-8 W/m² (negligible)
- B-class: 10^-7 W/m² (minor)
- C-class: 10^-6 W/m² (moderate)
- M-class: 10^-5 W/m² (strong)
- X-class: 10^-4 W/m² (extreme)

Reference: https://www.swpc.noaa.gov/noaa-scales-explanation
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
from datetime import timedelta


# Official NOAA thresholds
FLUX_THRESHOLDS = {
    'A': 1e-8,
    'B': 1e-7,
    'C': 1e-6,
    'M': 1e-5,
    'X': 1e-4,
}

# NOAA scale labels for output
NOAA_SCALE_LABELS = {
    0: 'Quiet (<B)',
    1: 'B-class',
    2: 'C-class',
    3: 'M-class',
    4: 'X-class',
}

# Standard forecast horizons used in operational forecasting
# (These are what NOAA SWPC uses, plus short-term nowcasting horizons)
FORECAST_HORIZONS = {
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '6h': timedelta(hours=6),
    '12h': timedelta(hours=12),
    '24h': timedelta(hours=24),
    '48h': timedelta(hours=48),
    '72h': timedelta(hours=72),
}


def classify_flare(flux: float) -> int:
    """
    Classify flare based on peak flux using NOAA scale.
    
    Parameters
    ----------
    flux : float
        Peak X-ray flux in W/m² (typically XRS-B 1-8 Å)
        
    Returns
    -------
    int
        0: Quiet, 1: B, 2: C, 3: M, 4: X
    """
    if flux >= FLUX_THRESHOLDS['X']:
        return 4
    elif flux >= FLUX_THRESHOLDS['M']:
        return 3
    elif flux >= FLUX_THRESHOLDS['C']:
        return 2
    elif flux >= FLUX_THRESHOLDS['B']:
        return 1
    else:
        return 0


def get_flare_class_string(flux: float) -> str:
    """Get NOAA class string for a flux value."""
    if flux >= FLUX_THRESHOLDS['X']:
        return f"X{flux / FLUX_THRESHOLDS['X']:.1f}"
    elif flux >= FLUX_THRESHOLDS['M']:
        return f"M{flux / FLUX_THRESHOLDS['M']:.1f}"
    elif flux >= FLUX_THRESHOLDS['C']:
        return f"C{flux / FLUX_THRESHOLDS['C']:.1f}"
    elif flux >= FLUX_THRESHOLDS['B']:
        return f"B{flux / FLUX_THRESHOLDS['B']:.1f}"
    else:
        return "A or below"


def create_flare_labels(
    flux: pd.Series,
    horizon: str = '24h',
    threshold_class: str = 'M',
    require_sustained: bool = False,
    sustained_minutes: int = 10
) -> pd.Series:
    """
    Create binary flare labels for a given forecast horizon.
    
    This is the CORRECT way to define labels for flare forecasting.
    
    Parameters
    ----------
    flux : pd.Series
        X-ray flux time series (indexed by time)
    horizon : str
        Forecast horizon ('1h', '6h', '12h', '24h', '48h', '72h')
    threshold_class : str
        Minimum flare class to label as positive ('C', 'M', or 'X')
    require_sustained : bool
        If True, require flare to be above threshold for sustained_minutes
    sustained_minutes : int
        Minimum duration above threshold for sustained=True
        
    Returns
    -------
    pd.Series
        Binary labels (1 = flare event, 0 = no event)
        
    Example
    -------
    >>> # Predict M1+ events in next 24 hours
    >>> labels = create_flare_labels(
    ...     flux=goes['xrs_b_flux'],
    ...     horizon='24h',
    ...     threshold_class='M'
    ... )
    """
    threshold_flux = FLUX_THRESHOLDS.get(threshold_class, FLUX_THRESHOLDS['M'])
    
    # Get the horizon as timedelta
    horizon_delta = FORECAST_HORIZONS.get(horizon, timedelta(hours=24))
    horizon_minutes = int(horizon_delta.total_seconds() / 60)
    
    # Compute forward-looking maximum
    # This is the "correct" label: does a flare occur in the next N hours?
    forward_max = flux.shift(-horizon_minutes).rolling(
        window=horizon_minutes,
        min_periods=1
    ).max()
    
    # Alternative: use the peak in the horizon window
    # This looks at the max over the next N minutes
    forward_max = flux.rolling(
        window=horizon_minutes,
        min_periods=1
    ).max().shift(-horizon_minutes)
    
    # Create binary labels
    labels = (forward_max >= threshold_flux).astype(float)
    
    # Optionally require sustained duration
    if require_sustained:
        # Mark as positive only if above threshold for sustained_minutes
        above_threshold = (flux >= threshold_flux).astype(float)
        sustained_count = above_threshold.rolling(
            window=sustained_minutes,
            min_periods=sustained_minutes
        ).sum()
        labels = ((sustained_count >= sustained_minutes) & 
                  (forward_max >= threshold_flux)).astype(float)
    
    return labels


def create_multiclass_labels(
    flux: pd.Series,
    horizon: str = '24h'
) -> pd.Series:
    """
    Create multiclass labels (quiet, B, C, M, X).
    
    Parameters
    ----------
    flux : pd.Series
        X-ray flux time series
    horizon : str
        Forecast horizon
        
    Returns
    -------
    pd.Series
        Multiclass labels (0-4)
    """
    horizon_delta = FORECAST_HORIZONS.get(horizon, timedelta(hours=24))
    horizon_minutes = int(horizon_delta.total_seconds() / 60)
    
    forward_max = flux.rolling(
        window=horizon_minutes,
        min_periods=1
    ).max().shift(-horizon_minutes)
    
    labels = forward_max.apply(classify_flare)
    return labels


def compute_climatological_rate(
    flux: pd.Series,
    threshold_class: str = 'M',
    horizon: str = '24h'
) -> float:
    """
    Compute the climatological flare rate (baseline probability).
    
    This is the fraction of time periods that have a flare.
    Used as a baseline comparison for model skill.
    
    Parameters
    ----------
    flux : pd.Series
        X-ray flux time series
    threshold_class : str
        Minimum flare class
    horizon : str
        Forecast horizon (determines how often we "predict")
        
    Returns
    -------
    float
        Climatological rate (probability of flare)
    """
    labels = create_flare_labels(flux, horizon, threshold_class)
    return labels.mean()


def get_event_statistics(
    flux: pd.Series,
    horizon: str = '24h',
    threshold_class: str = 'M'
) -> Dict:
    """
    Get statistics about flare events in the dataset.
    
    Returns
    -------
    dict
        Event statistics including rate, typical peak, etc.
    """
    threshold_flux = FLUX_THRESHOLDS.get(threshold_class, FLUX_THRESHOLDS['M'])
    labels = create_flare_labels(flux, horizon, threshold_class)
    
    # Find event peaks
    event_mask = labels > 0
    event_indices = flux[event_mask].index
    
    # Compute event statistics
    stats = {
        'total_periods': len(labels),
        'event_periods': int(event_mask.sum()),
        'climatological_rate': float(event_mask.mean()),
        'no_event_periods': int((~event_mask).sum()),
        'data_span_hours': (flux.index[-1] - flux.index[0]).total_seconds() / 3600,
        'threshold_used': threshold_class,
        'horizon_used': horizon,
    }
    
    # If there are events, compute more stats
    if len(event_indices) > 0:
        event_fluxes = flux[event_mask]
        stats['mean_peak_flux'] = float(event_fluxes.mean())
        stats['max_peak_flux'] = float(event_fluxes.max())
        stats['min_peak_flux'] = float(event_fluxes.min())
    
    return stats


def print_label_summary(
    flux: pd.Series,
    horizons: list = None,
    threshold_classes: list = None
):
    """
    Print summary of flare labels for different configurations.
    """
    if horizons is None:
        horizons = ['1h', '6h', '24h', '48h']
    if threshold_classes is None:
        threshold_classes = ['C', 'M', 'X']
    
    print("\n" + "=" * 70)
    print("FLARE LABEL SUMMARY")
    print("=" * 70)
    print(f"\nData range: {flux.index.min()} to {flux.index.max()}")
    print(f"Total records: {len(flux)}")
    print(f"Time span: {(flux.index[-1] - flux.index[0]).total_seconds() / 3600:.1f} hours")
    print("\n" + "-" * 70)
    
    for horizon in horizons:
        print(f"\n{horizon} Forecast Horizon:")
        print(f"{'Threshold':<12} {'Event Rate':>12} {'Total Events':>14}")
        print("-" * 40)
        
        for thresh in threshold_classes:
            rate = compute_climatological_rate(flux, thresh, horizon)
            n_events = int(rate * len(flux))
            print(f"{thresh+'-class':<12} {rate:>11.4%} {n_events:>14,}")


if __name__ == "__main__":
    # Test label creation
    from reader import get_sample_data
    
    print("Testing label creation...")
    
    # Generate sample data
    df = get_sample_data(n_points=10000)
    flux = df['xrs_b_flux']
    
    # Print summary
    print_label_summary(flux)
    
    # Test different horizons
    print("\n" + "=" * 70)
    print("Testing different configurations...")
    
    for horizon in ['1h', '6h', '24h']:
        for thresh in ['C', 'M']:
            labels = create_flare_labels(flux, horizon, thresh)
            print(f"Horizon={horizon}, Threshold={thresh}: "
                  f"Event rate={labels.mean():.4%}")