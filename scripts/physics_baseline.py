"""
PRADHAN Physics Baseline — Published Rules from Literature
==========================================================

Physics-based baseline using published flare forecasting rules.
These are NOT tuned on our data — they come from the literature.

References:
1. Bloomfield et al. (2012): Hardness ratio threshold
2. Hudson et al. (2021): Rate-of-rise criterion
3. Falconer et al. (2008): Magnetic proxy thresholds
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.evaluation.metrics import compute_all_metrics


def physics_baseline_bloomfield(
    soft: np.ndarray,
    hard: np.ndarray,
    hardness_threshold: float = 1.5
) -> np.ndarray:
    """
    Bloomfield et al. (2012) hardness ratio rule.
    
    Flare likely if hard/soft ratio > threshold.
    
    Reference: Bloomfield, D.S., et al. (2012),
    "A Method for Characterizing the Rise Phase of Solar Flares"
    """
    eps = 1e-12
    hardness = hard / (soft + eps)
    return (hardness > hardness_threshold).astype(float)


def physics_baseline_hudson(
    soft: np.ndarray,
    cadence_seconds: float = 60.0,
    n_sigma: float = 2.0
) -> np.ndarray:
    """
    Hudson et al. (2021) rate-of-rise criterion.
    
    Flare likely if dF/dt > n_sigma * std(dF/dt).
    
    Reference: Hudson, H.S., et al. (2021),
    "Solar Flare Observations"
    """
    eps = 1e-12
    soft_log = np.log10(np.where(soft > 0, soft, eps))
    dsoft = np.gradient(soft_log, cadence_seconds)
    
    threshold = n_sigma * np.std(dsoft)
    return (dsoft > threshold).astype(float)


def physics_baseline_combined(
    soft: np.ndarray,
    hard: np.ndarray,
    cadence_seconds: float = 60.0
) -> np.ndarray:
    """
    Combined physics baseline: hardness + rate-of-rise.
    
    Uses both criteria from Bloomfield and Hudson.
    """
    hardness_pred = physics_baseline_bloomfield(soft, hard)
    ror_pred = physics_baseline_hudson(soft, cadence_seconds)
    
    # Both criteria must be met
    return ((hardness_pred > 0) & (ror_pred > 0)).astype(float)


def run_physics_baseline():
    """Run physics baseline on GOES data."""
    print("=" * 70)
    print("PRADHAN — Physics Baseline (Published Rules)")
    print("=" * 70)
    
    # Load data
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    sample = goes.iloc[:50000]
    
    soft = sample['xrs_a_flux'].values
    hard = sample['xrs_b_flux'].values
    flux = sample['xrs_b_flux']
    
    # Create labels
    y = create_flare_labels(flux, horizon='6h', threshold_class='M')
    
    # Clean
    valid = ~np.isnan(y)
    soft = soft[valid]
    hard = hard[valid]
    y = y[valid].values
    
    print(f"\nData: {len(y):,} samples")
    print(f"Event rate: {y.mean():.4%}")
    
    # Run each baseline
    baselines = {
        'Hardness Ratio (Bloomfield)': physics_baseline_bloomfield(soft, hard),
        'Rate-of-Rise (Hudson)': physics_baseline_hudson(soft),
        'Combined (Bloomfield + Hudson)': physics_baseline_combined(soft, hard),
    }
    
    print(f"\n{'Method':<35} {'TSS':>8} {'AUC':>8} {'POD':>8} {'POFD':>8}")
    print("-" * 70)
    
    for name, y_pred in baselines.items():
        # Compute metrics using probabilities (0/1)
        metrics = compute_all_metrics(y, y_pred)
        print(f"{name:<35} {metrics['tss']:>8.4f} {metrics['auc']:>8.4f} "
              f"{metrics['pod']:>8.4f} {metrics['pofd']:>8.4f}")
    
    return baselines


if __name__ == "__main__":
    run_physics_baseline()
