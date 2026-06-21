"""
PRADHAN Calibration Analysis — Proper Reliability Assessment
============================================================

Calibration analysis ensures predicted probabilities match true frequencies.

For rare events like flares, this is CRITICAL because:
1. Users need to interpret probabilities correctly
2. Overconfident predictions can lead to poor decisions
3. Brier score penalizes both discrimination and calibration

This module provides:
- Reliability diagrams with confidence intervals
- Calibration metrics (Brier score, expected calibration error)
- Proper comparison across forecast horizons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy import stats


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    method: str = 'quantile'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve with bin statistics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins for calibration assessment
    method : str
        'quantile' (equal sample size) or 'uniform' (equal probability range)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean_predicted, true_fraction, bin_counts)
    """
    if method == 'quantile':
        # Equal sample sizes per bin
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    else:
        # Equal probability range
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    mean_predicted = []
    true_fraction = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted.append(np.mean(y_proba[mask]))
            true_fraction.append(np.mean(y_true[mask]))
            bin_counts.append(mask.sum())
        else:
            mean_predicted.append(np.nan)
            true_fraction.append(np.nan)
            bin_counts.append(0)
    
    return (
        np.array(mean_predicted),
        np.array(true_fraction),
        np.array(bin_counts)
    )


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = Σ (|B_k| / n) * |acc(B_k) - conf(B_k)|
    
    Where:
    - B_k is bin k
    - acc(B_k) is accuracy in bin k
    - conf(B_k) is average confidence in bin k
    
    Range: [0, 1]
    Perfect: 0
    
    Reference: Naeini et al. (2015)
    """
    mean_pred, true_frac, bin_counts = compute_calibration_curve(
        y_true, y_proba, n_bins, method='quantile'
    )
    
    ece = 0
    total = len(y_true)
    
    for i, count in enumerate(bin_counts):
        if count > 0 and not np.isnan(mean_pred[i]):
            ece += (count / total) * abs(true_frac[i] - mean_pred[i])
    
    return ece


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.
    
    Returns
    -------
    dict
        Dictionary of calibration metrics including Brier, ECE,
        intercept, and slope (logistic calibration).
    """
    from sklearn.linear_model import LogisticRegression
    
    # Brier score
    brier = brier_score_loss(y_true, y_proba)
    
    # Base rate Brier (for comparison)
    base_rate = np.mean(y_true)
    brier_random = brier_score_loss(y_true, np.full(len(y_true), base_rate))
    
    # Brier Skill Score
    bss = 1 - (brier / brier_random) if brier_random > 0 else 0
    
    # Expected Calibration Error
    ece = compute_expected_calibration_error(y_true, y_proba, n_bins)
    
    # Maximum Calibration Error
    mean_pred, true_frac, _ = compute_calibration_curve(y_true, y_proba, n_bins)
    valid = ~np.isnan(mean_pred)
    mce = np.max(np.abs(true_frac[valid] - mean_pred[valid])) if valid.sum() > 0 else 0
    
    # Reliability (inverse of ECE-like measure)
    valid_idx = ~np.isnan(mean_pred) & ~np.isnan(true_frac)
    if valid_idx.sum() > 1:
        reliability_corr = np.corrcoef(mean_pred[valid_idx], true_frac[valid_idx])[0, 1]
    else:
        reliability_corr = np.nan
    
    # Logistic calibration: intercept and slope
    # Perfect calibration: intercept=0, slope=1
    # Overconfident: |intercept| > 0, slope < 1
    # Underconfident: slope > 1
    try:
        cal_model = LogisticRegression(random_state=42, max_iter=1000)
        cal_model.fit(y_proba.reshape(-1, 1), y_true)
        cal_intercept = float(cal_model.intercept_[0])
        cal_slope = float(cal_model.coef_[0][0])
    except:
        cal_intercept = 0.0
        cal_slope = 1.0
    
    return {
        'brier_score': brier,
        'brier_random': brier_random,
        'brier_skill_score': bss,
        'expected_calibration_error': ece,
        'max_calibration_error': mce,
        'reliability_correlation': reliability_corr if not np.isnan(reliability_corr) else 0,
        'calibration_intercept': cal_intercept,
        'calibration_slope': cal_slope,
        'base_rate': base_rate,
    }


def bootstrap_calibration_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for calibration curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (default 0.95 for 95% CI)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (lower_bound, upper_bound, median) for each bin
    """
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    # Initialize storage for bootstrap calibration curves
    bootstrap_curves = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_proba_boot = y_proba[indices]
        
        # Compute calibration curve for this bootstrap sample
        try:
            mean_pred, true_frac, _ = compute_calibration_curve(
                y_true_boot, y_proba_boot, n_bins=10
            )
            bootstrap_curves.append(true_frac)
        except:
            continue
    
    if len(bootstrap_curves) < 10:
        # Not enough successful bootstraps
        return None, None, None
    
    bootstrap_curves = np.array(bootstrap_curves)
    
    # Compute percentiles
    lower = np.percentile(bootstrap_curves, alpha / 2 * 100, axis=0)
    upper = np.percentile(bootstrap_curves, (1 - alpha / 2) * 100, axis=0)
    median = np.median(bootstrap_curves, axis=0)
    
    return lower, upper, median


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Reliability Diagram",
    filename: Optional[str] = None,
    show_ci: bool = True,
    n_bins: int = 10
) -> plt.Figure:
    """
    Create a reliability diagram with optional confidence intervals.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    title : str
        Plot title
    filename : str, optional
        If provided, save figure to this path
    show_ci : bool
        Whether to show bootstrap confidence intervals
    n_bins : int
        Number of bins
        
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute calibration curve
    mean_pred, true_frac, bin_counts = compute_calibration_curve(
        y_true, y_proba, n_bins, method='quantile'
    )
    
    # Plot confidence intervals if requested
    if show_ci:
        lower, upper, _ = bootstrap_calibration_ci(y_true, y_proba)
        if lower is not None:
            x_bins = np.linspace(0, 1, n_bins)
            ax.fill_between(x_bins, lower, upper, alpha=0.2, color='blue',
                          label='95% CI')
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Plot actual calibration
    valid = ~np.isnan(mean_pred) & ~np.isnan(true_frac)
    ax.plot(mean_pred[valid], true_frac[valid], 'o-', 
            color='blue', linewidth=2, markersize=10,
            label='Model Calibration')
    
    # Add bin counts as annotations
    for i, (mp, tf, count) in enumerate(zip(mean_pred, true_frac, bin_counts)):
        if count > 0:
            ax.annotate(f'n={count}', (mp, tf), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='center',
                       fontsize=8)
    
    # Compute metrics for title
    metrics = compute_calibration_metrics(y_true, y_proba, n_bins)
    
    # Labels and title
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'{title}\n'
                f'ECE={metrics["expected_calibration_error"]:.3f}, '
                f'Brier={metrics["brier_score"]:.4f}', fontsize=12)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add text box with key metrics
    textstr = '\n'.join([
        f'Brier Score: {metrics["brier_score"]:.4f}',
        f'BSS: {metrics["brier_skill_score"]:.3f}',
        f'ECE: {metrics["expected_calibration_error"]:.3f}',
        f'N: {len(y_true):,}',
        f'Event Rate: {metrics["base_rate"]:.2%}',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {filename}")
    
    return fig


def compare_calibration_across_horizons(
    results_by_horizon: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Compare calibration across different forecast horizons.
    
    Parameters
    ----------
    results_by_horizon : dict
        Dictionary mapping horizon name to (y_true, y_proba) tuple
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    rows = []
    
    for horizon, (y_true, y_proba) in results_by_horizon.items():
        metrics = compute_calibration_metrics(y_true, y_proba)
        metrics['horizon'] = horizon
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    df = df.set_index('horizon')
    
    return df[['base_rate', 'brier_score', 'brier_skill_score',
               'expected_calibration_error', 'reliability_correlation']]


def print_calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "CALIBRATION REPORT"
) -> str:
    """
    Generate and print a comprehensive calibration report.
    
    Returns
    -------
    str
        The report text
    """
    metrics = compute_calibration_metrics(y_true, y_proba)
    
    report = []
    report.append("\n" + "=" * 60)
    report.append(title)
    report.append("=" * 60)
    
    report.append(f"\nBrier Score: {metrics['brier_score']:.4f}")
    report.append(f"Brier Score (Random): {metrics['brier_random']:.4f}")
    report.append(f"Brier Skill Score: {metrics['brier_skill_score']:.3f}")
    
    report.append(f"\nExpected Calibration Error (ECE): {metrics['expected_calibration_error']:.4f}")
    report.append(f"Maximum Calibration Error (MCE): {metrics['max_calibration_error']:.4f}")
    report.append(f"Reliability Correlation: {metrics['reliability_correlation']:.3f}")
    
    report.append(f"\nBase Rate (Event Frequency): {metrics['base_rate']:.4%}")
    report.append(f"Sample Size: {len(y_true):,}")
    
    # Interpretation
    report.append("\n" + "-" * 60)
    report.append("INTERPRETATION:")
    report.append("-" * 60)
    
    if metrics['expected_calibration_error'] < 0.05:
        report.append("[OK] ECE < 0.05: Well calibrated")
    elif metrics['expected_calibration_error'] < 0.10:
        report.append("[!!] ECE 0.05-0.10: Reasonably calibrated")
    else:
        report.append("[XX] ECE > 0.10: Poorly calibrated")
    
    if metrics['brier_skill_score'] > 0.5:
        report.append("[OK] BSS > 0.5: Significant improvement over climatology")
    elif metrics['brier_skill_score'] > 0:
        report.append("[!!] BSS 0-0.5: Some improvement over climatology")
    else:
        report.append("[XX] BSS < 0: Worse than climatology")
    
    if metrics['reliability_correlation'] > 0.9:
        report.append("[OK] Reliability correlation > 0.9: Strong monotonic relationship")
    elif metrics['reliability_correlation'] > 0.7:
        report.append("[!!] Reliability correlation 0.7-0.9: Moderate relationship")
    else:
        report.append("[XX] Reliability correlation < 0.7: Weak relationship")
    
    report_text = "\n".join(report)
    print(report_text)
    
    return report_text


if __name__ == "__main__":
    # Test calibration analysis
    print("Testing calibration analysis...")
    
    # Generate synthetic predictions with some miscalibration
    np.random.seed(42)
    n = 5000
    y_true = np.random.binomial(1, 0.1, n)
    
    # Well-calibrated predictions
    y_proba_good = y_true * 0.7 + np.random.random(n) * 0.25
    
    # Miscalibrated (overconfident) predictions
    y_proba_bad = np.clip(y_proba_good * 1.5, 0, 1)
    
    print("\n--- Well-Calibrated Model ---")
    print_calibration_report(y_true, y_proba_good, "WELL-CALIBRATED MODEL")
    
    print("\n--- Overconfident Model ---")
    print_calibration_report(y_true, y_proba_bad, "OVERCONFIDENT MODEL")
    
    # Create reliability diagrams
    print("\nGenerating reliability diagrams...")
    
    fig1 = plot_reliability_diagram(
        y_true, y_proba_good, 
        title="Well-Calibrated Model",
        filename="results/figures/calibration_good.png"
    )
    
    fig2 = plot_reliability_diagram(
        y_true, y_proba_bad,
        title="Overconfident Model",
        filename="results/figures/calibration_bad.png"
    )
    
    plt.show()