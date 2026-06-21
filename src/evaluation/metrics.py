"""
PRADHAN Evaluation — Proper Metrics
====================================

Metrics suitable for imbalanced rare-event classification:
- TSS (True Skill Statistic) — primary metric
- HSS (Heidke Skill Score) — secondary
- AUC (Area Under ROC Curve)
- PR-AUC (Area Under Precision-Recall Curve)
- Brier Score (calibration quality)
- Maximum Critical Success Index
- Lead Time (warning time before event onset)

Reference: Woodcock & Jolliffe (2008) for metric selection
in imbalanced geophysical forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix
)


def compute_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute True Skill Statistic (TSS).
    
    TSS = TPR - FPR = TP/(TP+FN) - FP/(FP+TN)
    
    Range: [-1, 1]
    Perfect: 1
    Random: 0
    Anti-skill: -1
    
    NOTE: TSS is PREFERRED over accuracy for rare events
    because it is not affected by the base rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    tss = tpr - fpr
    return float(tss)


def compute_hss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Heidke Skill Score (HSS).
    
    HSS = 2(AD - BC) / [(A+C)(B+D) + (A+B)(C+D)]
    
    Where:
    A = TP, B = FP, C = FN, D = TN
    
    Range: (-∞, 1]
    Perfect: 1
    Random: 0
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # HSS formula
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    
    if denominator == 0:
        return 0.0
    
    hss = numerator / denominator
    return float(hss)


def compute_pod(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Probability of Detection (Hit Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def compute_pofd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Probability of False Detection (False Alarm Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def compute_csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Critical Success Index (CSI) / Threat Score.
    
    CSI = TP / (TP + FP + FN)
    
    Useful for comparing models when events are rare.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0


def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.
    
    Range: [0, 1]
    Perfect: 1
    Random: 0.5
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return auc(fpr, tpr)


def compute_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    More informative than ROC-AUC for imbalanced data.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Brier Score.
    
    BS = (1/N) * Σ(prob - label)²
    
    Range: [0, 0.25] for rare events
    Perfect: 0
    Random: base_rate * (1 - base_rate)
    """
    return brier_score_loss(y_true, y_proba)


def compute_base_rate_brier(y_true: np.ndarray) -> float:
    """Compute the Brier score for climatological (base rate) predictions."""
    base_rate = np.mean(y_true)
    return brier_score_loss(y_true, np.full(len(y_true), base_rate))


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all metrics at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'threshold': threshold,
        'accuracy': float(np.mean(y_pred == y_true)),
        'tss': compute_tss(y_true, y_pred),
        'hss': compute_hss(y_true, y_pred),
        'pod': compute_pod(y_true, y_pred),
        'pofd': compute_pofd(y_true, y_pred),
        'csi': compute_csi(y_true, y_pred),
        'precision': float(np.mean(y_pred[y_pred == 1] == y_true[y_pred == 1])) if y_pred.sum() > 0 else 0,
        'recall': compute_pod(y_true, y_pred),
        'n_positive_pred': int(y_pred.sum()),
        'n_positive_true': int(y_true.sum()),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: float = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for flare forecasting.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities
    optimal_threshold : float, optional
        If provided, optimize threshold for TSS first
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Find optimal threshold for TSS
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    tss_scores = tpr - fpr
    
    if optimal_threshold is None:
        optimal_idx = np.argmax(tss_scores)
        optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    metrics = {
        # AUC metrics
        'auc': compute_auc(y_true, y_proba),
        'pr_auc': compute_pr_auc(y_true, y_proba),
        
        # Brier score (calibration)
        'brier': compute_brier_score(y_true, y_proba),
        'brier_climatological': compute_base_rate_brier(y_true),
        
        # Threshold-optimized metrics
        'optimal_threshold': float(optimal_threshold),
        'tss': compute_tss(y_true, y_pred),
        'hss': compute_hss(y_true, y_pred),
        'pod': compute_pod(y_true, y_pred),
        'pofd': compute_pofd(y_true, y_pred),
        'csi': compute_csi(y_true, y_pred),
        
        # Confusion matrix counts
        'n_samples': len(y_true),
        'n_positive_true': int(y_true.sum()),
        'n_positive_pred': int(y_pred.sum()),
        'event_rate': float(y_true.mean()),
    }
    
    return metrics


def compute_skill_score(
    model_brier: float,
    climatological_brier: float
) -> float:
    """
    Compute Brier Skill Score (BSS).
    
    BSS = 1 - (BS_model / BS_climatological)
    
    Range: (-∞, 1]
    Perfect: 1
    Equal to climatology: 0
    Worse than climatology: negative
    """
    if climatological_brier == 0:
        return 0.0
    return 1 - (model_brier / climatological_brier)


def compute_lead_times(
    prediction_times: pd.DatetimeIndex,
    predicted_proba: np.ndarray,
    event_times: pd.DatetimeIndex,
    threshold: float = 0.5,
    max_lead_minutes: int = 60
) -> Dict[str, float]:
    """
    Compute lead times for correct predictions.
    
    Lead time is the warning time: how far in advance the model
    correctly predicts an event before it occurs.
    
    Parameters
    ----------
    prediction_times : pd.DatetimeIndex
        Timestamps of each prediction
    predicted_proba : np.ndarray
        Predicted probabilities at each time
    event_times : pd.DatetimeIndex
        Actual onset times of flare events
    threshold : float
        Probability threshold for positive prediction
    max_lead_minutes : int
        Maximum lead time to consider (minutes)
        
    Returns
    -------
    dict
        Dictionary with lead time statistics:
        - mean_lead_minutes: Mean lead time
        - median_lead_minutes: Median lead time
        - min_lead_minutes: Minimum lead time
        - max_lead_minutes: Maximum lead time
        - n_correct_predictions: Number of correct early predictions
    """
    # Find times where model predicts positive
    positive_mask = predicted_proba >= threshold
    positive_times = prediction_times[positive_mask]
    
    if len(positive_times) == 0 or len(event_times) == 0:
        return {
            'mean_lead_minutes': 0.0,
            'median_lead_minutes': 0.0,
            'min_lead_minutes': 0.0,
            'max_lead_minutes': 0.0,
            'n_correct_predictions': 0,
        }
    
    lead_times = []
    
    for event_time in event_times:
        # Find predictions made BEFORE this event
        before_mask = positive_times < event_time
        if not before_mask.any():
            continue
        
        # Get the latest prediction before event onset
        valid_predictions = positive_times[before_mask]
        latest_prediction = valid_predictions.max()
        
        # Compute lead time
        lead_minutes = (event_time - latest_prediction).total_seconds() / 60
        
        if 0 < lead_minutes <= max_lead_minutes:
            lead_times.append(lead_minutes)
    
    if not lead_times:
        return {
            'mean_lead_minutes': 0.0,
            'median_lead_minutes': 0.0,
            'min_lead_minutes': 0.0,
            'max_lead_minutes': 0.0,
            'n_correct_predictions': 0,
        }
    
    lead_times = np.array(lead_times)
    
    return {
        'mean_lead_minutes': float(lead_times.mean()),
        'median_lead_minutes': float(np.median(lead_times)),
        'min_lead_minutes': float(lead_times.min()),
        'max_lead_minutes': float(lead_times.max()),
        'n_correct_predictions': int(len(lead_times)),
    }


def compute_warning_efficiency(
    prediction_times: pd.DatetimeIndex,
    predicted_proba: np.ndarray,
    event_times: pd.DatetimeIndex,
    threshold: float = 0.5,
    warning_window_minutes: int = 60
) -> Dict[str, float]:
    """
    Compute warning efficiency metrics.
    
    Measures how efficiently the model warns before events.
    
    Parameters
    ----------
    prediction_times : pd.DatetimeIndex
        Timestamps of each prediction
    predicted_proba : np.ndarray
        Predicted probabilities
    event_times : pd.DatetimeIndex
        Actual event onset times
    threshold : float
        Probability threshold for warning
    warning_window_minutes : int
        Time window to consider for valid warnings
        
    Returns
    -------
    dict
        Warning efficiency metrics
    """
    positive_mask = predicted_proba >= threshold
    positive_times = prediction_times[positive_mask]
    
    if len(event_times) == 0:
        return {
            'events_with_warnings': 0,
            'events_without_warnings': 0,
            'warning_rate': 0.0,
            'false_alarm_rate': 0.0,
            'mean_warning_time': 0.0,
        }
    
    events_with_warnings = 0
    warning_times = []
    
    for event_time in event_times:
        # Check if any positive prediction within warning window before event
        time_diffs = (event_time - positive_times).total_seconds() / 60
        valid_warnings = (time_diffs > 0) & (time_diffs <= warning_window_minutes)
        
        if valid_warnings.any():
            events_with_warnings += 1
            # Use the earliest valid warning
            warning_times.append(time_diffs[valid_warnings].min())
    
    # Count false alarms: positive predictions not near any event
    false_alarms = 0
    for pred_time in positive_times:
        time_diffs = np.abs((pred_time - event_times).total_seconds() / 60)
        if time_diffs.min() > warning_window_minutes:
            false_alarms += 1
    
    events_without_warnings = len(event_times) - events_with_warnings
    warning_rate = events_with_warnings / len(event_times) if len(event_times) > 0 else 0
    false_alarm_rate = false_alarms / len(positive_times) if len(positive_times) > 0 else 0
    mean_warning_time = np.mean(warning_times) if warning_times else 0
    
    return {
        'events_with_warnings': events_with_warnings,
        'events_without_warnings': events_without_warnings,
        'warning_rate': float(warning_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'mean_warning_time': float(mean_warning_time),
    }


def print_metrics_report(metrics: Dict[str, float], title: str = "METRICS REPORT"):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    
    # AUC metrics
    print(f"\n  --- Discrimination ---")
    print(f"  {'AUC-ROC':<25} {metrics.get('auc', 0):>12.4f}")
    print(f"  {'AUC-PR':<25} {metrics.get('pr_auc', 0):>12.4f}")
    
    # Skill scores
    print(f"\n  --- Skill Scores ---")
    print(f"  {'TSS':<25} {metrics.get('tss', 0):>12.4f}")
    print(f"  {'HSS':<25} {metrics.get('hss', 0):>12.4f}")
    
    # Component metrics
    print(f"\n  --- Components ---")
    print(f"  {'POD (Hit Rate)':<25} {metrics.get('pod', 0):>12.4f}")
    print(f"  {'POFD (FAR)':<25} {metrics.get('pofd', 0):>12.4f}")
    print(f"  {'CSI':<25} {metrics.get('csi', 0):>12.4f}")
    
    # Calibration
    print(f"\n  --- Calibration ---")
    print(f"  {'Brier Score':<25} {metrics.get('brier', 0):>12.4f}")
    print(f"  {'Brier (Climatology)':<25} {metrics.get('brier_climatological', 0):>12.4f}")
    
    bss = compute_skill_score(
        metrics.get('brier', 0),
        metrics.get('brier_climatological', 0)
    )
    print(f"  {'Brier Skill Score':<25} {bss:>12.4f}")
    print(f"  {'Optimal Threshold':<25} {metrics.get('optimal_threshold', 0):>12.4f}")
    
    # Counts
    print(f"\n  --- Counts ---")
    print(f"  {'N Samples':<25} {metrics.get('n_samples', 0):>12,}")
    print(f"  {'Event Rate':<25} {metrics.get('event_rate', 0):>11.2%}")
    print(f"  {'True Positives':<25} {metrics.get('n_positive_true', 0):>12,}")
    print(f"  {'Predicted Positives':<25} {metrics.get('n_positive_pred', 0):>12,}")
    
    # Lead time (if available)
    if 'lead_time' in metrics:
        lt = metrics['lead_time']
        print(f"\n  --- Lead Time ---")
        print(f"  {'Mean Lead Time':<25} {lt.get('mean_lead_minutes', 0):>11.1f} min")
        print(f"  {'Median Lead Time':<25} {lt.get('median_lead_minutes', 0):>11.1f} min")
        print(f"  {'Correct Warnings':<25} {lt.get('n_correct_predictions', 0):>12,}")
    
    # Warning efficiency (if available)
    if 'warning_efficiency' in metrics:
        we = metrics['warning_efficiency']
        print(f"\n  --- Warning Efficiency ---")
        print(f"  {'Events with Warnings':<25} {we.get('events_with_warnings', 0):>12,}")
        print(f"  {'Warning Rate':<25} {we.get('warning_rate', 0):>11.2%}")
        print(f"  {'False Alarm Rate':<25} {we.get('false_alarm_rate', 0):>11.2%}")
        print(f"  {'Mean Warning Time':<25} {we.get('mean_warning_time', 0):>11.1f} min")
    
    print("\n" + "=" * 60)


def compare_with_baselines(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    climatological_rate: float = None
) -> pd.DataFrame:
    """
    Compare model performance with baseline methods.
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    if climatological_rate is None:
        climatological_rate = y_true.mean()
    
    baselines = {
        'PRADHAN (XGBoost)': y_proba,
        'Random (Base Rate)': np.full(len(y_true), climatological_rate),
        'Persistence': np.roll(y_true, 1),  # Simplified
    }
    
    results = []
    for name, probs in baselines.items():
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        tss_scores = tpr - fpr
        optimal_idx = np.argmax(tss_scores)
        optimal_thresh = thresholds[optimal_idx]
        
        y_pred = (probs >= optimal_thresh).astype(int)
        
        metrics = compute_all_metrics(y_true, probs, optimal_thresh)
        metrics['method'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('method')
    
    return df[['auc', 'pr_auc', 'tss', 'hss', 'brier', 'optimal_threshold']]


if __name__ == "__main__":
    # Quick test
    print("Testing evaluation metrics...")
    
    # Generate sample predictions
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.1, n)  # 10% event rate
    y_proba = y_true * 0.7 + np.random.random(n) * 0.3  # Some skill
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_proba)
    print_metrics_report(metrics)
    
    # Compare with baselines
    print("\nComparing with baselines:")
    comparison = compare_with_baselines(y_true, y_proba)
    print(comparison)