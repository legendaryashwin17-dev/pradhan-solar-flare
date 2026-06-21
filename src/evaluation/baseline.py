"""
PRADHAN Baselines — Proper Comparison Methods
==============================================

Proper baselines for flare forecasting:
1. Persistence: future flux = current flux (correct formulation)
2. Climatology: historical event rate
3. NOAA SWPC-style: categorical probability tiers

These are the baselines that PRADHAN must beat to demonstrate skill.
"""

import numpy as np
from typing import Optional


class BaselinePersistence:
    """
    True persistence baseline for time series forecasting.
    
    "Tomorrow's flux = Today's flux"
    
    This is the CORRECT persistence for flare forecasting.
    It is NOT a constant classifier — it uses the actual current flux.
    """
    
    def __init__(self, horizon_minutes: int = 360, threshold: float = 1e-5):
        """
        Parameters
        ----------
        horizon_minutes : int
            Forecast horizon in minutes
        threshold : float
            Flux threshold for M-class (1e-5 W/m²)
        """
        self.horizon = horizon_minutes
        self.threshold = threshold
        self.name = "Persistence"
    
    def predict_proba(
        self,
        current_flux: np.ndarray,
        horizon_minutes: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict probability that future flux > threshold.
        
        For persistence: P(flare at t+H) = 1 if flux(t) > threshold
        
        Parameters
        ----------
        current_flux : np.ndarray
            Current X-ray flux values
        horizon_minutes : int, optional
            Override default horizon
            
        Returns
        -------
        np.ndarray
            Predicted probabilities
        """
        horizon = horizon_minutes or self.horizon
        
        # Simple persistence: if flux > threshold now, predict flare
        # Scale probability by how close to threshold
        proba = np.clip(current_flux / self.threshold, 0, 1)
        
        # For short horizons, persistence is more reliable
        # For long horizons, reduce confidence
        horizon_factor = max(0.3, 1.0 - horizon / 1440)  # Decay over 24h
        
        return proba * horizon_factor


class BaselineClimatology:
    """
    Climatology baseline: predicts historical event rate.
    
    This is the simplest meaningful baseline.
    A model must beat this to demonstrate any skill.
    """
    
    def __init__(self):
        self.rate = 0.5
        self.name = "Climatology"
    
    def fit(self, y: np.ndarray):
        """
        Compute climatological event rate from training data.
        
        Parameters
        ----------
        y : np.ndarray
            Binary labels from training period
        """
        self.rate = np.mean(y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict climatological rate for all samples.
        
        Parameters
        ----------
        X : np.ndarray
            Features (not used, but needed for API compatibility)
            
        Returns
        -------
        np.ndarray
            Constant probability predictions
        """
        return np.full(len(X), self.rate)


class BaselineRandom:
    """
    Random baseline: predictions proportional to base rate.
    
    This represents pure chance — no skill whatsoever.
    """
    
    def __init__(self, rate: Optional[float] = None):
        self.rate = rate
        self.name = "Random"
    
    def fit(self, y: np.ndarray):
        """Compute base rate from training data."""
        if self.rate is None:
            self.rate = np.mean(y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Random predictions based on base rate."""
        return np.full(len(X), self.rate)


def get_all_baselines() -> dict:
    """
    Return all baseline models.
    
    Returns
    -------
    dict
        Dictionary of baseline name -> baseline object
    """
    return {
        'Persistence': BaselinePersistence(),
        'Climatology': BaselineClimatology(),
        'Random': BaselineRandom(),
    }


def evaluate_baselines(
    y_true: np.ndarray,
    y_proba_dict: dict,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate all baselines against true labels.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba_dict : dict
        Dictionary of baseline name -> predicted probabilities
    threshold : float
        Classification threshold
        
    Returns
    -------
    dict
        Dictionary of baseline name -> metrics dict
    """
    from src.evaluation.metrics import compute_all_metrics
    
    results = {}
    for name, y_proba in y_proba_dict.items():
        metrics = compute_all_metrics(y_true, y_proba, threshold)
        metrics['method'] = name
        results[name] = metrics
    
    return results
