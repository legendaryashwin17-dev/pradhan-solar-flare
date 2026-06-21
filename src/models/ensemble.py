"""
PRADHAN Ensemble — Multi-Model with Working Weight Optimization
===============================================================

Properly implements ensemble with optimized weights.

NOTE: For a hackathon demo, a single well-tuned XGBoost often matches
ensembles. The ensemble is included for completeness but the single
model should be the primary demonstration.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional
from scipy.optimize import minimize

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class FlareEnsemble:
    """
    Ensemble of multiple models for flare forecasting.
    
    Uses weighted averaging with optimized weights.
    
    Models included:
    - XGBoost (primary)
    - LightGBM
    - Random Forest
    - Gradient Boosting
    - Logistic Regression (for calibration reference)
    
    NOTE: For rare events like flares, the marginal benefit of
    ensemble over a well-tuned single model is often small.
    """
    
    def __init__(self, config: dict = None):
        default_config = {
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'scale_pos_weight': 20,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'scale_pos_weight': 20,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'logistic': {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
        }
        
        self.config = config or default_config
        self.models = {}
        self.weights = None
        self.trained = False
        self.feature_names = None
        self._init_models()
    
    def _init_models(self):
        """Initialize model instances."""
        self.models = {}
        
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBClassifier(
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss',
                **self.config.get('xgboost', {})
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = LGBMClassifier(
                verbosity=-1,
                **self.config.get('lightgbm', {})
            )
        
        self.models['random_forest'] = RandomForestClassifier(
            **self.config.get('random_forest', {})
        )
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            **self.config.get('gradient_boosting', {})
        )
        self.models['logistic'] = LogisticRegression(
            **self.config.get('logistic', {})
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        validation_split: float = 0.2,
        optimize_weights: bool = True
    ) -> 'FlareEnsemble':
        """
        Train ensemble with optional weight optimization.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Binary labels
        feature_names : list, optional
            Feature names for interpretability
        validation_split : float
            Fraction of data for weight optimization
        optimize_weights : bool
            If True, optimize weights to minimize Brier score
        """
        self.feature_names = feature_names
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training ensemble on {len(X_train)} samples")
        print(f"Validation on {len(X_val)} samples")
        
        # Train each model and collect validation predictions
        val_predictions = {}
        for name, model in self.models.items():
            print(f"  Training {name}...", end=" ")
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_val)[:, 1]
            val_predictions[name] = pred
            print(f"done")
        
        # Compute optimal weights
        if optimize_weights:
            self.weights = self._optimize_weights(val_predictions, y_val)
        else:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        print(f"\nOptimized weights:")
        for name, w in zip(self.models.keys(), self.weights):
            print(f"  {name}: {w:.4f}")
        
        # Retrain on full data with optimized weights
        print("\nRetraining on full data...")
        for name, model in self.models.items():
            model.fit(X, y)
        
        self.trained = True
        return self
    
    def _optimize_weights(
        self,
        predictions: dict,
        y_true: np.ndarray
    ) -> np.ndarray:
        """
        Optimize ensemble weights using Brier score minimization.
        
        Uses scipy minimize with constraints to ensure weights sum to 1.
        """
        # Stack predictions: (n_models, n_samples)
        preds_array = np.array(list(predictions.values()))
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        def brier_score(weights):
            """Compute Brier score for given weights."""
            weights = np.array(weights)
            weights = np.maximum(weights, 0)  # No negative weights
            weights = weights / (weights.sum() + 1e-10)  # Normalize
            
            ensemble_pred = np.dot(weights, preds_array)
            brier = np.mean((ensemble_pred - y_true) ** 2)
            return brier
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            brier_score,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        # Normalize final weights
        optimal_weights = np.maximum(result.x, 0)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using weighted ensemble.
        
        Returns
        -------
        np.ndarray
            Ensemble probabilities
        """
        if not self.trained:
            raise ValueError("Ensemble not trained yet")
        
        preds = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            preds.append(pred)
        
        preds = np.array(preds)
        ensemble_pred = np.dot(self.weights, preds)
        
        return ensemble_pred
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_bootstrap: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification via bootstrap.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (mean, std, lower_95, upper_95)
        """
        if not self.trained:
            raise ValueError("Ensemble not trained yet")
        
        model_list = list(self.models.values())
        model_names = list(self.models.keys())
        n_models = len(model_list)
        
        bootstrap_preds = []
        
        for _ in range(n_bootstrap):
            # Sample models with replacement
            indices = np.random.choice(n_models, n_models, replace=True)
            
            # Average predictions from sampled models
            boot_pred = np.zeros(len(X))
            for idx in indices:
                boot_pred += model_list[idx].predict_proba(X)[:, 1]
            boot_pred /= n_models
            
            bootstrap_preds.append(boot_pred)
        
        bootstrap_preds = np.array(bootstrap_preds)
        
        mean = bootstrap_preds.mean(axis=0)
        std = bootstrap_preds.std(axis=0)
        lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper = np.percentile(bootstrap_preds, 97.5, axis=0)
        
        return mean, std, lower, upper
    
    def individual_model_predictions(
        self,
        X: np.ndarray
    ) -> dict:
        """Get predictions from each model individually."""
        if not self.trained:
            raise ValueError("Ensemble not trained yet")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        return predictions
    
    def save(self, path: str):
        """Save ensemble to disk."""
        for name, model in self.models.items():
            joblib.dump(model, f"{path}_{name}.joblib")
        joblib.dump({
            'weights': self.weights,
            'config': self.config,
            'trained': self.trained,
            'feature_names': self.feature_names
        }, f"{path}_config.joblib")
    
    def load(self, path: str):
        """Load ensemble from disk."""
        for name in self.models.keys():
            try:
                self.models[name] = joblib.load(f"{path}_{name}.joblib")
            except FileNotFoundError:
                print(f"Warning: Model {name} not found")
        
        try:
            config = joblib.load(f"{path}_config.joblib")
            self.weights = config['weights']
            self.config = config['config']
            self.trained = config['trained']
            self.feature_names = config['feature_names']
        except FileNotFoundError:
            print("Warning: Config not found")


if __name__ == "__main__":
    # Quick test
    from ..data.reader import get_sample_data
    from ..data.features import compute_features, get_feature_names
    from ..data.labels import create_flare_labels
    
    print("Testing FlareEnsemble...")
    
    # Generate sample data
    df = get_sample_data(n_points=5000)
    soft = df['xrs_a_flux'].values
    hard = df['xrs_b_flux'].values
    
    # Compute features
    features = compute_features(soft, hard)
    feature_names = get_feature_names()
    
    # Create labels
    flux = df['xrs_b_flux']
    labels = create_flare_labels(flux, horizon='24h', threshold_class='M')
    
    # Use valid indices
    valid = ~(features[feature_names].isna().any(axis=1) | labels.isna())
    X = features.loc[valid, feature_names].values
    y = labels[valid].values
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Event rate: {y_train.mean():.4%}")
    
    # Train ensemble
    ensemble = FlareEnsemble()
    ensemble.fit(X_train, y_train, feature_names, optimize_weights=True)
    
    # Evaluate
    y_pred = ensemble.predict_proba(X_test)
    
    from ..evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics(y_test, y_pred)
    
    print(f"\nEnsemble Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Uncertainty
    mean, std, lower, upper = ensemble.predict_with_uncertainty(X_test[:10])
    print(f"\nSample predictions with uncertainty:")
    for i in range(5):
        print(f"  {i}: {mean[i]:.3f} ± {std[i]:.3f} [{lower[i]:.3f}, {upper[i]:.3f}]")