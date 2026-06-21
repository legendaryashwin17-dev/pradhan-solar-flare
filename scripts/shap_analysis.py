"""
PRADHAN SHAP Analysis — Model Interpretability
===============================================

Uses SHAP (SHapley Additive exPlanations) to explain
model predictions. This provides:
1. Feature importance rankings
2. Feature interaction effects
3. Individual prediction explanations
4. Global model behavior

SHAP is the gold standard for ML interpretability
(Lundberg & Lee, NeurIPS 2017).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. Install with: pip install shap")


def run_shap_analysis():
    """Run full SHAP analysis on trained model."""
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return
    
    print("=" * 70)
    print("PRADHAN — SHAP Interpretability Analysis")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n[1] Loading GOES data...")
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    sample = goes.iloc[:50000]  # Use subset for SHAP speed
    
    soft = sample['xrs_a_flux'].values
    hard = sample['xrs_b_flux'].values
    df_features = compute_features(soft, hard, cadence_seconds=60.0)
    df_features.index = sample.index
    feature_names = get_feature_names()
    
    flux = sample['xrs_b_flux']
    y = create_flare_labels(flux, horizon='6h', threshold_class='M')
    
    valid = ~df_features[feature_names].isna().any(axis=1) & ~y.isna()
    X = df_features.loc[valid, feature_names].values
    y_clean = y[valid].values
    
    # 2. Train model
    print("\n[2] Training model...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
    
    model = FlareForecaster(scale_pos_weight=50)
    model.fit(X_train, y_train, feature_names)
    
    # 3. Compute SHAP values
    print("\n[3] Computing SHAP values (this may take a minute)...")
    
    # Use TreeExplainer for XGBoost (exact, fast)
    explainer = shap.TreeExplainer(model.model)
    
    # Compute on test set (use subset for speed)
    X_shap = X_test[:1000]
    shap_values = explainer.shap_values(X_shap)
    
    # 4. Global feature importance
    print("\n[4] Global Feature Importance (SHAP):")
    print("-" * 50)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_idx = np.argsort(mean_abs_shap)[::-1]
    
    for i, idx in enumerate(importance_idx):
        name = feature_names[idx]
        importance = mean_abs_shap[idx]
        bar = '█' * min(40, int(importance / mean_abs_shap[importance_idx[0]] * 40))
        print(f"  {i+1:>2}. {name:<25} {importance:>8.4f}  {bar}")
    
    # 5. Feature interactions (top 5)
    print("\n[5] Top Feature Interactions:")
    print("-" * 50)
    
    try:
        shap_interaction = explainer.shap_interaction_values(X_shap[:200])
        mean_interaction = np.abs(shap_interaction).mean(axis=0)
        
        # Find top interactions (excluding diagonal)
        np.fill_diagonal(mean_interaction, 0)
        flat_idx = np.argsort(mean_interaction.ravel())[::-1][:10]
        
        for i, idx in enumerate(flat_idx):
            row, col = np.unravel_index(idx, mean_interaction.shape)
            name1 = feature_names[row]
            name2 = feature_names[col]
            value = mean_interaction[row, col]
            print(f"  {i+1:>2}. {name1} × {name2}: {value:.4f}")
    except Exception as e:
        print(f"  Interaction computation skipped: {e}")
    
    # 6. Individual prediction explanations
    print("\n[6] Individual Prediction Examples:")
    print("-" * 50)
    
    # Find a positive and negative prediction
    y_pred = model.predict_proba(X_test[:1000])
    
    pos_idx = np.where(y_pred > 0.5)[0]
    neg_idx = np.where(y_pred < 0.1)[0]
    
    if len(pos_idx) > 0:
        idx = pos_idx[0]
        print(f"\n  High-probability prediction (p={y_pred[idx]:.4f}):")
        top_features = np.argsort(np.abs(shap_values[idx]))[::-1][:5]
        for f_idx in top_features:
            print(f"    {feature_names[f_idx]:<25} = {X_shap[idx, f_idx]:>10.4f} "
                  f"(SHAP: {shap_values[idx, f_idx]:>+.4f})")
    
    if len(neg_idx) > 0:
        idx = neg_idx[0]
        print(f"\n  Low-probability prediction (p={y_pred[idx]:.4f}):")
        top_features = np.argsort(np.abs(shap_values[idx]))[::-1][:5]
        for f_idx in top_features:
            print(f"    {feature_names[f_idx]:<25} = {X_shap[idx, f_idx]:>10.4f} "
                  f"(SHAP: {shap_values[idx, f_idx]:>+.4f})")
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTop 5 most important features:")
    for i, idx in enumerate(importance_idx[:5]):
        print(f"  {i+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")
    
    print(f"\nSHAP provides mathematically optimal feature attribution")
    print(f"based on cooperative game theory (Shapley values).")
    
    return shap_values, importance_idx


if __name__ == "__main__":
    run_shap_analysis()
