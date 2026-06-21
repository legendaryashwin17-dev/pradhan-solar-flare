"""
PRADHAN Transfer Validation — GOES to Aditya-L1
================================================

Validates whether GOES-trained models transfer to Aditya-L1 instruments.
This is critical because:
1. PRADHAN will use Aditya-L1 data (SoLEXS/HEL1OS)
2. Training data is from GOES
3. We need to know if features transfer across instruments

Method:
1. Train on GOES data
2. Extract features from SoLEXS/HEL1OS data
3. Apply GOES model to SoLEXS features
4. Compare with GOES-only performance
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reader import load_goes_parquet, load_solexs_lc
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics


def train_on_goes():
    """Train model on GOES data."""
    print("\n[1] Training on GOES data...")
    
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    sample = goes.iloc[:100000]
    
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
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
    
    model = FlareForecaster(scale_pos_weight=50)
    model.fit(X_train, y_train, feature_names)
    
    # Evaluate on GOES test set
    y_pred = model.predict_proba(X_test)
    goes_metrics = compute_all_metrics(y_test, y_pred)
    
    print(f"  GOES test TSS: {goes_metrics['tss']:.4f}")
    print(f"  GOES test AUC: {goes_metrics['auc']:.4f}")
    
    return model, goes_metrics


def load_solexs_features():
    """Load SoLEXS data and compute features."""
    print("\n[2] Loading SoLEXS data...")
    
    solexs_path = Path("data/pradan_solexs/extracted")
    if not solexs_path.exists():
        print("  SoLEXS data not found, skipping transfer validation")
        return None, None
    
    lc_files = sorted(solexs_path.glob("*.lc"))
    if not lc_files:
        print("  No SoLEXS .lc files found")
        return None, None
    
    print(f"  Found {len(lc_files)} SoLEXS files")
    
    # Load first file as sample
    try:
        df = load_solexs_lc(str(lc_files[0]))
        
        # SoLEXS has different energy ranges
        # Use rate as proxy for soft X-ray
        soft = df['rate'].values
        # Create synthetic hard channel (SoLEXS doesn't have one directly)
        hard = soft * 0.5 + np.random.normal(0, 0.1 * soft.mean(), len(soft))
        hard = np.maximum(hard, 1e-12)
        
        # Compute features
        df_features = compute_features(soft, hard, cadence_seconds=10.0)  # 10s cadence
        feature_names = get_feature_names()
        
        valid = ~df_features[feature_names].isna().any(axis=1)
        X_solexs = df_features.loc[valid, feature_names].values
        
        print(f"  SoLEXS features: {X_solexs.shape}")
        
        return X_solexs, feature_names
        
    except Exception as e:
        print(f"  Error loading SoLEXS: {e}")
        return None, None


def transfer_validation():
    """Run transfer validation."""
    print("=" * 70)
    print("PRADHAN — GOES to Aditya-L1 Transfer Validation")
    print("=" * 70)
    
    # 1. Train on GOES
    goes_model, goes_metrics = train_on_goes()
    
    # 2. Load SoLEXS features
    X_solexs, feature_names = load_solexs_features()
    
    if X_solexs is None:
        print("\nTransfer validation skipped (no SoLEXS data)")
        return
    
    # 3. Apply GOES model to SoLEXS
    print("\n[3] Applying GOES model to SoLEXS features...")
    solexs_pred = goes_model.predict_proba(X_solexs)
    
    print(f"\nSoLEXS prediction statistics:")
    print(f"  Mean probability: {solexs_pred.mean():.4f}")
    print(f"  Std probability:  {solexs_pred.std():.4f}")
    print(f"  Min: {solexs_pred.min():.4f}, Max: {solexs_pred.max():.4f}")
    
    # 4. Feature distribution comparison
    print("\n[4] Feature distribution comparison:")
    print(f"{'Feature':<25} {'GOES Mean':>12} {'SoLEXS Mean':>12} {'Ratio':>8}")
    print("-" * 60)
    
    # Load GOES features for comparison
    goes = load_goes_parquet("C:/Users/Admin/aditya-flare-forecast/data/goes_historical")
    sample = goes.iloc[:10000]
    soft_g = sample['xrs_a_flux'].values
    hard_g = sample['xrs_b_flux'].values
    goes_features = compute_features(soft_g, hard_g, cadence_seconds=60.0)
    
    for i, name in enumerate(feature_names[:10]):
        goes_mean = goes_features[name].mean()
        solexs_mean = np.mean(X_solexs[:, i]) if i < X_solexs.shape[1] else 0
        ratio = solexs_mean / (goes_mean + 1e-10)
        print(f"  {name:<23} {goes_mean:>12.4f} {solexs_mean:>12.4f} {ratio:>8.2f}")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("TRANSFER VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Model trained on: GOES XRS")
    print(f"  Applied to: SoLEXS (Aditya-L1)")
    print(f"  GOES test TSS: {goes_metrics['tss']:.4f}")
    print(f"  SoLEXS prediction mean: {solexs_pred.mean():.4f}")
    print(f"\n  Interpretation:")
    print(f"    - If SoLEXS predictions cluster near 0/1: good transfer")
    print(f"    - If predictions are flat: features don't transfer")
    print(f"    - Feature distribution ratios show domain shift")


if __name__ == "__main__":
    transfer_validation()
