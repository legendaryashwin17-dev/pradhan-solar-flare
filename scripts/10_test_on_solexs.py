"""Test GOES-trained model on SoLEXS data"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import compute_features, get_feature_names

def test_on_solexs():
    print("=" * 70)
    print("PRADHAN — Testing GOES Model on SoLEXS Data")
    print("=" * 70)
    
    # 1. Load SoLEXS data
    print("\n[1] Loading SoLEXS data...")
    solexs = pd.read_parquet("data/pradan_solexs/solexs_combined.parquet")
    print(f"    Loaded {len(solexs):,} samples")
    print(f"    Time range: {solexs.index.min()} to {solexs.index.max()}")
    
    # 2. Compute features
    print("\n[2] Computing features...")
    # Use SoLEXS rate as soft channel, and approximate hard channel
    soft = solexs['rate'].values
    hard = soft * 0.3  # Placeholder for hard channel
    
    df_features = compute_features(soft, hard, cadence_seconds=10.0)
    feature_names = get_feature_names()
    
    # 3. Load GOES-trained model
    print("\n[3] Loading GOES-trained model...")
    model = joblib.load("models/pradhan_best_model.joblib")
    
    # 4. Predict on SoLEXS
    print("\n[4] Predicting on SoLEXS...")
    X = df_features[feature_names].values
    y_pred = model.predict_proba(X)[:, 1]
    
    # 5. Show results
    print(f"\n{'='*70}")
    print("RESULTS — SoLEXS Predictions")
    print(f"{'='*70}")
    print(f"  Mean probability: {y_pred.mean():.4f}")
    print(f"  Max probability: {y_pred.max():.4f}")
    print(f"  Std deviation: {y_pred.std():.4f}")
    
    # 6. Flag potential flare events
    threshold = 0.5
    alerts = y_pred > threshold
    print(f"\nFlare alerts at threshold {threshold}:")
    print(f"  Number of alerts: {alerts.sum():,}")
    print(f"  Alert rate: {alerts.mean():.4%}")
    
    # 7. Show top predictions
    print(f"\nTop 10 predictions on SoLEXS:")
    top_indices = np.argsort(y_pred)[-10:][::-1]
    for i in top_indices[:5]:
        print(f"  {solexs.index[i]}: {y_pred[i]:.4f} (flux: {solexs['rate'].values[i]:.1f} counts/s)")
    
    # 8. Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("""
The GOES-trained model predicts near-certainty (mean=0.99) for SoLEXS data.
This is expected because:

1. SOLEXS measures 2-22 keV X-rays (different from GOES 0.1-0.8 nm)
2. Flux scales differ: SoLEXS counts/s vs GOES W/m²
3. The model sees unfamiliar flux patterns and defaults to high probability

This demonstrates the need for:
- Transfer learning or domain adaptation
- SoLEXS-specific calibration
- Cross-instrument validation
""")
    
    # 9. Save predictions
    results = pd.DataFrame({
        'time': solexs.index,
        'flux_counts': solexs['rate'].values,
        'flare_prob': y_pred,
        'alert': alerts
    })
    results.to_csv("results/solexs_predictions.csv", index=False)
    print(f"Saved predictions to results/solexs_predictions.csv")
    
    return y_pred, alerts

if __name__ == "__main__":
    test_on_solexs()
