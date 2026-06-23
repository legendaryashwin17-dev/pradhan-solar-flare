"""
Train Best Model (1h C-class) and Save
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, FIGURES_DIR
from src.data.reader import load_goes_parquet
from src.data.features import compute_features, get_feature_names
from src.data.labels import create_flare_labels
from src.models.forecaster import FlareForecaster
from src.evaluation.metrics import compute_all_metrics, print_metrics_report


def main():
    print("=" * 70)
    print("PRADHAN — Training Best Model (1h C-class)")
    print("=" * 70)

    mlflow.set_experiment("pradhan-solar-flare")

    with mlflow.start_run(run_name="best-model-1h-c-class"):
        # Load data
        print("\n[1] Loading GOES data...")
        goes = load_goes_parquet(str(DATA_DIR / "goes"))
        print(f"    Loaded {len(goes):,} records")

        # Compute features
        print("\n[2] Computing features...")
        soft = goes['xrs_a_flux'].values
        hard = goes['xrs_b_flux'].values
        features_df = compute_features(soft, hard, cadence_seconds=60.0)
        feature_names = get_feature_names()
        features_df.index = goes.index

        # Create labels
        print("\n[3] Creating labels (1h C-class)...")
        flux = goes['xrs_b_flux']
        y = create_flare_labels(flux, horizon='1h', threshold_class='C')
        print(f"    Event rate: {y.mean():.4%}")

        # Prepare data
        valid = ~(features_df[feature_names].isna().any(axis=1) | y.isna())
        X_all = features_df.loc[valid, feature_names].values
        y_all = y[valid].values
        times_all = features_df.loc[valid].index

        # Chronological split
        split_idx = int(len(X_all) * 0.8)
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]
        times_train, times_test = times_all[:split_idx], times_all[split_idx:]

        print(f"\n[4] Training set: {len(X_train):,} samples")
        print(f"    Training period: {times_train[0]} to {times_train[-1]}")
        print(f"    Test set: {len(X_test):,} samples")
        print(f"    Test period: {times_test[0]} to {times_test[-1]}")

        # Log parameters
        mlflow.log_param("horizon", "1h")
        mlflow.log_param("threshold_class", "C")
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("train_event_rate", float(y_train.mean()))
        mlflow.log_param("test_event_rate", float(y_test.mean()))

        # Train
        print("\n[5] Training XGBoost model...")
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / max(n_pos, 1)
        print(f"    Class weight: {spw:.1f}x")

        mlflow.log_param("scale_pos_weight", spw)

        model = FlareForecaster(scale_pos_weight=spw)
        model.fit(X_train, y_train, feature_names)

        # Evaluate
        print("\n[6] Evaluating...")
        y_pred = model.predict_proba(X_test)
        optimal_threshold = model.optimize_threshold(X_test, y_test)
        metrics = compute_all_metrics(y_test, y_pred, optimal_threshold)
        print_metrics_report(metrics, "BEST MODEL (1h C-class)")

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, (np.floating, float, int)):
                mlflow.log_metric(k, float(v))
        mlflow.log_metric("optimal_threshold", float(optimal_threshold))

        # Log model
        mlflow.xgboost.log_model(model.model, "xgboost-model")

        # Save model
        print("\n[7] Saving model...")
        model.save("models/pradhan_best")
        print("    Saved to models/pradhan_best")

        # Save results
        results = {
            'config': '1h C-class',
            'horizon': '1h',
            'threshold_class': 'C',
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in metrics.items()},
            'optimal_threshold': float(optimal_threshold),
            'data_info': {
                'n_train': len(X_train),
                'n_test': len(X_test),
                'train_event_rate': float(y_train.mean()),
                'test_event_rate': float(y_test.mean()),
                'train_period': f"{times_train[0]} to {times_train[-1]}",
                'test_period': f"{times_test[0]} to {times_test[-1]}",
            },
            'feature_importance': {
                row['feature']: float(row['importance'])
                for _, row in model.get_feature_importance().iterrows()
            }
        }

        with open("results/best_model_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("    Results saved to results/best_model_results.json")

        mlflow.log_artifact("results/best_model_results.json")

        print("\n" + "=" * 70)
        print("DONE — Best model trained and saved")
        print("MLflow run logged to: mlflow ui")
        print("=" * 70)

        return model, metrics


if __name__ == "__main__":
    main()
