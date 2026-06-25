#!/usr/bin/env python3
"""
PRADHAN Cross-Solar-Cycle Evaluation
======================================

Evaluates GOES expert performance across different time periods.
Uses time-based splits (not random) to simulate real forecasting.

Splits:
  1. Train on April-May 2026, test on June 2026 (within-cycle)
  2. Train on all 2026, test on June 2024 (cross-year)
  3. Train on 2024 + early 2026, test on late 2026 (combined)

Usage:
    python scripts/34_cross_cycle_eval.py
"""

import os
import sys
import json
import joblib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
FEATURES_PATH = os.path.join(WORKSPACE, 'data', 'historical', 'goes_cross_cycle_features.parquet')
OUTPUT_DIR = os.path.join(WORKSPACE, 'data', 'experiments', 'cross_cycle')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GOES_FEATURES = [
    'goes_log_xrsa', 'goes_log_xrsb', 'goes_xrsb_baseline',
    'goes_xrsb_log_grad', 'goes_xrsb_log_std', 'goes_xrsb_log_mean',
    'goes_xrsa_xrsb_ratio', 'goes_xrsb_log_zscore'
]


def load_features():
    """Load historical GOES features."""
    print(f'Loading features from {FEATURES_PATH}...')
    df = pd.read_parquet(FEATURES_PATH)
    print(f'  Loaded: {len(df)} samples')
    print(f'  Date range: {df.index[0]} to {df.index[-1]}')
    print(f'  M+ flare rate: {df["label_mflare"].mean():.1%}')
    print(f'  C+ flare rate: {df["label_cflare"].mean():.1%}')
    return df


def train_and_evaluate(X_train, y_train, X_test, y_test, label='GOES Expert'):
    """Train XGBoost and evaluate."""
    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5
    
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    # TSS (True Skill Statistic)
    pod = tp / max(tp + fn, 1)  # Probability of Detection
    pofd = fp / max(fp + tn, 1)  # Probability of False Detection
    tss = pod - pofd
    
    # HSS (Heidke Skill Score)
    expected = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / max(tp + fp + fn + tn, 1)
    hss = (tp + tn - expected) / max(tp + fp + fn + tn - expected, 1)
    
    results = {
        'label': label,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_pos_train': int(y_train.sum()),
        'n_pos_test': int(y_test.sum()),
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'tss': float(tss),
        'hss': float(hss),
        'pod': float(pod),
        'pofd': float(pofd),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'scale_pos_weight': float(scale_pos),
    }
    
    return model, results


def temporal_split(df, train_end, test_start):
    """Split data by time (no leakage)."""
    train = df[df.index < train_end]
    test = df[df.index >= test_start]
    return train, test


def main():
    print('=' * 60)
    print('PRADHAN Cross-Solar-Cycle Evaluation')
    print('=' * 60)
    
    df = load_features()
    
    all_results = []
    
    # Split 1: Train on April-May 2026, test on June 2026 (within-cycle)
    print('\n' + '=' * 60)
    print('SPLIT 1: Within-Cycle (Train: Apr-May 2026, Test: Jun 2026)')
    print('=' * 60)
    
    train1, test1 = temporal_split(
        df,
        train_end='2026-06-01',
        test_start='2026-06-01'
    )
    
    if len(train1) > 50 and len(test1) > 10:
        X_tr = train1[GOES_FEATURES].values
        y_tr = train1['label_mflare'].values
        X_te = test1[GOES_FEATURES].values
        y_te = test1['label_mflare'].values
        
        print(f'  Train: {len(train1)} samples ({y_tr.sum()} M+ flares)')
        print(f'  Test: {len(test1)} samples ({y_te.sum()} M+ flares)')
        
        model1, res1 = train_and_evaluate(X_tr, y_tr, X_te, y_te, 'GOES Within-Cycle')
        all_results.append(res1)
        print(f'  Results: TSS={res1["tss"]:.3f}, AUC={res1["auc"]:.3f}, POD={res1["pod"]:.3f}')
        print(f'  Confusion: TP={res1["tp"]}, FP={res1["fp"]}, TN={res1["tn"]}, FN={res1["fn"]}')
    else:
        print('  Insufficient data for this split')
    
    # Split 2: Train on all 2026, test on June 2024 (cross-year)
    print('\n' + '=' * 60)
    print('SPLIT 2: Cross-Year (Train: All 2026, Test: Jun 2024)')
    print('=' * 60)
    
    train2 = df[df.index >= '2026-01-01']
    test2 = df[df.index < '2025-01-01']
    
    if len(train2) > 50 and len(test2) > 10:
        X_tr = train2[GOES_FEATURES].values
        y_tr = train2['label_mflare'].values
        X_te = test2[GOES_FEATURES].values
        y_te = test2['label_mflare'].values
        
        print(f'  Train: {len(train2)} samples ({y_tr.sum()} M+ flares)')
        print(f'  Test: {len(test2)} samples ({y_te.sum()} M+ flares)')
        
        model2, res2 = train_and_evaluate(X_tr, y_tr, X_te, y_te, 'GOES Cross-Year')
        all_results.append(res2)
        print(f'  Results: TSS={res2["tss"]:.3f}, AUC={res2["auc"]:.3f}, POD={res2["pod"]:.3f}')
        print(f'  Confusion: TP={res2["tp"]}, FP={res2["fp"]}, TN={res2["tn"]}, FN={res2["fn"]}')
    else:
        print('  Insufficient data for this split')
    
    # Split 3: Train on June 2024 + April-May 2026, test on June 2026
    print('\n' + '=' * 60)
    print('SPLIT 3: Combined (Train: Jun 2024 + Apr-May 2026, Test: Jun 2026)')
    print('=' * 60)
    
    train3_early = df[df.index < '2025-01-01']  # June 2024
    train3_late = df[(df.index >= '2026-04-01') & (df.index < '2026-06-01')]  # Apr-May 2026
    test3 = df[df.index >= '2026-06-01']  # June 2026
    
    train3 = pd.concat([train3_early, train3_late])
    
    if len(train3) > 50 and len(test3) > 10:
        X_tr = train3[GOES_FEATURES].values
        y_tr = train3['label_mflare'].values
        X_te = test3[GOES_FEATURES].values
        y_te = test3['label_mflare'].values
        
        print(f'  Train: {len(train3)} samples ({y_tr.sum()} M+ flares)')
        print(f'  Test: {len(test3)} samples ({y_te.sum()} M+ flares)')
        
        model3, res3 = train_and_evaluate(X_tr, y_tr, X_te, y_te, 'GOES Combined')
        all_results.append(res3)
        print(f'  Results: TSS={res3["tss"]:.3f}, AUC={res3["auc"]:.3f}, POD={res3["pod"]:.3f}')
        print(f'  Confusion: TP={res3["tp"]}, FP={res3["fp"]}, TN={res3["tn"]}, FN={res3["fn"]}')
    else:
        print('  Insufficient data for this split')
    
    # Split 4: Random CV baseline (5-fold stratified)
    print('\n' + '=' * 60)
    print('SPLIT 4: Random 5-Fold CV Baseline (all data)')
    print('=' * 60)
    
    X_all = df[GOES_FEATURES].values
    y_all = df['label_mflare'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        
        model_cv, res_cv = train_and_evaluate(X_tr, y_tr, X_te, y_te, f'GOES CV-Fold{fold+1}')
        cv_results.append(res_cv)
    
    # Average CV results
    cv_avg = {
        'label': 'GOES 5-Fold CV (avg)',
        'n_train': int(np.mean([r['n_train'] for r in cv_results])),
        'n_test': int(np.mean([r['n_test'] for r in cv_results])),
        'n_pos_train': int(np.mean([r['n_pos_train'] for r in cv_results])),
        'n_pos_test': int(np.mean([r['n_pos_test'] for r in cv_results])),
        'auc': float(np.mean([r['auc'] for r in cv_results])),
        'f1': float(np.mean([r['f1'] for r in cv_results])),
        'precision': float(np.mean([r['precision'] for r in cv_results])),
        'recall': float(np.mean([r['recall'] for r in cv_results])),
        'tss': float(np.mean([r['tss'] for r in cv_results])),
        'hss': float(np.mean([r['hss'] for r in cv_results])),
        'pod': float(np.mean([r['pod'] for r in cv_results])),
        'pofd': float(np.mean([r['pofd'] for r in cv_results])),
        'tp': int(np.mean([r['tp'] for r in cv_results])),
        'fp': int(np.mean([r['fp'] for r in cv_results])),
        'tn': int(np.mean([r['tn'] for r in cv_results])),
        'fn': int(np.mean([r['fn'] for r in cv_results])),
        'scale_pos_weight': float(np.mean([r['scale_pos_weight'] for r in cv_results])),
        'std_tss': float(np.std([r['tss'] for r in cv_results])),
        'std_auc': float(np.std([r['auc'] for r in cv_results])),
    }
    all_results.append(cv_avg)
    
    print(f'  Average TSS: {cv_avg["tss"]:.3f} ± {cv_avg["std_tss"]:.3f}')
    print(f'  Average AUC: {cv_avg["auc"]:.3f} ± {cv_avg["std_auc"]:.3f}')
    print(f'  Average POD: {cv_avg["pod"]:.3f}')
    
    # Save all results
    output = {
        'generated': datetime.now().isoformat(),
        'feature_columns': GOES_FEATURES,
        'n_total_samples': len(df),
        'date_range': [str(df.index[0]), str(df.index[-1])],
        'splits': all_results,
    }
    
    out_path = os.path.join(OUTPUT_DIR, 'cross_cycle_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved results: {out_path}')
    
    # Save best model
    best_idx = np.argmax([r['tss'] for r in all_results])
    best_model_name = all_results[best_idx]['label']
    print(f'\nBest model: {best_model_name} (TSS={all_results[best_idx]["tss"]:.3f})')
    
    # Summary table
    print(f'\n{"=" * 60}')
    print('SUMMARY TABLE')
    print(f'{"=" * 60}')
    print(f'{"Split":<35} {"TSS":>6} {"AUC":>6} {"POD":>6} {"F1":>6}')
    print('-' * 60)
    for r in all_results:
        print(f'{r["label"]:<35} {r["tss"]:>6.3f} {r["auc"]:>6.3f} {r["pod"]:>6.3f} {r["f1"]:>6.3f}')


if __name__ == '__main__':
    main()
