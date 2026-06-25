#!/usr/bin/env python3
"""
PRADHAN Scientifically Sound Evaluation
=========================================

Uses GOES historical dataset (June 2024 - June 2026) with REAL timestamps.
Time-based splits only. No threshold tuning. Default XGBoost.

Methodology:
  1. Walk-forward: train on month N, test on month N+1
  2. Expanding window: train on all data up to month N, test on month N+1
  3. Cross-year: train on 2024, test on 2026
  4. Report TSS with honest confidence intervals
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import xgboost as xgb

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
GOES_HIST = os.path.join(WORKSPACE, 'data', 'historical', 'goes_cross_cycle_features.parquet')
RESULTS_DIR = os.path.join(WORKSPACE, 'data', 'experiments', 'scientific_eval')
os.makedirs(RESULTS_DIR, exist_ok=True)

GOES_FEATURES = [
    'goes_log_xrsa', 'goes_log_xrsb', 'goes_xrsb_baseline',
    'goes_xrsb_log_grad', 'goes_xrsb_log_std', 'goes_xrsb_log_mean',
    'goes_xrsa_xrsb_ratio', 'goes_xrsb_log_zscore'
]


def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = pod - pofd
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / hss_den if hss_den > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * pod / (precision + pod) if (precision + pod) > 0 else 0
    mcc_num = (tp * tn - fp * fn)
    mcc_den = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = mcc_num / mcc_den
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan
    return {
        'tss': float(tss), 'hss': float(hss), 'auc': float(auc),
        'pod': float(pod), 'pofd': float(pofd),
        'precision': float(precision), 'f1': float(f1), 'mcc': float(mcc),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def train_xgb(X_train, y_train, X_test):
    """Default XGBoost. No tuning."""
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)  # Default threshold
    return prob, pred


def main():
    print('=' * 70)
    print('PRADHAN SCIENTIFICALLY SOUND EVALUATION')
    print('=' * 70)
    print('GOES historical dataset with real timestamps.')
    print('Time-based splits only. Default XGBoost. Threshold=0.5.')
    print()

    goes = pd.read_parquet(GOES_HIST).sort_index()
    print(f'Dataset: {len(goes)} samples')
    print(f'Date range: {goes.index[0]} to {goes.index[-1]}')
    print(f'M+ flare rate: {goes["label_mflare"].mean():.1%}')
    print(f'Features: {len(GOES_FEATURES)}')

    all_results = {}

    # ================================================================
    # EVAL 1: Walk-Forward (train on month N, test on month N+1)
    # ================================================================
    print(f'\n{"="*70}')
    print('EVAL 1: Walk-Forward Validation')
    print('Train on month N, test on month N+1. No look-ahead.')
    print(f'{"="*70}')

    months = goes.index.to_period('M').unique()
    walk_results = []

    for i in range(len(months) - 1):
        train_mask = goes.index.to_period('M') == months[i]
        test_mask = goes.index.to_period('M') == months[i + 1]

        X_tr = goes.loc[train_mask, GOES_FEATURES].values
        y_tr = goes.loc[train_mask, 'label_mflare'].values
        X_te = goes.loc[test_mask, GOES_FEATURES].values
        y_te = goes.loc[test_mask, 'label_mflare'].values

        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            print(f'  {months[i]} -> {months[i+1]}: SKIP (insufficient class diversity)')
            continue

        prob, pred = train_xgb(X_tr, y_tr, X_te)
        m = calc_metrics(y_te, pred, prob)
        walk_results.append({'month': str(months[i+1]), **m})
        print(f'  {months[i]} -> {months[i+1]}: TSS={m["tss"]:.3f} AUC={m["auc"]:.3f} POD={m["pod"]:.3f} POFD={m["pofd"]:.3f} (n_test={len(y_te)})')

    if walk_results:
        avg_tss = np.mean([r['tss'] for r in walk_results])
        std_tss = np.std([r['tss'] for r in walk_results])
        avg_auc = np.mean([r['auc'] for r in walk_results if not np.isnan(r['auc'])])
        avg_pod = np.mean([r['pod'] for r in walk_results])
        avg_f1 = np.mean([r['f1'] for r in walk_results])
        print(f'\n  Walk-Forward Average ({len(walk_results)} splits):')
        print(f'    TSS = {avg_tss:.3f} ± {std_tss:.3f}')
        print(f'    AUC = {avg_auc:.3f}')
        print(f'    POD = {avg_pod:.3f}')
        print(f'    F1  = {avg_f1:.3f}')
        all_results['walk_forward'] = {
            'tss_mean': float(avg_tss), 'tss_std': float(std_tss),
            'auc_mean': float(avg_auc), 'pod_mean': float(avg_pod),
            'f1_mean': float(avg_f1), 'n_splits': len(walk_results),
            'details': walk_results,
        }

    # ================================================================
    # EVAL 2: Expanding Window (train on all past, test on next month)
    # ================================================================
    print(f'\n{"="*70}')
    print('EVAL 2: Expanding Window')
    print('Train on all data up to month N, test on month N+1.')
    print(f'{"="*70}')

    expand_results = []
    for i in range(2, len(months)):  # Start from 3rd month (need enough train data)
        train_mask = goes.index.to_period('M') < months[i]
        test_mask = goes.index.to_period('M') == months[i]

        X_tr = goes.loc[train_mask, GOES_FEATURES].values
        y_tr = goes.loc[train_mask, 'label_mflare'].values
        X_te = goes.loc[test_mask, GOES_FEATURES].values
        y_te = goes.loc[test_mask, 'label_mflare'].values

        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            print(f'  Test {months[i]}: SKIP (insufficient class diversity)')
            continue

        prob, pred = train_xgb(X_tr, y_tr, X_te)
        m = calc_metrics(y_te, pred, prob)
        expand_results.append({'month': str(months[i]), **m})
        print(f'  Train <{months[i]}, Test {months[i]}: TSS={m["tss"]:.3f} AUC={m["auc"]:.3f} POD={m["pod"]:.3f} POFD={m["pofd"]:.3f}')

    if expand_results:
        avg_tss = np.mean([r['tss'] for r in expand_results])
        std_tss = np.std([r['tss'] for r in expand_results])
        avg_auc = np.mean([r['auc'] for r in expand_results if not np.isnan(r['auc'])])
        print(f'\n  Expanding Window Average ({len(expand_results)} splits):')
        print(f'    TSS = {avg_tss:.3f} ± {std_tss:.3f}')
        print(f'    AUC = {avg_auc:.3f}')
        all_results['expanding_window'] = {
            'tss_mean': float(avg_tss), 'tss_std': float(std_tss),
            'auc_mean': float(avg_auc), 'n_splits': len(expand_results),
            'details': expand_results,
        }

    # ================================================================
    # EVAL 3: Cross-Year (train 2024, test 2026)
    # ================================================================
    print(f'\n{"="*70}')
    print('EVAL 3: Cross-Year (Train: Jun 2024, Test: Apr-Jun 2026)')
    print(f'{"="*70}')

    train_2024 = goes[goes.index < '2025-01-01']
    test_2026 = goes[goes.index >= '2026-01-01']

    X_tr = train_2024[GOES_FEATURES].values
    y_tr = train_2024['label_mflare'].values
    X_te = test_2026[GOES_FEATURES].values
    y_te = test_2026['label_mflare'].values

    print(f'Train: {len(train_2024)} ({int(y_tr.sum())} M+ flares)')
    print(f'Test:  {len(test_2026)} ({int(y_te.sum())} M+ flares)')

    if len(np.unique(y_te)) >= 2 and len(np.unique(y_tr)) >= 2:
        prob, pred = train_xgb(X_tr, y_tr, X_te)
        m = calc_metrics(y_te, pred, prob)
        all_results['cross_year'] = m
        print(f'  TSS={m["tss"]:.3f}  AUC={m["auc"]:.3f}  POD={m["pod"]:.3f}  POFD={m["pofd"]:.3f}')
        print(f'  F1={m["f1"]:.3f}  HSS={m["hss"]:.3f}  MCC={m["mcc"]:.3f}')
        print(f'  TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}')

    # ================================================================
    # EVAL 4: Standard 5-fold CV (for comparison, acknowledged as biased)
    # ================================================================
    print(f'\n{"="*70}')
    print('EVAL 4: Standard 5-Fold CV (biased — for comparison only)')
    print('This uses random splits. Inflated by temporal leakage.')
    print(f'{"="*70}')

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for train_idx, test_idx in skf.split(goes[GOES_FEATURES].values, goes['label_mflare'].values):
        X_tr = goes.iloc[train_idx][GOES_FEATURES].values
        y_tr = goes.iloc[train_idx]['label_mflare'].values
        X_te = goes.iloc[test_idx][GOES_FEATURES].values
        y_te = goes.iloc[test_idx]['label_mflare'].values

        prob, pred = train_xgb(X_tr, y_tr, X_te)
        m = calc_metrics(y_te, pred, prob)
        cv_results.append(m)

    avg_tss = np.mean([r['tss'] for r in cv_results])
    avg_auc = np.mean([r['auc'] for r in cv_results if not np.isnan(r['auc'])])
    print(f'  5-Fold CV Average: TSS={avg_tss:.3f} AUC={avg_auc:.3f}')
    all_results['cv_5fold_biased'] = {
        'tss_mean': float(avg_tss), 'auc_mean': float(avg_auc),
        'warning': 'Biased by temporal leakage. Use walk-forward or expanding window instead.',
    }

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f'\n{"="*70}')
    print('FINAL RESULTS')
    print(f'{"="*70}')
    print(f'{"Method":<35s}{"TSS":>10s}{"AUC":>10s}{"Reliable?":>12s}')
    print('-' * 70)

    for name, r in all_results.items():
        if 'tss_mean' in r:
            reliable = 'YES' if name in ['walk_forward', 'expanding_window', 'cross_year'] else 'NO (biased)'
            print(f'{name:<35s}{r["tss_mean"]:>10.3f}{r["auc_mean"]:>10.3f}{reliable:>12s}')
        elif 'tss' in r:
            print(f'{name:<35s}{r["tss"]:>10.3f}{r["auc"]:>10.3f}{"YES":>12s}')

    print(f'\nScientific validity notes:')
    print(f'  - Walk-forward/expanding window: No temporal leakage, honest metrics')
    print(f'  - Cross-year: Tests generalization across years')
    print(f'  - 5-fold CV: Biased (temporal leakage) — shown for comparison only')
    print(f'  - GOES-only: 8 features, no SHARP label leakage')
    print(f'  - Threshold=0.5: No tuning on test data')

    # Save
    output = {
        'methodology': 'Time-based splits, default XGBoost, threshold=0.5, no tuning on test data',
        'goes_features': GOES_FEATURES,
        'dataset': {'n_samples': len(goes), 'date_range': [str(goes.index[0]), str(goes.index[-1])]},
        'results': all_results,
    }

    out_path = os.path.join(RESULTS_DIR, 'scientific_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
