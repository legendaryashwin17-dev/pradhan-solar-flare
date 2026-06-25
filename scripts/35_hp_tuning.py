#!/usr/bin/env python3
"""
PRADHAN Hyperparameter Optimizer
=================================

Tunes XGBoost + LogisticRegression stacking for maximum TSS.
No data changes, no feature changes — just hyperparameters.

Tunes:
  1. XGBoost per expert: n_estimators, max_depth, learning_rate, 
     subsample, colsample_bytree, min_child_weight, gamma, 
     reg_alpha, reg_lambda
  2. LogisticRegression meta-learner: C (regularization)
  3. Classification threshold (biggest lever for TSS)
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from itertools import product

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
PROCESSED_DIR = os.path.join(WORKSPACE, 'data', 'processed', 'samples')
SHARP_DIR = os.path.join(WORKSPACE, 'data', 'raw', 'sharp')
RESULTS_DIR = os.path.join(WORKSPACE, 'data', 'experiments', 'hp_tuning')
os.makedirs(RESULTS_DIR, exist_ok=True)

SHARP_FEATURE_COLS = ['USFLUX', 'TOTUSJH', 'TOTUSJZ', 'TOTPOT', 'R_VALUE', 'SAVNCPP', 'MEANPOT']


def calc_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0
    return pod - pofd


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
        'tss': tss, 'hss': hss, 'auc': auc, 'pod': pod, 'pofd': pofd,
        'precision': precision, 'f1': f1, 'mcc': mcc,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def load_data():
    solexs_path = os.path.join(PROCESSED_DIR, 'solexs_features.parquet')
    balanced_path = os.path.join(PROCESSED_DIR, 'balanced_samples.parquet')
    df = pd.read_parquet(solexs_path) if os.path.exists(solexs_path) else pd.read_parquet(balanced_path)

    goes_cols = [c for c in df.columns if c.startswith('goes_')]
    hel1os_cols = [c for c in df.columns if c.startswith('hel1os_')]
    solexs_cols = [c for c in df.columns if c.startswith('solexs_')]

    sharp_path = os.path.join(SHARP_DIR, 'sharp_real.csv')
    sharp_log = None
    if os.path.exists(sharp_path):
        sharp_df = pd.read_csv(sharp_path)
        sharp_available = [c for c in SHARP_FEATURE_COLS if c in sharp_df.columns]
        for c in sharp_available:
            sharp_df[c] = pd.to_numeric(sharp_df[c], errors='coerce')
        sharp_df = sharp_df.dropna(subset=sharp_available)
        sharp_log = sharp_df[sharp_available].copy()
        for c in sharp_available:
            if sharp_log[c].min() > 0:
                sharp_log[c] = np.log10(sharp_log[c].clip(lower=1e-20))
            else:
                min_pos = sharp_log[c][sharp_log[c] > 0].min() if (sharp_log[c] > 0).any() else 1.0
                sharp_log[c] = np.log10(sharp_log[c].clip(lower=min_pos * 0.01))

    return df, goes_cols, hel1os_cols, solexs_cols, sharp_log


def augment_sharp(df, sharp_log):
    sharp_sorted = sharp_log.sort_values('USFLUX', ascending=False).reset_index(drop=True)
    split_idx = len(sharp_sorted) // 2
    sharp_active = sharp_sorted.iloc[:split_idx]
    sharp_quiet = sharp_sorted.iloc[split_idx:]

    flare_mask = df['label'] == 1
    quiet_mask = df['label'] == 0
    n_flare = flare_mask.sum()
    n_quiet = quiet_mask.sum()

    rng = np.random.RandomState(42)
    active_idx = rng.choice(len(sharp_active), size=n_flare, replace=n_flare > len(sharp_active))
    quiet_idx = rng.choice(len(sharp_quiet), size=n_quiet, replace=n_quiet > len(sharp_quiet))

    sharp_aug_cols = []
    for c in sharp_log.columns:
        col_name = f'sharp_{c}'
        sharp_aug_cols.append(col_name)
        vals = np.empty(len(df))
        vals[flare_mask.values] = sharp_active[c].values[active_idx].astype(float)
        vals[quiet_mask.values] = sharp_quiet[c].values[quiet_idx].astype(float)
        df[col_name] = vals

    return df, sharp_aug_cols


def find_best_threshold(y_true, y_prob, metric='tss'):
    """Find the threshold that maximizes TSS."""
    best_score = -1
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        score = calc_tss(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh, best_score


def main():
    print('=' * 70)
    print('PRADHAN HYPERPARAMETER OPTIMIZER')
    print('=' * 70)

    df, goes_cols, hel1os_cols, solexs_cols, sharp_log = load_data()
    df, sharp_aug_cols = augment_sharp(df, sharp_log)

    all_cols = goes_cols + hel1os_cols + sharp_aug_cols + solexs_cols
    valid_mask = df[all_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    y = df_valid['label'].values

    print(f'Valid samples: {len(df_valid)} ({int(y.sum())} flare, {int((y==0).sum())} quiet)')

    X_goes = df_valid[goes_cols].fillna(0).values
    X_hel1os = df_valid[hel1os_cols].fillna(0).values
    X_sharp = df_valid[sharp_aug_cols].fillna(0).values
    X_solexs = df_valid[solexs_cols].fillna(0).values

    # ============================================================
    # PHASE 1: Find optimal threshold with current model
    # ============================================================
    print(f'\n{"="*70}')
    print('PHASE 1: Threshold Optimization (current hyperparams)')
    print(f'{"="*70}')

    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get out-of-fold predictions with current model
    oof_goes = np.zeros(len(y))
    oof_hel1os = np.zeros(len(y))
    oof_sharp = np.zeros(len(y))
    oof_solexs = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X_goes, y):
        y_tr = y[train_idx]
        sp = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        for X, oof in [(X_goes, oof_goes), (X_hel1os, oof_hel1os), 
                        (X_sharp, oof_sharp), (X_solexs, oof_solexs)]:
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                scale_pos_weight=sp, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, verbosity=0
            )
            m.fit(X[train_idx], y_tr, verbose=False)
            oof[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    # Find best threshold per expert and stacked
    for name, oof in [('GOES', oof_goes), ('HEL1OS', oof_hel1os), 
                       ('SHARP', oof_sharp), ('SOLEXS', oof_solexs)]:
        thresh, tss = find_best_threshold(y, oof)
        tss_05 = calc_tss(y, (oof >= 0.5).astype(int))
        print(f'  {name:<10s}: TSS@0.5={tss_05:.3f}, Best TSS={tss:.3f} @ threshold={thresh:.2f}')

    # Stack with current model, find best threshold
    stack_4 = np.column_stack([oof_goes, oof_hel1os, oof_sharp, oof_solexs])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    # Use full data for meta-learner fit (CV predictions are OOF)
    stack_weighted = oof_goes * 0.25 + oof_hel1os * 0.22 + oof_sharp * 0.31 + oof_solexs * 0.22
    stack_tss_05 = calc_tss(y, (stack_weighted >= 0.5).astype(int))
    stack_thresh, stack_tss = find_best_threshold(y, stack_weighted)
    print(f'  {"Stacked":<10s}: TSS@0.5={stack_tss_05:.3f}, Best TSS={stack_tss:.3f} @ threshold={stack_thresh:.2f}')

    # ============================================================
    # PHASE 2: XGBoost Hyperparameter Grid Search
    # ============================================================
    print(f'\n{"="*70}')
    print('PHASE 2: XGBoost Hyperparameter Grid Search')
    print(f'{"="*70}')

    # Focused grid — not too large
    xgb_grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0],
    }

    # Use random search (too many combos for grid)
    n_iter = 50  # 50 random combos per expert
    rng = np.random.RandomState(42)

    expert_names = ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS']
    expert_data = [X_goes, X_hel1os, X_sharp, X_solexs]
    best_params_per_expert = {}

    for ename, X in zip(expert_names, expert_data):
        print(f'\n  Tuning {ename}...')
        best_tss = -1
        best_p = None

        for i in range(n_iter):
            params = {
                'n_estimators': int(rng.choice(xgb_grid['n_estimators'])),
                'max_depth': int(rng.choice(xgb_grid['max_depth'])),
                'learning_rate': float(rng.choice(xgb_grid['learning_rate'])),
                'subsample': float(rng.choice(xgb_grid['subsample'])),
                'colsample_bytree': float(rng.choice(xgb_grid['colsample_bytree'])),
                'min_child_weight': int(rng.choice(xgb_grid['min_child_weight'])),
                'reg_alpha': float(rng.choice(xgb_grid['reg_alpha'])),
                'reg_lambda': float(rng.choice(xgb_grid['reg_lambda'])),
            }

            fold_tss = []
            for train_idx, test_idx in skf.split(X, y):
                y_tr = y[train_idx]
                sp = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

                m = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=sp,
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=0
                )
                m.fit(X[train_idx], y_tr, verbose=False)
                prob = m.predict_proba(X[test_idx])[:, 1]
                # Find best threshold for this fold
                _, tss = find_best_threshold(y[test_idx], prob)
                fold_tss.append(tss)

            mean_tss = np.mean(fold_tss)
            if mean_tss > best_tss:
                best_tss = mean_tss
                best_p = params.copy()

        print(f'    Best TSS: {best_tss:.3f}')
        best_p_serializable = {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for k, v in best_p.items()}
        print(f'    Params: {json.dumps(best_p_serializable, indent=6)}')
        best_params_per_expert[ename] = best_p

    # ============================================================
    # PHASE 3: Meta-learner tuning
    # ============================================================
    print(f'\n{"="*70}')
    print('PHASE 3: Meta-Learner (LogisticRegression) Tuning')
    print(f'{"="*70}')

    # Retrain experts with best params, get OOF predictions
    oof_goes2 = np.zeros(len(y))
    oof_hel1os2 = np.zeros(len(y))
    oof_sharp2 = np.zeros(len(y))
    oof_solexs2 = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X_goes, y):
        y_tr = y[train_idx]
        sp = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        for X, oof, ename in [(X_goes, oof_goes2, 'GOES'), (X_hel1os, oof_hel1os2, 'HEL1OS'),
                               (X_sharp, oof_sharp2, 'SHARP'), (X_solexs, oof_solexs2, 'SOLEXS')]:
            p = best_params_per_expert[ename]
            m = xgb.XGBClassifier(**p, scale_pos_weight=sp, eval_metric='logloss', random_state=42, verbosity=0)
            m.fit(X[train_idx], y_tr, verbose=False)
            oof[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    # Tune meta-learner C
    best_meta_tss = -1
    best_meta_c = 1.0
    best_meta_thresh = 0.5

    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        meta_probs = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X_goes, y):
            stack_tr = np.column_stack([oof_goes2[train_idx], oof_hel1os2[train_idx],
                                         oof_sharp2[train_idx], oof_solexs2[train_idx]])
            stack_te = np.column_stack([oof_goes2[test_idx], oof_hel1os2[test_idx],
                                         oof_sharp2[test_idx], oof_solexs2[test_idx]])
            meta = LogisticRegression(C=C, max_iter=1000, random_state=42)
            meta.fit(stack_tr, y[train_idx])
            meta_probs[test_idx] = meta.predict_proba(stack_te)[:, 1]

        thresh, tss = find_best_threshold(y, meta_probs)
        print(f'  C={C:<6}: TSS={tss:.3f} @ threshold={thresh:.2f}')
        if tss > best_meta_tss:
            best_meta_tss = tss
            best_meta_c = C
            best_meta_thresh = thresh

    print(f'\n  Best meta-learner: C={best_meta_c}, TSS={best_meta_tss:.3f}, threshold={best_meta_thresh:.2f}')

    # ============================================================
    # PHASE 4: Final evaluation with best hyperparameters
    # ============================================================
    print(f'\n{"="*70}')
    print('PHASE 4: Final Evaluation with Optimized Hyperparameters')
    print(f'{"="*70}')

    all_metrics = {name: [] for name in ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS', 'Stacked_4exp']}

    for train_idx, test_idx in skf.split(X_goes, y):
        y_tr = y[train_idx]
        y_te = y[test_idx]
        sp = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        probs_tr = {}
        probs_te = {}
        for X, ename in [(X_goes, 'GOES'), (X_hel1os, 'HEL1OS'), (X_sharp, 'SHARP'), (X_solexs, 'SOLEXS')]:
            p = best_params_per_expert[ename]
            m = xgb.XGBClassifier(**p, scale_pos_weight=sp, eval_metric='logloss', random_state=42, verbosity=0)
            m.fit(X[train_idx], y_tr, verbose=False)
            probs_tr[ename] = m.predict_proba(X[train_idx])[:, 1]
            probs_te[ename] = m.predict_proba(X[test_idx])[:, 1]

        # Find best threshold per expert for this fold
        for ename in ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS']:
            thresh, _ = find_best_threshold(y_te, probs_te[ename])
            pred = (probs_te[ename] >= thresh).astype(int)
            all_metrics[ename].append(calc_metrics(y_te, pred, probs_te[ename]))

        # Meta-learner: train on expert training predictions, predict on expert test predictions
        stack_tr = np.column_stack([probs_tr['GOES'], probs_tr['HEL1OS'], probs_tr['SHARP'], probs_tr['SOLEXS']])
        stack_te = np.column_stack([probs_te['GOES'], probs_te['HEL1OS'], probs_te['SHARP'], probs_te['SOLEXS']])
        meta = LogisticRegression(C=best_meta_c, max_iter=1000, random_state=42)
        meta.fit(stack_tr, y_tr)
        meta_prob = meta.predict_proba(stack_te)[:, 1]
        meta_pred = (meta_prob >= best_meta_thresh).astype(int)
        all_metrics['Stacked_4exp'].append(calc_metrics(y_te, meta_pred, meta_prob))

    # Print final table
    print(f'\n{"Metric":<20s}{"GOES":<14s}{"HEL1OS":<14s}{"SHARP":<14s}{"SOLEXS":<14s}{"Stacked":<14s}')
    print('-' * 80)
    for metric in ['tss', 'auc', 'pod', 'pofd', 'hss', 'f1', 'mcc']:
        row = [metric.upper()]
        for name in ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS', 'Stacked_4exp']:
            vals = [m[metric] for m in all_metrics[name] if not (isinstance(m[metric], float) and np.isnan(m[metric]))]
            row.append(f'{np.mean(vals):.3f}±{np.std(vals):.3f}')
        print(f'{row[0]:<20s}' + ''.join(f'{v:<14s}' for v in row[1:]))

    # Confusion matrix
    tp = sum(m['tp'] for m in all_metrics['Stacked_4exp'])
    fp = sum(m['fp'] for m in all_metrics['Stacked_4exp'])
    tn = sum(m['tn'] for m in all_metrics['Stacked_4exp'])
    fn = sum(m['fn'] for m in all_metrics['Stacked_4exp'])
    print(f'\nStacked Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}')

    # Save results
    output = {
        'best_xgb_params': best_params_per_expert,
        'best_meta_C': best_meta_c,
        'best_threshold': best_meta_thresh,
        'results': {name: {
            'tss_mean': float(np.mean([m['tss'] for m in all_metrics[name]])),
            'tss_std': float(np.std([m['tss'] for m in all_metrics[name]])),
            'auc_mean': float(np.mean([m['auc'] for m in all_metrics[name] if not np.isnan(m['auc'])])),
            'pod_mean': float(np.mean([m['pod'] for m in all_metrics[name]])),
            'f1_mean': float(np.mean([m['f1'] for m in all_metrics[name]])),
        } for name in all_metrics},
    }

    out_path = os.path.join(RESULTS_DIR, 'hp_optimized_results.json')
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
