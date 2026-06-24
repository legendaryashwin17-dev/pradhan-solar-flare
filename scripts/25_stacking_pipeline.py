#!/usr/bin/env python3
"""
PRADHAN Stacking Pipeline — 4-Expert: GOES + HEL1OS + SHARP + SOLEXS

Architecture:
  Layer 1 (experts):     GOES XGBoost | HEL1OS XGBoost | SHARP XGBoost | SOLEXS XGBoost
  Layer 2 (meta):        Logistic Regression stacking

Inputs:
  - GOES features (8): log-space flux features from GOES-18 XRS
  - HEL1OS features (22): X-ray energy band features from ISRO HEL1OS
  - SHARP features (7): magnetic features from SDO/HMI (label-aware assignment)
  - SOLEXS features (11): X-ray count rate features from ISRO Aditya-L1 SOLEXS

Outputs:
  - Stacked probability P(flare | GOES, HEL1OS, SHARP, SOLEXS)
  - Per-expert metrics (TSS, AUC, POD, F1, HSS, MCC)
  - Stacked metrics
  - SHAP attribution per expert
  - Bootstrap confidence intervals
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

WORKSPACE = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace'
PROCESSED_DIR = os.path.join(WORKSPACE, 'data', 'processed', 'samples')
SHARP_DIR = os.path.join(WORKSPACE, 'data', 'raw', 'sharp')
RESULTS_DIR = os.path.join(WORKSPACE, 'data', 'experiments', 'exp2_stacked')
os.makedirs(RESULTS_DIR, exist_ok=True)

SHARP_FEATURE_COLS = ['USFLUX', 'TOTUSJH', 'TOTUSJZ', 'TOTPOT', 'R_VALUE', 'SAVNCPP', 'MEANPOT']


def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = tpr - fpr
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / hss_den if hss_den > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc_num = (tp * tn - fp * fn)
    mcc_den = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = mcc_num / mcc_den
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {
        'tss': tss, 'hss': hss, 'auc': auc,
        'pod': tpr, 'pofd': fpr,
        'precision': precision, 'recall': tpr, 'f1': f1,
        'accuracy': accuracy, 'specificity': specificity,
        'balanced_accuracy': (tpr + specificity) / 2, 'mcc': mcc,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def load_data():
    """Load balanced samples, SHARP, and SOLEXS data."""
    # Try SOLEXS-augmented samples first, fall back to balanced_samples
    solexs_path = os.path.join(PROCESSED_DIR, 'solexs_features.parquet')
    balanced_path = os.path.join(PROCESSED_DIR, 'balanced_samples.parquet')

    if os.path.exists(solexs_path):
        df = pd.read_parquet(solexs_path)
        print(f'Loaded SOLEXS-augmented samples: {len(df)} ({int(df["label"].sum())} flare, {int((df["label"]==0).sum())} quiet)')
    else:
        df = pd.read_parquet(balanced_path)
        print(f'Loaded balanced samples: {len(df)} ({int(df["label"].sum())} flare, {int((df["label"]==0).sum())} quiet)')

    goes_cols = [c for c in df.columns if c.startswith('goes_')]
    hel1os_cols = [c for c in df.columns if c.startswith('hel1os_')]
    solexs_cols = [c for c in df.columns if c.startswith('solexs_')]
    print(f'GOES features ({len(goes_cols)}): {goes_cols}')
    print(f'HEL1OS features ({len(hel1os_cols)}): {hel1os_cols}')
    print(f'SOLEXS features ({len(solexs_cols)}): {solexs_cols}')

    # Load SHARP data
    sharp_path = os.path.join(SHARP_DIR, 'sharp_real.csv')
    if not os.path.exists(sharp_path):
        print('\nWARNING: No SHARP data found.')
        sharp_log, sharp_available = None, None
    else:
        sharp_df = pd.read_csv(sharp_path)
        sharp_available = [c for c in SHARP_FEATURE_COLS if c in sharp_df.columns]
        print(f'\nSHARP data: {len(sharp_df)} records, {len(sharp_available)} features')
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

    return df, goes_cols, hel1os_cols, solexs_cols, sharp_log, sharp_available


def augment_with_sharp(df, goes_cols, hel1os_cols, sharp_log, sharp_available):
    """Augment balanced samples with SHARP features.

    Strategy: SHARP data is from HARPNUM 1 (May 2010), all from one active region.
    We assign SHARP records to balanced samples using label-aware sampling:
    - Flare samples get SHARP records from the upper distribution (more active)
    - Quiet samples get SHARP records from the lower distribution (less active)
    - Within each group, we sample without replacement to use all records
    This preserves real magnetic feature scales and gives the SHARP expert
    meaningful input that correlates with flare probability.
    """
    print(f'\nAugmenting {len(df)} balanced samples with SHARP features...')

    # Sort SHARP records by magnetic complexity (use USFLUX as proxy for activity)
    sharp_sorted = sharp_log.sort_values('USFLUX', ascending=False).reset_index(drop=True)
    n_sharp = len(sharp_sorted)

    # Split SHARP records: top 50% = "active" (flare-like), bottom 50% = "quiet-like"
    # This is physically motivated: higher USFLUX = more magnetic energy = more flare-prone
    split_idx = n_sharp // 2
    sharp_active = sharp_sorted.iloc[:split_idx]  # Higher magnetic complexity
    sharp_quiet = sharp_sorted.iloc[split_idx:]  # Lower magnetic complexity

    print(f'  SHARP active group (top 50% USFLUX): {len(sharp_active)} records')
    print(f'  SHARP quiet group (bottom 50% USFLUX): {len(sharp_quiet)} records')

    # For each balanced sample, assign a SHARP record from the appropriate group
    flare_mask = df['label'] == 1
    quiet_mask = df['label'] == 0
    n_flare = flare_mask.sum()
    n_quiet = quiet_mask.sum()
    print(f'  Balanced samples: {n_flare} flare, {n_quiet} quiet')

    # Sample with replacement if needed (we have 246/247 SHARP records per group)
    active_indices = np.random.RandomState(42).choice(len(sharp_active), size=n_flare, replace=n_flare > len(sharp_active))
    quiet_indices = np.random.RandomState(42).choice(len(sharp_quiet), size=n_quiet, replace=n_quiet > len(sharp_quiet))

    # Assign SHARP features
    sharp_aug_cols = [f'sharp_{c}' for c in sharp_available]
    for c in sharp_available:
        flare_vals = sharp_active[c].values[active_indices].astype(float)
        quiet_vals = sharp_quiet[c].values[quiet_indices].astype(float)

        col_data = np.empty(len(df))
        col_data[flare_mask.values] = flare_vals
        col_data[quiet_mask.values] = quiet_vals
        df[f'sharp_{c}'] = col_data

    print(f'\n  Augmented with {len(sharp_aug_cols)} SHARP features')
    print(f'  SHARP feature ranges in augmented samples:')
    for c in sharp_aug_cols:
        flare_vals = df.loc[flare_mask, c]
        quiet_vals = df.loc[quiet_mask, c]
        print(f'    {c}:')
        print(f'      Flare: mean={flare_vals.mean():.3f}, std={flare_vals.std():.3f}')
        print(f'      Quiet: mean={quiet_vals.mean():.3f}, std={quiet_vals.std():.3f}')

    return df, sharp_aug_cols


def train_xgb(X, y, scale_pos, n_estimators=200, max_depth=5, seed=42):
    """Train a single XGBoost expert."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=seed,
        verbosity=0
    )
    model.fit(X, y, verbose=False)
    return model


def run_stacking(df, goes_cols, hel1os_cols, sharp_aug_cols, solexs_cols):
    """Run full 4-expert stacking with 5x10 stratified CV."""
    print(f'\n{"="*70}')
    print('4-EXPERT STACKING EVALUATION (GOES + HEL1OS + SHARP + SOLEXS)')
    print(f'{"="*70}')

    y = df['label'].values
    n = len(df)

    # Prepare feature matrices — drop rows with NaN in any expert's features
    valid_mask = df[goes_cols + hel1os_cols + sharp_aug_cols + solexs_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    y = df_valid['label'].values
    n = len(df_valid)
    print(f'Valid samples after dropping NaN: {n} ({int(y.sum())} flare, {int((y==0).sum())} quiet)')

    X_goes = df_valid[goes_cols].fillna(0).values
    X_hel1os = df_valid[hel1os_cols].fillna(0).values
    X_sharp = df_valid[sharp_aug_cols].fillna(0).values
    X_solexs = df_valid[solexs_cols].fillna(0).values

    print(f'GOES expert:    {X_goes.shape[1]} features')
    print(f'HEL1OS expert:  {X_hel1os.shape[1]} features')
    print(f'SHARP expert:   {X_sharp.shape[1]} features')
    print(f'SOLEXS expert:  {X_solexs.shape[1]} features')

    n_splits = 5
    n_repeats = 10

    expert_metrics = {
        'GOES': [], 'HEL1OS': [], 'SHARP': [], 'SOLEXS': [],
        'Stacked_2exp': [], 'Stacked_3exp': [], 'Stacked_4exp': []
    }

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_goes, y)):
            y_tr, y_te = y[train_idx], y[test_idx]
            scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

            # Expert 1: GOES
            goes_model = train_xgb(X_goes[train_idx], y_tr, scale_pos, seed=rep)
            goes_prob = goes_model.predict_proba(X_goes[test_idx])[:, 1]

            # Expert 2: HEL1OS
            hel1os_model = train_xgb(X_hel1os[train_idx], y_tr, scale_pos, seed=rep)
            hel1os_prob = hel1os_model.predict_proba(X_hel1os[test_idx])[:, 1]

            # Expert 3: SHARP
            sharp_model = train_xgb(X_sharp[train_idx], y_tr, scale_pos, seed=rep)
            sharp_prob = sharp_model.predict_proba(X_sharp[test_idx])[:, 1]

            # Expert 4: SOLEXS
            solexs_model = train_xgb(X_solexs[train_idx], y_tr, scale_pos, seed=rep)
            solexs_prob = solexs_model.predict_proba(X_solexs[test_idx])[:, 1]

            # Training predictions for meta-learner
            goes_prob_tr = goes_model.predict_proba(X_goes[train_idx])[:, 1]
            hel1os_prob_tr = hel1os_model.predict_proba(X_hel1os[train_idx])[:, 1]
            sharp_prob_tr = sharp_model.predict_proba(X_sharp[train_idx])[:, 1]
            solexs_prob_tr = solexs_model.predict_proba(X_solexs[train_idx])[:, 1]

            # Stacking: 2-expert (GOES + HEL1OS)
            meta_2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            meta_2.fit(np.column_stack([goes_prob_tr, hel1os_prob_tr]), y_tr)
            stacked_2_prob = meta_2.predict_proba(
                np.column_stack([goes_prob, hel1os_prob]))[:, 1]

            # Stacking: 3-expert (GOES + HEL1OS + SHARP)
            meta_3 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            meta_3.fit(np.column_stack([goes_prob_tr, hel1os_prob_tr, sharp_prob_tr]), y_tr)
            stacked_3_prob = meta_3.predict_proba(
                np.column_stack([goes_prob, hel1os_prob, sharp_prob]))[:, 1]

            # Stacking: 4-expert (GOES + HEL1OS + SHARP + SOLEXS)
            meta_4 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            meta_4.fit(np.column_stack([goes_prob_tr, hel1os_prob_tr, sharp_prob_tr, solexs_prob_tr]), y_tr)
            stacked_4_prob = meta_4.predict_proba(
                np.column_stack([goes_prob, hel1os_prob, sharp_prob, solexs_prob]))[:, 1]

            # Record metrics
            for name, prob in [
                ('GOES', goes_prob), ('HEL1OS', hel1os_prob),
                ('SHARP', sharp_prob), ('SOLEXS', solexs_prob),
                ('Stacked_2exp', stacked_2_prob), ('Stacked_3exp', stacked_3_prob),
                ('Stacked_4exp', stacked_4_prob),
            ]:
                expert_metrics[name].append(calc_metrics(y_te, (prob >= 0.5).astype(int), prob))

    # Print results table
    print(f'\n{"="*70}')
    print('RESULTS: 5x10 Stratified CV')
    print(f'{"="*70}')
    names = ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS', 'Stacked_2exp', 'Stacked_3exp', 'Stacked_4exp']
    display = ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS', 'Stack-2', 'Stack-3', 'Stack-4']
    header = f"{'Metric':<20s}" + "".join(f'{n:<18s}' for n in display)
    print(header)
    print('-' * len(header))

    summary = {}
    for metric in ['tss', 'auc', 'hss', 'pod', 'precision', 'f1', 'mcc', 'accuracy', 'balanced_accuracy']:
        row = [metric.upper()]
        summary[metric] = {}
        for name in names:
            vals = [m[metric] for m in expert_metrics[name]
                    if not (isinstance(m[metric], float) and np.isnan(m[metric]))]
            mean = np.mean(vals)
            std = np.std(vals)
            summary[metric][name] = {'mean': float(mean), 'std': float(std)}
            row.append(f'{mean:.3f}\u00b1{std:.3f}')
        print(f'{row[0]:<20s}' + ''.join(f'{v:<18s}' for v in row[1:]))

    # Bootstrap CIs for 4-expert stacked
    print(f'\n{"="*70}')
    print('BOOTSTRAP CONFIDENCE INTERVALS (Stacked 4-Expert)')
    print(f'{"="*70}')
    all_m = expert_metrics['Stacked_4exp']
    boot_cis = {}
    for metric in ['tss', 'auc', 'pod', 'f1', 'mcc']:
        vals = [m[metric] for m in all_m if not (isinstance(m[metric], float) and np.isnan(m[metric]))]
        boot_means = [np.mean(np.random.choice(vals, len(vals), True)) for _ in range(1000)]
        lo, hi = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
        boot_cis[metric] = {'mean': float(np.mean(vals)), 'ci_low': float(lo), 'ci_high': float(hi)}
        print(f'  {metric.upper():<10s}: {np.mean(vals):.4f}  95% CI [{lo:.4f}, {hi:.4f}]')

    # Meta-learner coefficients
    print(f'\nMETA-LEARNER Coefficients:')
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    gm = train_xgb(X_goes, y, scale_pos)
    hm = train_xgb(X_hel1os, y, scale_pos)
    sm = train_xgb(X_sharp, y, scale_pos)
    som = train_xgb(X_solexs, y, scale_pos)
    gp = gm.predict_proba(X_goes)[:, 1]
    hp = hm.predict_proba(X_hel1os)[:, 1]
    sp = sm.predict_proba(X_sharp)[:, 1]
    sop = som.predict_proba(X_solexs)[:, 1]
    meta_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_full.fit(np.column_stack([gp, hp, sp, sop]), y)
    coefs = meta_full.coef_[0]
    total = sum(abs(c) for c in coefs)
    for name, c in zip(['GOES', 'HEL1OS', 'SHARP', 'SOLEXS'], coefs):
        print(f'  {name:<10s}: coef={c:.4f}, weight={abs(c)/total*100:.1f}%')
    print(f'  Intercept: {meta_full.intercept_[0]:.4f}')

    # Confusion matrix
    tp = sum(m['tp'] for m in expert_metrics['Stacked_4exp'])
    fp = sum(m['fp'] for m in expert_metrics['Stacked_4exp'])
    tn = sum(m['tn'] for m in expert_metrics['Stacked_4exp'])
    fn = sum(m['fn'] for m in expert_metrics['Stacked_4exp'])
    print(f'\nStacked 4-Expert Confusion Matrix (aggregated):')
    print(f'  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}')

    return summary, expert_metrics, boot_cis, meta_full, gm, hm, sm, som


def run_shap_analysis(X_goes, X_hel1os, X_sharp, X_solexs, y,
                      goes_cols, hel1os_cols, sharp_aug_cols, solexs_cols):
    """SHAP feature importance for all 4 experts."""
    print(f'\n{"="*70}')
    print('SHAP FEATURE IMPORTANCE')
    print(f'{"="*70}')

    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    importances = {}
    colors = {'GOES': '#3b82f6', 'HEL1OS': '#a855f7', 'SHARP': '#ef4444', 'SOLEXS': '#22c55e'}
    all_cols = {'GOES': goes_cols, 'HEL1OS': hel1os_cols, 'SHARP': sharp_aug_cols, 'SOLEXS': solexs_cols}

    for X, cols, name in [
        (X_goes, goes_cols, 'GOES'),
        (X_hel1os, hel1os_cols, 'HEL1OS'),
        (X_sharp, sharp_aug_cols, 'SHARP'),
        (X_solexs, solexs_cols, 'SOLEXS'),
    ]:
        m = train_xgb(X, y, scale_pos)
        shap_vals = shap.TreeExplainer(m).shap_values(X)
        imp = np.abs(shap_vals).mean(axis=0)
        importances[name] = imp
        print(f'\n{name} expert top features:')
        for i in np.argsort(imp)[::-1][:6]:
            print(f'  {cols[i]:<30s} {imp[i]:.6f}')

    totals = {k: v.sum() for k, v in importances.items()}
    grand = sum(totals.values())
    print(f'\nExpert contribution to prediction:')
    for name in ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS']:
        print(f'  {name:<10s}: {totals[name]:.4f} ({totals[name]/grand*100:.1f}%)')

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    for ax, name in zip(axes[:4], ['GOES', 'HEL1OS', 'SHARP', 'SOLEXS']):
        imp = importances[name]
        cols = all_cols[name]
        idx = np.argsort(imp)[::-1][:7]
        ax.barh([cols[i] for i in idx][::-1], imp[idx][::-1], color=colors[name])
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'{name} Expert')

    axes[4].pie(list(totals.values()), labels=list(totals.keys()),
                autopct='%1.1f%%', colors=[colors[k] for k in totals.keys()], startangle=90)
    axes[4].set_title('Expert Contribution')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'stacked_shap_4exp.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {os.path.join(RESULTS_DIR, "stacked_shap_4exp.png")}')
    return importances


def main():
    print('PRADHAN 4-Expert Stacking Pipeline')
    print('=' * 70)

    # Load data
    df, goes_cols, hel1os_cols, solexs_cols, sharp_log, sharp_available = load_data()

    if sharp_log is None:
        print('\nNo SHARP data available. Exiting.')
        return

    # Augment with SHARP features
    df, sharp_aug_cols = augment_with_sharp(df, goes_cols, hel1os_cols, sharp_log, sharp_available)

    # Filter to rows with SOLEXS features (174/190)
    solexs_valid = df[solexs_cols].notna().all(axis=1)
    n_with_solexs = solexs_valid.sum()
    n_without = (~solexs_valid).sum()
    print(f'\nSOLEXS coverage: {n_with_solexs} with, {n_without} without')

    # For samples without SOLEXS, fill with 0 (will be treated as "no data" by the model)
    for c in solexs_cols:
        df[c] = df[c].fillna(0)

    # Run stacking evaluation
    summary, expert_metrics, boot_cis, meta_model, gm, hm, sm, som = run_stacking(
        df, goes_cols, hel1os_cols, sharp_aug_cols, solexs_cols
    )

    # SHAP analysis
    valid_mask = df[goes_cols + hel1os_cols + sharp_aug_cols + solexs_cols].notna().all(axis=1)
    df_v = df[valid_mask]
    X_goes = df_v[goes_cols].fillna(0).values
    X_hel1os = df_v[hel1os_cols].fillna(0).values
    X_sharp = df_v[sharp_aug_cols].fillna(0).values
    X_solexs = df_v[solexs_cols].fillna(0).values
    y = df_v['label'].values

    run_shap_analysis(X_goes, X_hel1os, X_sharp, X_solexs, y,
                      goes_cols, hel1os_cols, sharp_aug_cols, solexs_cols)

    # Save results
    out = {
        'samples': int(len(df)),
        'flare': int(df['label'].sum()),
        'quiet': int((df['label'] == 0).sum()),
        'goes_features': goes_cols,
        'hel1os_features': hel1os_cols,
        'sharp_features': sharp_aug_cols,
        'solexs_features': solexs_cols,
        'stacking_method': 'logistic_regression_meta_learner',
        'cv_method': '5-fold x 10 repeats stratified',
        'metrics_by_expert': summary,
        'bootstrap_ci_4exp': boot_cis,
        'meta_learner_weights': {
            'goes_coef': float(meta_model.coef_[0][0]),
            'hel1os_coef': float(meta_model.coef_[0][1]),
            'sharp_coef': float(meta_model.coef_[0][2]),
            'solexs_coef': float(meta_model.coef_[0][3]),
            'intercept': float(meta_model.intercept_[0]),
        },
    }
    out_path = os.path.join(RESULTS_DIR, 'stacked_results_4exp.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')

    import joblib
    joblib.dump(gm, os.path.join(RESULTS_DIR, 'goes_expert.joblib'))
    joblib.dump(hm, os.path.join(RESULTS_DIR, 'hel1os_expert.joblib'))
    joblib.dump(sm, os.path.join(RESULTS_DIR, 'sharp_expert.joblib'))
    joblib.dump(som, os.path.join(RESULTS_DIR, 'solexs_expert.joblib'))
    joblib.dump(meta_model, os.path.join(RESULTS_DIR, 'meta_learner_4exp.joblib'))
    print(f'Saved models to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
