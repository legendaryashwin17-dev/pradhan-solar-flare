#!/usr/bin/env python3
"""
PRADHAN Multi-Input Pipeline v2 — Train on Balanced Data

Uses balanced_samples.parquet (50/50 flare/quiet) instead of
skewed hel1os_goes_samples.parquet (87/13).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\processed\samples'
RESULTS_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0ba1c8744ba1a972\workspace\data\experiments\exp1_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    balanced_acc = (tpr + specificity) / 2
    mcc_num = (tp * tn - fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {
        'tss': tss, 'hss': hss, 'auc': auc,
        'pod': tpr, 'pofd': fpr,
        'precision': precision, 'recall': tpr, 'f1': f1,
        'accuracy': accuracy, 'specificity': specificity, 'npv': npv,
        'balanced_accuracy': balanced_acc, 'mcc': mcc,
        'fpr': fpr, 'fnr': 1 - tpr,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def select_features(df, feature_cols, max_features=10):
    print(f'\n{"="*60}\nFEATURE SELECTION\n{"="*60}')
    variances = df[feature_cols].var()
    low_var = variances[variances < 1e-10].index.tolist()
    if low_var:
        print(f'Drop low-var: {low_var}')
        feature_cols = [f for f in feature_cols if f not in low_var]

    corrs = {}
    for f in feature_cols:
        try:
            corrs[f] = abs(df[f].fillna(0).corr(df['label']))
        except Exception:
            corrs[f] = 0

    corr_matrix = df[feature_cols].corr().abs()
    to_drop = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if corr_matrix.iloc[i, j] > 0.95:
                f1, f2 = feature_cols[i], feature_cols[j]
                if corrs.get(f1, 0) < corrs.get(f2, 0):
                    to_drop.add(f1)
                else:
                    to_drop.add(f2)
    if to_drop:
        print(f'Drop multicollinear: {to_drop}')
        feature_cols = [f for f in feature_cols if f not in to_drop]

    if len(feature_cols) > max_features:
        ranked = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
        feature_cols = [f for f, _ in ranked[:max_features]]

    print(f'Final features ({len(feature_cols)}): {feature_cols}')
    return feature_cols


def stratified_cv(X, y, n_splits=5, n_repeats=10):
    """Repeated stratified K-fold (works with small samples and balanced classes)."""
    print(f'\n{"="*60}\nSTRATIFIED {n_splits}-FOLD CV (x{n_repeats} repeats)\n{"="*60}')
    all_metrics = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, verbosity=0
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            m = calc_metrics(y_te, y_pred, y_prob)
            all_metrics.append(m)
    agg = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics if not (isinstance(m[k], float) and np.isnan(m[k]))]
        agg[k] = (np.mean(vals), np.std(vals))
    print(f'\nAggregated results ({len(all_metrics)} folds):')
    for k in ['tss', 'hss', 'auc', 'pod', 'precision', 'f1', 'mcc', 'accuracy']:
        mean, std = agg[k]
        print(f'  {k.upper():>20s}: {mean:.4f} ± {std:.4f}')
    return agg, all_metrics


def bootstrap_ci(X, y, n_bootstrap=1000):
    print(f'\n{"="*60}\nBOOTSTRAP ({n_bootstrap} iters)\n{"="*60}')
    tss_boot = []
    auc_boot = []
    for i in range(n_bootstrap):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        if len(np.unique(y_b)) < 2:
            continue
        split = int(0.7 * len(X_b))
        X_tr, X_te = X_b.iloc[:split], X_b.iloc[split:]
        y_tr, y_te = y_b.iloc[:split], y_b.iloc[split:]
        if len(np.unique(y_te)) < 2:
            continue
        scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=i, verbosity=0
        )
        model.fit(X_tr, y_tr, verbose=False)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        m = calc_metrics(y_te, y_pred, y_prob)
        tss_boot.append(m['tss'])
        auc_boot.append(m['auc'])
    tss_ci = np.percentile(tss_boot, [2.5, 50, 97.5])
    auc_ci = np.percentile(auc_boot, [2.5, 50, 97.5])
    print(f'TSS: {np.mean(tss_boot):.4f} ± {np.std(tss_boot):.4f}')
    print(f'TSS 95% CI: [{tss_ci[0]:.4f}, {tss_ci[2]:.4f}], median {tss_ci[1]:.4f}')
    print(f'AUC: {np.mean(auc_boot):.4f} ± {np.std(auc_boot):.4f}')
    print(f'AUC 95% CI: [{auc_ci[0]:.4f}, {auc_ci[2]:.4f}]')
    return {
        'tss_mean': float(np.mean(tss_boot)), 'tss_std': float(np.std(tss_boot)),
        'tss_ci_95': [float(tss_ci[0]), float(tss_ci[2])], 'tss_median': float(tss_ci[1]),
        'auc_mean': float(np.mean(auc_boot)), 'auc_std': float(np.std(auc_boot)),
        'auc_ci_95': [float(auc_ci[0]), float(auc_ci[2])],
    }


def shap_analysis(model, X, feature_names, output_dir):
    print(f'\n{"="*60}\nSHAP ANALYSIS\n{"="*60}')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({'feature': feature_names, 'mean_shap': mean_shap}).sort_values('mean_shap', ascending=False)
    print('Feature importance:')
    for _, row in importance.iterrows():
        bar = '#' * int(row['mean_shap'] / importance['mean_shap'].max() * 30)
        print(f'  {row["feature"]:<25s} {row["mean_shap"]:.6f} {bar}')

    goes_feats = importance[importance['feature'].str.startswith('goes_')]
    hel1os_feats = importance[importance['feature'].str.startswith('hel1os_')]
    goes_total = goes_feats['mean_shap'].sum()
    hel1os_total = hel1os_feats['mean_shap'].sum()
    total = goes_total + hel1os_total
    print(f'\nGOES contribution: {goes_total:.4f} ({goes_total/total*100:.1f}%)')
    print(f'HEL1OS contribution: {hel1os_total:.4f} ({hel1os_total/total*100:.1f}%)')
    if hel1os_total > goes_total:
        print(f'HEL1OS dominates over GOES by {hel1os_total/goes_total:.1f}x')
    else:
        print(f'GOES dominates over HEL1OS by {goes_total/hel1os_total:.1f}x')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#ff6b6b' if f.startswith('goes_') else '#4ecdc4' for f in importance['feature']]
    axes[0].barh(importance['feature'], importance['mean_shap'], color=colors)
    axes[0].set_xlabel('Mean |SHAP value|')
    axes[0].set_title('Feature Importance')
    axes[0].invert_yaxis()
    axes[1].pie([goes_total, hel1os_total], labels=['GOES', 'HEL1OS'],
                autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'], startangle=90)
    axes[1].set_title('Instrument Contribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return importance


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'balanced_samples.parquet'))
    print(f'Loaded {len(df)} balanced samples')
    print(f'Class balance: {df["label"].value_counts().to_dict()}')

    feature_cols = [c for c in df.columns if c.startswith(('hel1os_', 'goes_'))]
    selected = select_features(df, feature_cols, max_features=10)
    X = df[selected].copy()
    y = df['label'].copy()
    X = X.fillna(X.median())

    cv_results, _ = stratified_cv(X, y, n_splits=5, n_repeats=10)
    boot_results = bootstrap_ci(X, y, n_bootstrap=1000)

    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    final_model.fit(X, y, verbose=False)
    importance = shap_analysis(final_model, X, selected, RESULTS_DIR)

    # Get full metrics from final model on training (for reporting)
    y_pred = final_model.predict(X)
    y_prob = final_model.predict_proba(X)[:, 1]
    full_metrics = calc_metrics(y, y_pred, y_prob)

    import json
    results = {
        'samples': int(len(df)),
        'class_balance': df['label'].value_counts().to_dict(),
        'cv_5x10': {k: {'mean': float(v[0]), 'std': float(v[1])} for k, v in cv_results.items()},
        'bootstrap_1000': boot_results,
        'features': selected,
        'importance': importance.to_dict('records'),
        'final_train_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in full_metrics.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'balanced_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\nSaved: {os.path.join(RESULTS_DIR, "balanced_results.json")}')
    print(f'\nFinal training metrics (in-sample):')
    for k in ['tss', 'auc', 'hss', 'pod', 'precision', 'f1', 'mcc', 'accuracy']:
        print(f'  {k}: {full_metrics[k]:.4f}')


if __name__ == '__main__':
    main()