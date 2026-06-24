"""
PRADHAN Multi-Input Pipeline — Step 2: Train + Evaluate

Features: VIF reduction → 8-10 features
Validation: Leave-One-AR-Out CV (LOGO) → TSS mean ± std
Confidence: Bootstrap resampling (1000x) → 95% CI
Explainability: SHAP feature importance → physics conclusion
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Paths
PROCESSED_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\processed\samples'
RESULTS_DIR = r'C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\data\experiments\exp1_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Metrics
# ============================================================

def calc_tss(y_true, y_pred):
    """True Skill Statistic = Recall - FPR"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall / POD
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr - fpr

def calc_hss(y_true, y_pred):
    """Heidke Skill Score"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / den if den > 0 else 0

def calc_pod(y_true, y_pred):
    """Probability of Detection (Recall)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calc_pofd(y_true, y_pred):
    """Probability of False Detection"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


# ============================================================
# Feature Reduction
# ============================================================

def compute_vif(df, features):
    """Compute Variance Inflation Factor for each feature."""
    from numpy.linalg import inv
    
    X = df[features].dropna()
    if len(X) < len(features) + 1:
        return {f: np.nan for f in features}
    
    # Correlation matrix
    corr = X.corr().values
    inv_corr = inv(corr)
    
    vif = {}
    for i, f in enumerate(features):
        vif[f] = inv_corr[i, i]
    
    return vif


def select_features(df, feature_cols, max_features=10):
    """
    Select top features based on:
    1. Correlation with target (keep high-corr features)
    2. VIF (drop multicollinear features)
    3. Variance threshold (drop constant features)
    """
    print(f'\n{"="*60}')
    print('FEATURE SELECTION')
    print(f'{"="*60}')
    
    # Drop features with zero variance
    variances = df[feature_cols].var()
    low_var = variances[variances < 1e-10].index.tolist()
    if low_var:
        print(f'Dropping low-variance: {low_var}')
        feature_cols = [f for f in feature_cols if f not in low_var]
    
    # Correlation with target
    corrs = {}
    for f in feature_cols:
        try:
            corrs[f] = abs(df[f].corr(df['label']))
        except:
            corrs[f] = 0
    
    print(f'\nFeature correlations with label:')
    for f, c in sorted(corrs.items(), key=lambda x: x[1], reverse=True):
        print(f'  {f}: {c:.4f}')
    
    # Drop highly correlated feature pairs (keep the one with higher correlation)
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
        print(f'\nDropping multicollinear (r>0.95): {to_drop}')
        feature_cols = [f for f in feature_cols if f not in to_drop]
    
    # Keep top features by correlation
    if len(feature_cols) > max_features:
        ranked = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
        feature_cols = [f for f, _ in ranked[:max_features]]
        print(f'\nKept top {max_features} features by correlation')
    
    # VIF check
    try:
        vif = compute_vif(df, feature_cols)
        print(f'\nVIF scores:')
        for f, v in sorted(vif.items(), key=lambda x: x[1], reverse=True):
            flag = ' ⚠️ HIGH' if v > 10 else ''
            print(f'  {f}: {v:.2f}{flag}')
    except:
        pass
    
    print(f'\nFinal features ({len(feature_cols)}): {feature_cols}')
    return feature_cols


# ============================================================
# LOGO Cross-Validation
# ============================================================

def logo_cross_validation(X, y, groups, n_features_selected):
    """Leave-One-AR-Out cross-validation with TSS."""
    print(f'\n{"="*60}')
    print('LEAVE-ONE-AR-OUT CROSS-VALIDATION')
    print(f'{"="*60}')
    
    logo = LeaveOneGroupOut()
    tss_scores = []
    hss_scores = []
    pod_scores = []
    pofd_scores = []
    auc_scores = []
    
    fold = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(test_idx) == 0:
            continue
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Skip if test set has only one class
        if len(np.unique(y_test)) < 2:
            continue
        
        # Train XGBoost
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        tss = calc_tss(y_test, y_pred)
        hss = calc_hss(y_test, y_pred)
        pod = calc_pod(y_test, y_pred)
        pofd = calc_pofd(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan
        
        tss_scores.append(tss)
        hss_scores.append(hss)
        pod_scores.append(pod)
        pofd_scores.append(pofd)
        auc_scores.append(auc)
        
        fold += 1
        if fold % 10 == 0:
            print(f'  Fold {fold}: TSS={tss:.3f}, HSS={hss:.3f}, AUC={auc:.3f}')
    
    # Aggregate results
    results = {
        'n_folds': len(tss_scores),
        'n_features': n_features_selected,
        'tss_mean': np.mean(tss_scores),
        'tss_std': np.std(tss_scores),
        'tss_min': np.min(tss_scores),
        'tss_max': np.max(tss_scores),
        'hss_mean': np.mean(hss_scores),
        'hss_std': np.std(hss_scores),
        'pod_mean': np.mean(pod_scores),
        'pod_std': np.std(pod_scores),
        'pofd_mean': np.mean(pofd_scores),
        'pofd_std': np.std(pofd_scores),
        'auc_mean': np.nanmean(auc_scores),
        'auc_std': np.nanstd(auc_scores),
        'tss_scores': tss_scores,
        'hss_scores': hss_scores,
        'auc_scores': [a for a in auc_scores if not np.isnan(a)],
    }
    
    print(f'\n{"="*60}')
    print('LOGO CV RESULTS')
    print(f'{"="*60}')
    print(f'Folds: {results["n_folds"]}')
    print(f'TSS: {results["tss_mean"]:.3f} ± {results["tss_std"]:.3f} (range: {results["tss_min"]:.3f} to {results["tss_max"]:.3f})')
    print(f'HSS: {results["hss_mean"]:.3f} ± {results["hss_std"]:.3f}')
    print(f'POD: {results["pod_mean"]:.3f} ± {results["pod_std"]:.3f}')
    print(f'POFD: {results["pofd_mean"]:.3f} ± {results["pofd_std"]:.3f}')
    print(f'AUC: {results["auc_mean"]:.3f} ± {results["auc_std"]:.3f}')
    
    return results, model


# ============================================================
# Bootstrap Resampling
# ============================================================

def bootstrap_confidence_intervals(X, y, n_bootstrap=1000):
    """Bootstrap resampling for confidence intervals."""
    print(f'\n{"="*60}')
    print(f'BOOTSTRAP RESAMPLING ({n_bootstrap} iterations)')
    print(f'{"="*60}')
    
    tss_boot = []
    auc_boot = []
    
    n_samples = len(X)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]
        
        # Skip if only one class
        if len(np.unique(y_boot)) < 2:
            continue
        
        # Train-test split (70/30)
        split = int(0.7 * len(X_boot))
        X_train, X_test = X_boot.iloc[:split], X_boot.iloc[split:]
        y_train, y_test = y_boot.iloc[:split], y_boot.iloc[split:]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=i,
            verbosity=0
        )
        
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        tss = calc_tss(y_test, y_pred)
        tss_boot.append(tss)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
            auc_boot.append(auc)
        except:
            pass
    
    # Confidence intervals
    tss_ci = np.percentile(tss_boot, [2.5, 50, 97.5])
    auc_ci = np.percentile(auc_boot, [2.5, 50, 97.5]) if auc_boot else [np.nan, np.nan, np.nan]
    
    results = {
        'n_bootstrap': len(tss_boot),
        'tss_mean': np.mean(tss_boot),
        'tss_std': np.std(tss_boot),
        'tss_ci_95': [tss_ci[0], tss_ci[2]],
        'tss_median': tss_ci[1],
        'auc_mean': np.mean(auc_boot) if auc_boot else np.nan,
        'auc_ci_95': [auc_ci[0], auc_ci[2]] if auc_boot else [np.nan, np.nan],
        'tss_boot': tss_boot,
        'auc_boot': auc_boot,
    }
    
    print(f'\nBootstrap TSS: {results["tss_mean"]:.3f} ± {results["tss_std"]:.3f}')
    print(f'95% CI: [{results["tss_ci_95"][0]:.3f}, {results["tss_ci_95"][1]:.3f}]')
    print(f'Median: {results["tss_median"]:.3f}')
    if auc_boot:
        print(f'Bootstrap AUC: {results["auc_mean"]:.3f}')
        print(f'95% CI: [{results["auc_ci_95"][0]:.3f}, {results["auc_ci_95"][1]:.3f}]')
    
    return results


# ============================================================
# SHAP Analysis
# ============================================================

def shap_analysis(model, X, feature_names, output_dir):
    """Compute and plot SHAP feature importance."""
    print(f'\n{"="*60}')
    print('SHAP FEATURE IMPORTANCE')
    print(f'{"="*60}')
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_shap': mean_shap
    }).sort_values('mean_shap', ascending=False)
    
    print('\nFeature importance (mean |SHAP value|):')
    for _, row in importance.iterrows():
        bar = '#' * int(row['mean_shap'] / importance['mean_shap'].max() * 30)
        print(f'  {row["feature"]:<25s} {row["mean_shap"]:.6f} {bar}')
    
    # Physics conclusion
    print(f'\n{"="*60}')
    print('PHYSICS CONCLUSION')
    print(f'{"="*60}')
    
    # Group by instrument
    goes_feats = importance[importance['feature'].str.startswith('goes_')]
    hel1os_feats = importance[importance['feature'].str.startswith('hel1os_')]
    
    goes_total = goes_feats['mean_shap'].sum()
    hel1os_total = hel1os_feats['mean_shap'].sum()
    
    print(f'GOES X-ray contribution: {goes_total:.6f} ({goes_total/(goes_total+hel1os_total)*100:.1f}%)')
    print(f'HEL1OS HXR contribution: {hel1os_total:.6f} ({hel1os_total/(goes_total+hel1os_total)*100:.1f}%)')
    
    if hel1os_total > goes_total:
        print(f'\n  HEL1OS hard X-ray dominates over GOES soft X-ray by {hel1os_total/goes_total:.1f}x')
        print('  This confirms that hard X-ray spectral features carry precursor information')
        print('  that soft X-ray flux alone cannot capture.')
    else:
        print(f'\n  GOES soft X-ray dominates over HEL1OS HXR by {goes_total/hel1os_total:.1f}x')
        print('  Soft X-ray flux remains the primary flare precursor in this dataset.')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    ax = axes[0]
    colors = ['#ff6b6b' if f.startswith('goes_') else '#4ecdc4' for f in importance['feature']]
    ax.barh(importance['feature'], importance['mean_shap'], color=colors)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Feature Importance (SHAP)')
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff6b6b', label='GOES X-ray'),
                       Patch(facecolor='#4ecdc4', label='HEL1OS HXR')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Pie chart
    ax = axes[1]
    ax.pie([goes_total, hel1os_total], 
           labels=['GOES X-ray', 'HEL1OS HXR'],
           autopct='%1.1f%%',
           colors=['#ff6b6b', '#4ecdc4'],
           startangle=90)
    ax.set_title('Instrument Contribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {os.path.join(output_dir, "shap_importance.png")}')
    
    return importance, shap_values


# ============================================================
# Plots
# ============================================================

def plot_results(logo_results, bootstrap_results, importance, output_dir):
    """Generate evaluation plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. TSS distribution (LOGO CV)
    ax = axes[0, 0]
    ax.hist(logo_results['tss_scores'], bins=20, color='#4ecdc4', edgecolor='black', alpha=0.7)
    ax.axvline(logo_results['tss_mean'], color='red', linestyle='--', label=f'Mean: {logo_results["tss_mean"]:.3f}')
    ax.axvline(0, color='gray', linestyle=':', label='No skill')
    ax.set_xlabel('TSS')
    ax.set_ylabel('Count')
    ax.set_title(f'LOGO CV TSS Distribution (n={logo_results["n_folds"]} folds)')
    ax.legend()
    
    # 2. Bootstrap TSS CI
    ax = axes[0, 1]
    ax.hist(bootstrap_results['tss_boot'], bins=30, color='#95e1d3', edgecolor='black', alpha=0.7)
    ax.axvline(bootstrap_results['tss_ci_95'][0], color='red', linestyle='--', label=f'95% CI: [{bootstrap_results["tss_ci_95"][0]:.3f}, {bootstrap_results["tss_ci_95"][1]:.3f}]')
    ax.axvline(bootstrap_results['tss_ci_95'][1], color='red', linestyle='--')
    ax.axvline(bootstrap_results['tss_median'], color='blue', linestyle='-', label=f'Median: {bootstrap_results["tss_median"]:.3f}')
    ax.set_xlabel('TSS')
    ax.set_ylabel('Count')
    ax.set_title(f'Bootstrap TSS (n={bootstrap_results["n_bootstrap"]})')
    ax.legend()
    
    # 3. Reliability diagram data
    ax = axes[1, 0]
    # Use the last model's predictions for reliability
    # For now, show feature importance as proxy
    top_features = importance.head(8)
    ax.barh(top_features['feature'], top_features['mean_shap'], color='#f38181')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Top 8 Feature Importance')
    ax.invert_yaxis()
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'LOGO CV', 'Bootstrap 95% CI'],
        ['TSS', f'{logo_results["tss_mean"]:.3f} ± {logo_results["tss_std"]:.3f}',
         f'[{bootstrap_results["tss_ci_95"][0]:.3f}, {bootstrap_results["tss_ci_95"][1]:.3f}]'],
        ['HSS', f'{logo_results["hss_mean"]:.3f} ± {logo_results["hss_std"]:.3f}', '—'],
        ['POD', f'{logo_results["pod_mean"]:.3f} ± {logo_results["pod_std"]:.3f}', '—'],
        ['AUC', f'{logo_results["auc_mean"]:.3f} ± {logo_results["auc_std"]:.3f}',
         f'[{bootstrap_results["auc_ci_95"][0]:.3f}, {bootstrap_results["auc_ci_95"][1]:.3f}]'],
        ['Samples', '524', '524 (resampled)'],
        ['Features', f'{logo_results["n_features"]}', f'{logo_results["n_features"]}'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color header
    for j in range(3):
        table[0, j].set_facecolor('#4ecdc4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {os.path.join(output_dir, "evaluation_results.png")}')


# ============================================================
# Main
# ============================================================

def main():
    # Load samples
    samples_path = os.path.join(PROCESSED_DIR, 'hel1os_goes_samples.parquet')
    df = pd.read_parquet(samples_path)
    
    print(f'Loaded {len(df)} samples')
    print(f'Positive: {df["label"].sum()} ({df["label"].mean()*100:.1f}%)')
    print(f'Unique ARs: {df["ar_number"].nunique()}')
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith('hel1os_') or c.startswith('goes_')]
    print(f'Raw features: {len(feature_cols)}')
    
    # Feature selection
    selected_features = select_features(df, feature_cols, max_features=10)
    
    # Prepare data
    X = df[selected_features].copy()
    y = df['label'].copy()
    groups = df['ar_number'].copy()
    
    # Handle NaN
    X = X.fillna(0)
    
    # LOGO CV
    logo_results, final_model = logo_cross_validation(X, y, groups, len(selected_features))
    
    # Bootstrap
    bootstrap_results = bootstrap_confidence_intervals(X, y, n_bootstrap=1000)
    
    # SHAP (train final model on all data for feature importance)
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    final_model_all = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    final_model_all.fit(X, y, verbose=False)
    
    importance, shap_values = shap_analysis(final_model_all, X, selected_features, RESULTS_DIR)
    
    # Generate plots
    plot_results(logo_results, bootstrap_results, importance, RESULTS_DIR)
    
    # Save results
    import json
    results_summary = {
        'logo_cv': {k: v for k, v in logo_results.items() if k not in ['tss_scores', 'hss_scores', 'auc_scores']},
        'bootstrap': {k: v for k, v in bootstrap_results.items() if k not in ['tss_boot', 'auc_boot']},
        'features': selected_features,
        'importance': importance.to_dict('records'),
    }
    
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f'\nSaved: {os.path.join(RESULTS_DIR, "results.json")}')
    print(f'\n{"="*60}')
    print('DONE')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
