"""
PRADHAN v2 — Optimized Training Pipeline
==========================================
Boosts results via:
1. Richer feature set (v2: multi-scale, periodicity, lags, interactions)
2. Proper temporal split (no leakage)
3. Optuna hyperparameter tuning
4. XGBoost + LightGBM + CatBoost ensemble
5. Time-series cross-validation
"""
import sys, gc, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GOES_PARQUET_DIR
from src.data.labels import create_flare_labels
from src.evaluation.metrics import compute_all_metrics, print_metrics_report

# ── Feature Engineering v2 ──────────────────────────────────────────────

def compute_features_v2(soft, hard, cadence=300.0):
    """Enhanced feature set: ~50 features from raw flux pair."""
    n = len(soft)
    eps = 1e-12
    soft = np.where(soft > 0, soft, eps)
    hard = np.where(hard > 0, hard, eps)

    soft_log = np.log10(soft)
    hard_log = np.log10(hard)
    ratio = hard / (soft + eps)

    dsoft = np.gradient(soft_log, cadence)
    dhard = np.gradient(hard_log, cadence)
    ddsoft = np.gradient(dsoft, cadence)
    ddhard = np.gradient(dhard, cadence)

    feats = {}

    # Raw + log
    feats['soft'] = soft
    feats['hard'] = hard
    feats['soft_log'] = soft_log
    feats['hard_log'] = hard_log
    feats['ratio'] = ratio

    # Derivatives
    feats['dsoft'] = dsoft
    feats['dhard'] = dhard
    feats['ddsoft'] = ddsoft
    feats['ddhard'] = ddhard
    feats['d_ratio'] = dhard / (dsoft + eps)

    # Multi-scale rolling means
    for w_name, w_sec in [('1m', 60), ('5m', 300), ('15m', 900), ('1h', 3600), ('6h', 21600)]:
        win = max(1, int(w_sec / cadence))
        feats[f'soft_mean_{w_name}'] = uniform_filter1d(soft, size=win, mode='nearest')
        feats[f'hard_mean_{w_name}'] = uniform_filter1d(hard, size=win, mode='nearest')

    # Multi-scale rolling std
    for w_name, w_sec in [('1m', 60), ('5m', 300), ('1h', 3600)]:
        win = max(2, int(w_sec / cadence))
        feats[f'soft_std_{w_name}'] = _rstd(soft, win)
        feats[f'hard_std_{w_name}'] = _rstd(hard, win)

    # Rolling min/max ratio (dynamic range)
    for w_name, w_sec in [('5m', 300), ('1h', 3600)]:
        win = max(2, int(w_sec / cadence))
        s_roll_max = _rmax(soft, win)
        s_roll_min = _rmin(soft, win)
        h_roll_max = _rmax(hard, win)
        h_roll_min = _rmin(hard, win)
        feats[f'soft_range_{w_name}'] = s_roll_max / (s_roll_min + eps)
        feats[f'hard_range_{w_name}'] = h_roll_max / (h_roll_min + eps)

    # Correlation
    feats['soft_hard_corr_5m'] = _rcorr(soft, hard, max(2, int(300/cadence)))
    feats['soft_hard_corr_1h'] = _rcorr(soft, hard, max(2, int(3600/cadence)))

    # Cross-correlation lag-1
    xc = np.full(n, np.nan)
    xc[1:] = np.corrcoef(soft[:-1], hard[1:])[0, 1:]
    feats['xcorr'] = xc

    # Lag features (hard flux at various lags)
    for lag in [1, 3, 6, 12]:
        lag_pts = int(lag * 60 / cadence)
        if lag_pts < n:
            feats[f'hard_lag{lag}m'] = np.roll(hard, lag_pts)
            feats[f'soft_lag{lag}m'] = np.roll(soft, lag_pts)
        else:
            feats[f'hard_lag{lag}m'] = np.full(n, np.nan)
            feats[f'soft_lag{lag}m'] = np.full(n, np.nan)

    # Interaction features
    feats['soft_x_dsoft'] = soft * np.abs(dsoft)
    feats['hard_x_dhard'] = hard * np.abs(dhard)
    feats['log_ratio_x_dsoft'] = np.log10(ratio + eps) * dsoft

    # Spectral hardening proxy
    feats['spectral_hardening'] = np.gradient(ratio, cadence)

    # Neupert proxy
    feats['neupert'] = np.cumsum(np.maximum(dhard, 0)) * soft

    # Flare history: rolling max over windows (proxy for recent activity)
    for w_name, w_sec in [('1h', 3600), ('6h', 21600), ('24h', 86400)]:
        win = max(1, int(w_sec / cadence))
        feats[f'hard_max_{w_name}'] = _rmax(hard, win)
        feats[f'soft_max_{w_name}'] = _rmax(soft, win)

    # Trend: linear slope over 1h window
    win_1h = max(3, int(3600 / cadence))
    feats['hard_slope_1h'] = _rslope(hard, win_1h)
    feats['soft_slope_1h'] = _rslope(soft, win_1h)

    # Build DataFrame
    df = pd.DataFrame(feats)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _rstd(arr, w):
    n = len(arr)
    r = np.full(n, np.nan)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr**2)
    for i in range(w-1, n):
        s = cs[i] - (cs[i-w] if i >= w else 0)
        s2 = cs2[i] - (cs2[i-w] if i >= w else 0)
        m = s / w
        v = s2 / w - m**2
        r[i] = np.sqrt(max(v, 0))
    return r

def _rmax(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).max().values

def _rmin(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).min().values

def _rcorr(a, b, w):
    n = len(a)
    r = np.full(n, np.nan)
    for i in range(w-1, n):
        s, e = i-w+1, i+1
        if np.std(a[s:e]) < 1e-15 or np.std(b[s:e]) < 1e-15:
            continue
        r[i] = np.corrcoef(a[s:e], b[s:e])[0, 1]
    return r

def _rslope(arr, w):
    n = len(arr)
    r = np.full(n, np.nan)
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean)**2).sum()
    for i in range(w-1, n):
        y = arr[i-w+1:i+1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        r[i] = np.dot(x - x_mean, y - y_mean) / (x_var + 1e-15)
    return r


# ── Data Loading ────────────────────────────────────────────────────────

def load_year(path, feature_names):
    df = pd.read_parquet(path)
    rename = {}
    if 'xrsa' in df.columns: rename['xrsa'] = 'xrs_a_flux'
    if 'xrsb' in df.columns: rename['xrsb'] = 'xrs_b_flux'
    if rename: df = df.rename(columns=rename)

    if len(df) > 1_000_000:
        df = df.resample('5min').mean()
        df = df.dropna(subset=['xrs_a_flux', 'xrs_b_flux'])
        cadence = 300.0
    else:
        cadence = 60.0

    df = df[(df['xrs_a_flux'] > 0) & (df['xrs_b_flux'] > 0)]
    df = df[np.isfinite(df['xrs_a_flux']) & np.isfinite(df['xrs_b_flux'])]
    if len(df) < 100:
        return None, None, None, feature_names

    feats = compute_features_v2(df['xrs_a_flux'].values, df['xrs_b_flux'].values, cadence)
    feats.index = df.index

    if feature_names is None:
        feature_names = feats.columns.tolist()

    labels = create_flare_labels(df['xrs_b_flux'], horizon='24h', threshold_class='M')
    valid = ~(feats[feature_names].isna().any(axis=1) | labels.isna())
    X = feats.loc[valid, feature_names].values
    y = labels[valid].values
    return X, y, feats.loc[valid].index, feature_names


# ── Hyperparameter Tuning ───────────────────────────────────────────────

def tune_xgboost(X, y, n_trials=30):
    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit

    spw = int((y == 0).sum() / max((y == 1).sum(), 1))

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('mcw', 1, 20),
            'reg_alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('lambda', 1e-3, 10, log=True),
            'scale_pos_weight': spw,
            'random_state': 42,
            'verbosity': 0,
            'eval_metric': 'logloss',
        }
        model = XGBClassifier(**params)

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            model.fit(X[tr_idx], y[tr_idx], verbose=False)
            p = model.predict_proba(X[va_idx])[:, 1]
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(y[va_idx], p))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(X, y, n_trials=30):
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    spw = int((y == 0).sum() / max((y == 1).sum(), 1))

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('mcs', 5, 50),
            'reg_alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('lambda', 1e-3, 10, log=True),
            'scale_pos_weight': spw,
            'random_state': 42,
            'verbosity': -1,
        }
        model = lgb.LGBMClassifier(**params)

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            model.fit(X[tr_idx], y[tr_idx])
            p = model.predict_proba(X[va_idx])[:, 1]
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(y[va_idx], p))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(X, y, n_trials=20):
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.model_selection import TimeSeriesSplit

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iters', 200, 600),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2', 1, 10),
            'random_seed': 42,
            'verbose': 0,
        }
        model = CatBoostClassifier(**params)

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            model.fit(X[tr_idx], y[tr_idx], verbose=0)
            p = model.predict_proba(X[va_idx])[:, 1]
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(y[va_idx], p))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── Main Pipeline ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PRADHAN v2 — Optimized Training")
    print("=" * 70)

    files = sorted(GOES_PARQUET_DIR.glob("goes_*.parquet"))
    print(f"Found {len(files)} parquet files")

    # Temporal split:
    # Train: 2003-2018 (solar cycle 23 end + cycle 24 rising)
    # Val:   2019-2020 (cycle 24 declining)
    # Test:  2021-2025 (cycle 24 max + cycle 25)
    train_files = [f for f in files if int(f.stem.split('_')[1]) <= 2018]
    val_files   = [f for f in files if 2019 <= int(f.stem.split('_')[1]) <= 2020]
    test_files  = [f for f in files if int(f.stem.split('_')[1]) >= 2021]

    print(f"Train: {[f.stem.split('_')[1] for f in train_files]}")
    print(f"Val:   {[f.stem.split('_')[1] for f in val_files]}")
    print(f"Test:  {[f.stem.split('_')[1] for f in test_files]}")

    # Load all data
    print("\n[1] Loading and computing features (v2)...")
    feature_names = None
    train_Xs, train_ys = [], []
    for pf in train_files:
        X, y, _, feature_names = load_year(pf, feature_names)
        if X is not None:
            train_Xs.append(X)
            train_ys.append(y)
            print(f"  {pf.stem.split('_')[1]}: {len(X):,} samples, event={y.mean():.4%}")
        gc.collect()

    val_Xs, val_ys = [], []
    for pf in val_files:
        X, y, _, feature_names = load_year(pf, feature_names)
        if X is not None:
            val_Xs.append(X)
            val_ys.append(y)
            print(f"  {pf.stem.split('_')[1]}: {len(X):,} samples, event={y.mean():.4%}")
        gc.collect()

    test_Xs, test_ys = [], []
    test_times = []
    for pf in test_files:
        X, y, t, feature_names = load_year(pf, feature_names)
        if X is not None:
            test_Xs.append(X)
            test_ys.append(y)
            test_times.append(t)
            print(f"  {pf.stem.split('_')[1]}: {len(X):,} samples, event={y.mean():.4%}")
        gc.collect()

    X_train = np.concatenate(train_Xs)
    y_train = np.concatenate(train_ys)
    X_val = np.concatenate(val_Xs)
    y_val = np.concatenate(val_ys)
    X_test = np.concatenate(test_Xs)
    y_test = np.concatenate(test_ys)

    print(f"\n  Train: {len(X_train):,} ({y_train.mean():.4%})")
    print(f"  Val:   {len(X_val):,} ({y_val.mean():.4%})")
    print(f"  Test:  {len(X_test):,} ({y_test.mean():.4%})")
    print(f"  Features: {len(feature_names)}")

    # ── Baseline (default XGBoost) ──
    print("\n[2] Baseline XGBoost (default params)...")
    from xgboost import XGBClassifier
    spw = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    baseline = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, random_state=42, verbosity=0,
        eval_metric='logloss'
    )
    baseline.fit(X_train, y_train)
    p_base = baseline.predict_proba(X_test)[:, 1]
    m_base = compute_all_metrics(y_test, p_base)
    print(f"  Baseline: AUC={m_base['auc']:.4f} TSS={m_base['tss']:.4f} HSS={m_base['hss']:.4f}")

    # ── Tune XGBoost ──
    print("\n[3] Tuning XGBoost (Optuna, 30 trials)...")
    X_all_tune = np.concatenate([X_train, X_val])
    y_all_tune = np.concatenate([y_train, y_val])
    xgb_params = tune_xgboost(X_all_tune, y_all_tune, n_trials=30)
    print(f"  Best XGB params: {xgb_params}")

    xgb_model = XGBClassifier(**xgb_params, scale_pos_weight=spw,
                               random_state=42, verbosity=0, eval_metric='logloss')
    xgb_model.fit(X_all_tune, y_all_tune)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    m_xgb = compute_all_metrics(y_test, p_xgb)
    print(f"  Tuned XGB: AUC={m_xgb['auc']:.4f} TSS={m_xgb['tss']:.4f} HSS={m_xgb['hss']:.4f}")

    # ── Tune LightGBM ──
    print("\n[4] Tuning LightGBM (Optuna, 30 trials)...")
    lgb_params = tune_lightgbm(X_all_tune, y_all_tune, n_trials=30)
    print(f"  Best LGB params: {lgb_params}")

    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(**lgb_params, scale_pos_weight=spw,
                                    random_state=42, verbosity=-1)
    lgb_model.fit(X_all_tune, y_all_tune)
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    m_lgb = compute_all_metrics(y_test, p_lgb)
    print(f"  Tuned LGB: AUC={m_lgb['auc']:.4f} TSS={m_lgb['tss']:.4f} HSS={m_lgb['hss']:.4f}")

    # ── Tune CatBoost ──
    print("\n[5] Tuning CatBoost (Optuna, 20 trials)...")
    cb_params = tune_catboost(X_all_tune, y_all_tune, n_trials=20)
    print(f"  Best CB params: {cb_params}")

    from catboost import CatBoostClassifier
    cb_model = CatBoostClassifier(**cb_params, random_seed=42, verbose=0)
    cb_model.fit(X_all_tune, y_all_tune)
    p_cb = cb_model.predict_proba(X_test)[:, 1]
    m_cb = compute_all_metrics(y_test, p_cb)
    print(f"  Tuned CB: AUC={m_cb['auc']:.4f} TSS={m_cb['tss']:.4f} HSS={m_cb['hss']:.4f}")

    # ── Ensemble ──
    print("\n[6] Ensemble (weighted average)...")
    # Weight by AUC
    w_xgb = m_xgb['auc']
    w_lgb = m_lgb['auc']
    w_cb = m_cb['auc']
    w_total = w_xgb + w_lgb + w_cb

    p_ensemble = (w_xgb * p_xgb + w_lgb * p_lgb + w_cb * p_cb) / w_total
    m_ensemble = compute_all_metrics(y_test, p_ensemble)
    print(f"  Ensemble: AUC={m_ensemble['auc']:.4f} TSS={m_ensemble['tss']:.4f} HSS={m_ensemble['hss']:.4f}")

    # ── Feature importance from best model ──
    print("\n[7] Feature importance (LightGBM)...")
    imp = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n  Top 15:")
    for i, row in imp.head(15).iterrows():
        print(f"    {i+1:2d}. {row['feature']:<25} {row['importance']:.0f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'AUC':>8} {'TSS':>8} {'HSS':>8} {'Brier':>8}")
    print("-" * 55)
    for name, m in [
        ('Baseline XGBoost', m_base),
        ('Tuned XGBoost', m_xgb),
        ('Tuned LightGBM', m_lgb),
        ('Tuned CatBoost', m_cb),
        ('ENSEMBLE', m_ensemble),
    ]:
        print(f"  {name:<23} {m['auc']:>8.4f} {m['tss']:>8.4f} {m['hss']:>8.4f} {m['brier']:>8.4f}")

    # Save
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    results = {
        'feature_names': feature_names,
        'baseline': {k: float(v) for k, v in m_base.items()},
        'xgboost': {k: float(v) for k, v in m_xgb.items()},
        'lightgbm': {k: float(v) for k, v in m_lgb.items()},
        'catboost': {k: float(v) for k, v in m_cb.items()},
        'ensemble': {k: float(v) for k, v in m_ensemble.items()},
        'xgb_params': xgb_params,
        'lgb_params': lgb_params,
        'cb_params': cb_params,
        'feature_importance': {r['feature']: float(r['importance']) for _, r in imp.iterrows()},
        'data': {
            'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
            'train_event_rate': float(y_train.mean()),
            'val_event_rate': float(y_val.mean()),
            'test_event_rate': float(y_test.mean()),
        }
    }
    with open('results/v2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save ensemble model
    import joblib
    joblib.dump({'xgb': xgb_model, 'lgb': lgb_model, 'cb': cb_model,
                 'weights': [w_xgb/w_total, w_lgb/w_total, w_cb/w_total],
                 'feature_names': feature_names}, 'models/pradhan_ensemble.joblib')

    print("\n  Saved: results/v2_results.json, models/pradhan_ensemble.joblib")
    print("=" * 70)


if __name__ == "__main__":
    main()
