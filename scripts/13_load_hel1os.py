"""
PRADHAN Script 13 — Load HEL1OS Data & Apply GOES-Trained Model
================================================================

Loads HEL1OS data from downloaded zips, extracts FITS light curves,
combines into parquet, applies the GOES-trained model, and validates.

HEL1OS (High Energy L1 Object Spectrometer) on Aditya-L1:
- CdTe detector: 15-150 keV (hard X-ray)
- CZT detector: 8-30 keV (medium X-ray)

This script:
1. Extracts HEL1OS zips from Downloads
2. Loads CdTe light curves (CTR column)
3. Resamples to 10s cadence
4. Computes features (same 21 features as GOES)
5. Applies GOES-trained XGBoost model
6. Reports predictions and statistics

Usage:
    python scripts/13_load_hel1os.py
"""

import sys
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, HEL1OS_ZIP_DIR, HEL1OS_EXTRACT_DIR, HEL1OS_PARQUET
from src.data.features import compute_features, get_feature_names


def extract_hel1os_zips(
    zip_dir: str = None,
    extract_dir: str = None,
    max_files: int = None,
) -> int:
    """
    Extract ONLY lightcurve files from HEL1OS zips (saves disk space).

    Each zip is ~42 MB but extracts to ~280 MB. We only need
    cdte/lightcurve_cdte1.fits (~11 MB per zip).

    Parameters
    ----------
    zip_dir : str
        Directory containing HLS_*.zip files
    extract_dir : str
        Destination directory
    max_files : int, optional
        Maximum number of zips to extract (None = all)

    Returns
    -------
    int
        Number of zips extracted
    """
    if zip_dir is None:
        zip_dir = HEL1OS_ZIP_DIR
    if extract_dir is None:
        extract_dir = HEL1OS_EXTRACT_DIR

    zip_path = Path(zip_dir)
    ext_path = Path(extract_dir)
    ext_path.mkdir(parents=True, exist_ok=True)

    # Find all HEL1OS zips
    hel1os_zips = sorted(zip_path.glob("HLS_*.zip"))

    if not hel1os_zips:
        print(f"No HLS_*.zip files found in {zip_dir}")
        return 0

    if max_files:
        hel1os_zips = hel1os_zips[:max_files]

    print(f"Found {len(hel1os_zips)} HEL1OS zip files")
    print(f"Extracting ONLY lightcurve_cdte1.fits (saves disk space)")

    extracted = 0
    for i, z in enumerate(hel1os_zips):
        # Check if already extracted (look for lightcurve file)
        zip_stem = z.stem
        # Pattern: YYYY/MM/DD/HLS_.../cdte/lightcurve_cdte1.fits
        # We can check by date prefix
        date_part = zip_stem.split('_')[1]  # YYYYMMDD
        year = date_part[:4]
        month = date_part[4:6]
        day = date_part[6:8]
        expected_dir = ext_path / year / month / day / zip_stem / "cdte"
        expected_file = expected_dir / "lightcurve_cdte1.fits"

        if expected_file.exists():
            continue

        try:
            with zipfile.ZipFile(z, 'r') as zf:
                # Find the lightcurve_cdte1.fits entry
                lc_entries = [n for n in zf.namelist()
                              if n.endswith('lightcurve_cdte1.fits')]

                if not lc_entries:
                    continue

                for entry in lc_entries:
                    # Extract only this file
                    data = zf.read(entry)
                    # Create destination path
                    dest_file = ext_path / entry
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    dest_file.write_bytes(data)

                extracted += 1
        except Exception as e:
            if "No space" not in str(e):
                print(f"  Warning: {z.name}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(hel1os_zips)}")

    print(f"Extracted {extracted} new zips")
    return extracted


def load_hel1os_lightcurves(
    extract_dir: str = None,
    resample_to: str = '10s',
) -> pd.DataFrame:
    """
    Load all HEL1OS CdTe light curves from extracted directory.

    Parameters
    ----------
    extract_dir : str
        Directory with extracted HEL1OS data
    resample_to : str
        Resample cadence

    Returns
    -------
    pd.DataFrame
        Combined light curve data
    """
    from astropy.io import fits

    if extract_dir is None:
        extract_dir = HEL1OS_EXTRACT_DIR

    ext_path = Path(extract_dir)

    # Find all CdTe light curve files
    lc_files = sorted(ext_path.glob("**/lightcurve_cdte1.fits"))

    if not lc_files:
        raise FileNotFoundError(
            f"No lightcurve_cdte1.fits found in {extract_dir}\n"
            f"Run extract first."
        )

    print(f"Found {len(lc_files)} HEL1OS CdTe light curve files")

    dfs = []
    loaded = 0
    errors = 0

    for lc_file in lc_files:
        try:
            with fits.open(str(lc_file)) as hdul:
                data = hdul[1].data
                col_names = [c.upper() for c in data.dtype.names]

                # Parse ISOT string timestamps
                if 'ISOT' in col_names:
                    time_strs = [s.decode() if isinstance(s, bytes) else str(s)
                                 for s in data['ISOT']]
                    times_pd = pd.DatetimeIndex(time_strs)
                elif 'MJD' in col_names:
                    from astropy.time import Time
                    mjd_vals = data['MJD'].astype(float)
                    times_pd = pd.DatetimeIndex(
                        Time(mjd_vals, format='mjd').to_datetime()
                    )
                else:
                    errors += 1
                    continue

                # Count rate
                if 'CTR' in col_names:
                    rate = data['CTR'].astype(float)
                elif 'RATE' in col_names:
                    rate = data['RATE'].astype(float)
                else:
                    errors += 1
                    continue

                # Error
                if 'STAT_ERR' in col_names:
                    error = data['STAT_ERR'].astype(float)
                elif 'ERROR' in col_names:
                    error = data['ERROR'].astype(float)
                else:
                    error = np.sqrt(np.abs(rate))

                df = pd.DataFrame({
                    'rate': rate,
                    'error': error,
                }, index=times_pd)

                df.index.name = 'time'

            dfs.append(df)
            loaded += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: {lc_file.name}: {e}")
            continue

    if not dfs:
        raise ValueError("No HEL1OS light curves could be loaded")

    print(f"Loaded {loaded} files ({errors} errors)")

    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"Combined: {len(combined):,} data points")
    print(f"Time range: {combined.index.min()} to {combined.index.max()}")

    # Resample
    if resample_to and len(combined) > 0:
        print(f"Resampling to {resample_to} cadence...")
        combined = combined[['rate', 'error']].resample(resample_to).mean()
        combined = combined.dropna(subset=['rate'])
        print(f"After resample: {len(combined):,} data points")

    return combined


def apply_goes_model_to_hel1os(hel1os_df: pd.DataFrame) -> dict:
    """
    Apply the GOES-trained model to HEL1OS data.

    Since HEL1OS is a different instrument than GOES, we:
    1. Compute the same 21 statistical features
    2. Apply the trained XGBoost model
    3. Report what the model "sees" in the HEL1OS data

    This is a TRANSFER TEST — we're checking if patterns learned
    from GOES generalize to HEL1OS data.

    Parameters
    ----------
    hel1os_df : pd.DataFrame
        HEL1OS light curve with 'rate' column

    Returns
    -------
    dict
        Results with predictions, features, statistics
    """
    from src.models.forecaster import FlareForecaster

    print("\n[1] Loading GOES-trained model...")
    model_path = Path("models/pradhan_best")
    model_file = Path("models/pradhan_best_model.joblib")
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Run 03_train_best.py first."
        )

    forecaster = FlareForecaster()
    forecaster.load(str(model_path))
    print(f"    Model loaded (threshold={forecaster.threshold:.4f})")

    print("\n[2] Computing features from HEL1OS rate...")
    # Use HEL1OS rate as both "soft" and "hard" proxy
    # (since it's a single channel, we create synthetic dual-channel)
    # For proper dual-channel, pair with SoLEXS
    rate = hel1os_df['rate'].values
    eps = 1e-12
    soft = np.maximum(rate, eps)  # Proxy soft channel
    hard = np.maximum(rate, eps)  # Same as hard (single channel)

    features_df = compute_features(soft, hard, cadence_seconds=10.0)
    feature_names = get_feature_names()
    features_df.index = hel1os_df.index

    # Drop NaN rows
    valid_mask = ~features_df[feature_names].isna().any(axis=1)
    n_valid = valid_mask.sum()
    print(f"    Valid samples: {n_valid:,} / {len(features_df):,}")

    if n_valid == 0:
        return {'error': 'No valid samples after feature computation'}

    print("\n[3] Applying GOES-trained model...")
    X = features_df.loc[valid_mask, feature_names].values
    y_pred_proba = forecaster.predict_proba(X)
    y_pred_binary = (y_pred_proba >= forecaster.threshold).astype(int)

    # Statistics
    nflare = y_pred_binary.sum()
    flare_rate = y_pred_binary.mean()
    mean_proba = y_pred_proba.mean()
    max_proba = y_pred_proba.max()

    print(f"\n    Results:")
    print(f"    Total valid samples: {n_valid:,}")
    print(f"    Predicted flares: {nflare:,} ({flare_rate:.4%})")
    print(f"    Mean probability: {mean_proba:.4f}")
    print(f"    Max probability: {max_proba:.4f}")
    print(f"    Model threshold: {forecaster.threshold:.4f}")

    # Find peak flare times
    peak_idx = np.argsort(y_pred_proba)[-10:][::-1]
    peak_times = hel1os_df.index[valid_mask][peak_idx]
    peak_probas = y_pred_proba[peak_idx]

    print(f"\n    Top 10 peak flare predictions:")
    for t, p in zip(peak_times, peak_probas):
        print(f"      {t}: {p:.4f}")

    results = {
        'n_total': len(hel1os_df),
        'n_valid': int(n_valid),
        'n_predicted_flares': int(nflare),
        'flare_rate': float(flare_rate),
        'mean_probability': float(mean_proba),
        'max_probability': float(max_proba),
        'optimal_threshold': float(forecaster.threshold),
        'peak_times': [str(t) for t in peak_times],
        'peak_probabilities': [float(p) for p in peak_probas],
        'time_range': f"{hel1os_df.index.min()} to {hel1os_df.index.max()}",
        'data_points_per_file': '~43,000 (12h at 1s)',
        'n_files': int(len(list(
            Path(HEL1OS_EXTRACT_DIR).glob("**/lightcurve_cdte1.fits")
        ))),
    }

    return results


def main():
    print("=" * 70)
    print("PRADHAN — HEL1OS Data Loading & GOES Model Application")
    print("=" * 70)

    # Step 1: Extract zips
    print("\n[Step 1] Extracting HEL1OS zips...")
    n_extracted = extract_hel1os_zips()
    print(f"  Extracted: {n_extracted} new zips")

    # Step 2: Load light curves
    print("\n[Step 2] Loading HEL1OS CdTe light curves...")
    hel1os_df = load_hel1os_lightcurves()
    print(f"  Shape: {hel1os_df.shape}")

    # Step 3: Save to parquet
    print("\n[Step 3] Saving to parquet...")
    HEL1OS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    hel1os_df.to_parquet(str(HEL1OS_PARQUET))
    size_mb = Path(HEL1OS_PARQUET).stat().st_size / 1e6
    print(f"  Saved: {HEL1OS_PARQUET} ({size_mb:.1f} MB)")

    # Step 4: Apply GOES model
    print("\n[Step 4] Applying GOES-trained model to HEL1OS data...")
    try:
        results = apply_goes_model_to_hel1os(hel1os_df)

        # Save results
        import json
        results_path = Path("results/hel1os_goes_transfer.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved: {results_path}")

    except FileNotFoundError as e:
        print(f"\n  Skipping model application: {e}")
        print("  (Run 03_train_best.py first to train the GOES model)")

    print("\n" + "=" * 70)
    print("DONE — HEL1OS data loaded and processed")
    print("=" * 70)


if __name__ == "__main__":
    main()
