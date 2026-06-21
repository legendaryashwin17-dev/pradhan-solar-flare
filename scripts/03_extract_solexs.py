"""
PRADHAN Script 03 — Extract SoLEXS Light Curves from ZIPs
=========================================================

Extracts .lc files from 746 downloaded SoLEXS ZIP archives.
Only SDD2 (flare-optimized) .lc files are needed.

Usage:
    python scripts/03_extract_solexs.py

Input:  C:\\Users\\Admin\\Downloads\\AL1_SLX_*.zip  (746 files)
Output: data/pradan_solexs/extracted/  (directory of .lc files)
"""

import zipfile
import shutil
from pathlib import Path
import sys
import time


def extract_solexs_zips(
    zip_dir: str = r"C:\Users\Admin\Downloads",
    output_dir: str = "data/pradan_solexs/extracted",
    instrument: str = "SDD2",
    max_files: int = None,
):
    """
    Extract .lc files from SoLEXS ZIP archives.

    Parameters
    ----------
    zip_dir : str
        Directory containing AL1_SLX_*.zip files
    output_dir : str
        Where to extract .lc files
    instrument : str
        'SDD2' (flare-optimized) or 'SDD1' (quiet Sun)
    max_files : int, optional
        Maximum number of ZIPs to extract (None = all)
    """
    zip_path = Path(zip_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all SoLEXS ZIPs
    zip_files = sorted(zip_path.glob("AL1_SLX*.zip"))

    if not zip_files:
        print(f"No AL1_SLX*.zip files found in {zip_dir}")
        return

    if max_files:
        zip_files = zip_files[:max_files]

    print(f"Found {len(zip_files)} SoLEXS ZIP files")
    print(f"Extracting {instrument} light curves to {output_dir}")

    extracted = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i, zip_file in enumerate(zip_files):
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Find the .lc file for the target instrument
                lc_files = [
                    name for name in zf.namelist()
                    if ('.lc' in name or '.lc.gz' in name) and instrument in name
                ]

                if not lc_files:
                    skipped += 1
                    continue

                for lc_name in lc_files:
                    # Extract to flat structure: output_dir/AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc
                    basename = Path(lc_name).name
                    # Remove .gz extension if present
                    if basename.endswith('.gz'):
                        basename = basename[:-3]
                    out_file = out_path / basename

                    if out_file.exists():
                        skipped += 1
                        continue

                    # Read and write (decompress if .gz)
                    import gzip
                    with zf.open(lc_name) as src:
                        data = src.read()
                        if lc_name.endswith('.gz'):
                            data = gzip.decompress(data)
                        with open(out_file, 'wb') as dst:
                            dst.write(data)

                    extracted += 1

        except zipfile.BadZipFile:
            errors += 1
            print(f"  Bad ZIP: {zip_file.name}")
        except Exception as e:
            errors += 1
            print(f"  Error: {zip_file.name}: {e}")

        # Progress update every 50 files
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(zip_files) - i - 1) / rate
            print(f"  Progress: {i + 1}/{len(zip_files)} "
                  f"({extracted} extracted, {skipped} skipped, {errors} errors) "
                  f"ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped (existing/no {instrument}): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {out_path}")


def verify_extraction(output_dir: str = "data/pradan_solexs/extracted"):
    """Verify extracted files are readable FITS."""
    from astropy.io import fits

    out_path = Path(output_dir)
    lc_files = sorted(out_path.glob("*.lc"))

    print(f"\nVerifying {len(lc_files)} extracted .lc files...")

    readable = 0
    corrupted = 0

    for lc_file in lc_files[:20]:  # Check first 20
        try:
            with fits.open(lc_file) as hdul:
                n_rows = len(hdul[1].data)
                readable += 1
        except Exception:
            corrupted += 1
            print(f"  Corrupted: {lc_file.name}")

    print(f"  Readable: {readable}/{min(20, len(lc_files))}")
    if corrupted:
        print(f"  Corrupted: {corrupted}")
    else:
        print(f"  All files verified OK")


if __name__ == "__main__":
    extract_solexs_zips()
    verify_extraction()
