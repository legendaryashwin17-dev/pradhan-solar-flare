"""
Convert GOES NetCDF files to Parquet
=====================================
Converts 8,398+ daily .nc files into yearly parquet files for training.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOES_NETCDF_DIR, DATA_DIR


def convert_nc_to_parquet(
    nc_dir: Path = GOES_NETCDF_DIR,
    output_dir: Path = DATA_DIR / "goes",
    batch_year: bool = True,
):
    """
    Convert all GOES NetCDF files to yearly parquet files.

    Parameters
    ----------
    nc_dir : Path
        Directory containing .nc files
    output_dir : Path
        Output directory for parquet files
    batch_year : bool
        If True, combine by year. If False, one parquet per day.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(nc_dir.glob("*.nc"))
    print(f"Found {len(nc_files)} NetCDF files in {nc_dir}")

    if batch_year:
        # Group files by year
        year_groups = {}
        for f in nc_files:
            # Extract date from filename: sci_xrsf-l2-avg1m_g15_dYYYYMMDD_v*.nc
            parts = f.stem.split("_")
            for p in parts:
                if p.startswith("d") and len(p) == 9:
                    date_str = p[1:]  # Remove 'd' prefix
                    year = int(date_str[:4])
                    year_groups.setdefault(year, []).append(f)
                    break

        print(f"Grouped into {len(year_groups)} years: {sorted(year_groups.keys())}")

        for year in sorted(year_groups.keys()):
            year_files = year_groups[year]
            out_path = output_dir / f"goes_{year}.parquet"

            if out_path.exists():
                existing = pd.read_parquet(out_path)
                print(f"  {year}: Already exists ({len(existing):,} rows) - skipping")
                continue

            print(f"  {year}: Converting {len(year_files)} files...", end=" ", flush=True)
            t0 = time.time()

            dfs = []
            for f in year_files:
                try:
                    ds = xr.open_dataset(f)
                    df = ds.to_dataframe().reset_index()

                    # Keep only relevant columns
                    keep_cols = ["time", "xrsa_flux", "xrsb_flux", "xrsa_flag", "xrsb_flag"]
                    available = [c for c in keep_cols if c in df.columns]
                    df = df[available]

                    # Rename to match training pipeline
                    df = df.rename(columns={
                        "xrsa_flux": "xrsa",
                        "xrsb_flux": "xrsb",
                        "xrsa_flag": "xrsa_quality",
                        "xrsb_flag": "xrsb_quality",
                    })

                    dfs.append(df)
                    ds.close()
                except Exception as e:
                    print(f"\n    Warning: {f.name}: {e}")
                    continue

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                combined = combined.set_index("time").sort_index()
                combined.to_parquet(out_path, engine="pyarrow")
                elapsed = time.time() - t0
                print(f"{len(combined):,} rows ({elapsed:.1f}s)")
            else:
                print("No data loaded")

        print(f"\nConversion complete. Files in {output_dir}:")
        for f in sorted(output_dir.glob("*.parquet")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")

    return output_dir


if __name__ == "__main__":
    convert_nc_to_parquet()
