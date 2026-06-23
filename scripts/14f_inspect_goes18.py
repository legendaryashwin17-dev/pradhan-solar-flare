"""
Inspect GOES-18 XRS data structure and load it.
"""
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np

# Load one file to inspect
data_dir = Path("data/goes18_2026")
files = sorted(data_dir.glob("sci_xrsf-l2-flx1s_g18_d*.nc"))
if not files:
    files = sorted(data_dir.glob("dn_xrsf-l2-flx1s_g18_d*.nc"))

print(f"Found {len(files)} GOES-18 files")

# Load first file
ds = xr.open_dataset(str(files[0]))
print(f"\nDataset variables:")
for var in ds.data_vars:
    print(f"  {var}: {ds[var].dims} {ds[var].shape} dtype={ds[var].dtype}")

print(f"\nCoordinates:")
for coord in ds.coords:
    print(f"  {coord}: {ds[coord].values[:5]}")

# Try converting to pandas
df = ds.to_dataframe().reset_index()
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Check for xflux, quality flags
for col in df.columns:
    if "flux" in col.lower() or "xrs" in col.lower():
        print(f"\n{col} stats:")
        print(f"  min={df[col].min():.6e}, max={df[col].max():.6e}, mean={df[col].mean():.6e}")
        print(f"  NaN count: {df[col].isna().sum()}")
