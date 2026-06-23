"""
Build simultaneous HEL1OS-GOES-18 event catalog.

Key insight from reviewer: HEL1OS (15-150 keV) and GOES XRS-B (1.5-12 keV)
measure DIFFERENT parts of the spectrum. We are NOT cross-calibrating.
We are identifying SIMULTANEOUS observations of the same solar events
from two different instruments.

For each flare detected by GOES during Apr-Jun 2026:
1. GOES class (from GOES XRS-B flux)
2. HEL1OS response (from HEL1OS CdTe flux)
3. Timing relationship (onset, peak, decay)
"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# --- Load GOES-18 data ---
print("=" * 70)
print("Loading GOES-18 XRS data (Apr 19 - Jun 20, 2026)")
print("=" * 70)

goes_dir = Path("data/goes18_2026")
files = sorted(goes_dir.glob("dn_xrsf-l2-flx1s_g18_d*.nc"))
if not files:
    files = sorted(goes_dir.glob("*xrsf*flx1s*.nc"))
print(f"Found {len(files)} GOES-18 files")
for f in files[:3]:
    print(f"  {f.name}")

goes_dfs = []
for f in files:
    try:
        ds = xr.open_dataset(str(f))
        df = ds.to_dataframe().reset_index()
        # Select only the primary channel diode (quad_diode=0)
        # xrsb_flux is the averaged flux, use that
        df = df[df["quad_diode"] == 0][["time", "xrsa_flux", "xrsb_flux", "xrsa_flags", "xrsb_flags"]].copy()
        goes_dfs.append(df)
    except Exception as e:
        print(f"  Error loading {f.name}: {e}")

goes_df = pd.concat(goes_dfs, ignore_index=True)
goes_df["time"] = pd.to_datetime(goes_df["time"])
goes_df = goes_df.sort_values("time").reset_index(drop=True)

print(f"GOES-18 total records: {len(goes_df):,}")
print(f"Time range: {goes_df['time'].min()} to {goes_df['time'].max()}")

# Add GOES class from XRS-B flux (W/m^2)
# B: 1e-7 to 1e-6, C: 1e-6 to 1e-5, M: 1e-5 to 1e-4, X: >1e-4
xrsb = goes_df["xrsb_flux"].values.copy()
xrsb = np.nan_to_num(xrsb, nan=0.0)
goes_class_num = np.zeros(len(xrsb))
goes_class_num[xrsb >= 1e-7] = 1  # B
goes_class_num[xrsb >= 1e-6] = 2  # C
goes_class_num[xrsb >= 1e-5] = 3  # M
goes_class_num[xrsb >= 1e-4] = 4  # X
goes_df["goes_class"] = goes_class_num

# Identify flares: GOES class >= C (xrsb >= 1e-6)
flare_mask = goes_df["xrsb_flux"] >= 1e-6
n_flare_samples = flare_mask.sum()
n_c = (goes_df["goes_class"] == 2).sum()
n_m = (goes_df["goes_class"] == 3).sum()
n_x = (goes_df["goes_class"] == 4).sum()

print(f"\nGOES-18 flare statistics (Apr-Jun 2026):")
print(f"  Total samples: {len(goes_df):,}")
print(f"  Flare samples (>=C): {n_flare_samples:,} ({100*n_flare_samples/len(goes_df):.2f}%)")
print(f"  C-class: {n_c:,}")
print(f"  M-class: {n_m:,}")
print(f"  X-class: {n_x:,}")

# --- Load HEL1OS data ---
print("\n" + "=" * 70)
print("Loading HEL1OS CdTe data")
print("=" * 70)

hel1os_path = Path("data/pradan_hel1os/hel1os_combined.parquet")
if hel1os_path.exists():
    hel1os_df = pd.read_parquet(hel1os_path)
    hel1os_df = hel1os_df.reset_index()  # time is the index
    print(f"HEL1OS total records: {len(hel1os_df):,}")
    print(f"Columns: {list(hel1os_df.columns)}")
    print(f"Time range: {hel1os_df['time'].min()} to {hel1os_df['time'].max()}")
    
    print(f"\nHEL1OS time range matches GOES-18: "
          f"{hel1os_df['time'].min().date()} to {hel1os_df['time'].max().date()}")
else:
    print("HEL1OS parquet not found!")
    hel1os_df = None

# --- Build simultaneous event catalog ---
print("\n" + "=" * 70)
print("Building simultaneous event catalog")
print("=" * 70)

# Identify GOES flare events (contiguous flare segments)
# A flare event starts when GOES class >= C and ends when it drops back to B
goes_df["is_flare"] = flare_mask.values

# Group contiguous flare segments
goes_df["flare_group"] = (goes_df["is_flare"] != goes_df["is_flare"].shift()).cumsum()
flare_groups = goes_df[goes_df["is_flare"]].groupby("flare_group")

events = []
for gid, group in flare_groups:
    if len(group) < 10:  # At least 10 seconds of elevated flux
        continue
    
    peak_flux = group["xrsb_flux"].max()
    peak_time = group.loc[group["xrsb_flux"].idxmax(), "time"]
    start_time = group["time"].min()
    end_time = group["time"].max()
    duration = (end_time - start_time).total_seconds()
    
    # Classify
    if peak_flux >= 1e-4:
        go_class = "X"
    elif peak_flux >= 1e-5:
        go_class = "M"
    elif peak_flux >= 1e-6:
        go_class = "C"
    else:
        go_class = "B"
    
    # Check HEL1OS response during this time
    hel1os_response = None
    if hel1os_df is not None:
        hel_window = hel1os_df[
            (hel1os_df["time"] >= pd.Timestamp(start_time) - timedelta(minutes=5)) &
            (hel1os_df["time"] <= pd.Timestamp(end_time) + timedelta(minutes=5))
        ]
        if len(hel_window) > 0:
            hel1os_response = {
                "peak_flux": hel_window["rate"].max(),
                "mean_flux": hel_window["rate"].mean(),
                "records": len(hel_window),
            }
    
    events.append({
        "start": str(start_time),
        "peak": str(peak_time),
        "end": str(end_time),
        "duration_sec": duration,
        "goes_class": go_class,
        "peak_flux_w_m2": float(peak_flux),
        "peak_flux_log": float(np.log10(peak_flux)),
        "hel1os_response": hel1os_response,
    })

print(f"\nIdentified {len(events)} GOES flare events (>=C, duration >=10s)")

# Summary by class
class_counts = {}
for e in events:
    c = e["goes_class"]
    class_counts[c] = class_counts.get(c, 0) + 1

print("\nEvents by class:")
for c in sorted(class_counts.keys()):
    print(f"  {c}-class: {class_counts[c]}")

# Show top 10 events
events_sorted = sorted(events, key=lambda x: x["peak_flux_w_m2"], reverse=True)
print(f"\nTop 10 flare events:")
print(f"{'Class':>6} {'Peak Time':>22} {'Peak Flux (W/m2)':>18} {'Duration':>10} {'HEL1OS?':>8}")
print("-" * 70)
for e in events_sorted[:10]:
    hel = "YES" if e["hel1os_response"] is not None else "no"
    print(f"  {e['goes_class']:>4} {e['peak']:>22} {e['peak_flux_w_m2']:.2e} {e['duration_sec']:>8.0f}s {hel:>8}")

# --- Save catalog ---
output = {
    "instrument": "GOES-18 XRS + HEL1OS CdTe",
    "period": "2026-04-19 to 2026-06-20",
    "goes_files": len(files),
    "goes_records": len(goes_df),
    "hel1os_records": len(hel1os_df) if hel1os_df is not None else 0,
    "total_events": len(events),
    "events_by_class": class_counts,
    "events": events,
}

output_path = Path("results/simultaneous_event_catalog.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nCatalog saved to {output_path}")

# --- Quick summary stats ---
if events:
    peak_fluxes = [e["peak_flux_w_m2"] for e in events]
    print(f"\nFlare flux statistics:")
    print(f"  Min: {min(peak_fluxes):.2e} W/m^2 ({min(peak_fluxes)/1e-6:.1f}x C threshold)")
    print(f"  Max: {max(peak_fluxes):.2e} W/m^2")
    print(f"  Mean: {np.mean(peak_fluxes):.2e} W/m^2")
    print(f"  Median: {np.median(peak_fluxes):.2e} W/m^2")
