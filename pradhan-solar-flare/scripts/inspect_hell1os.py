"""Inspect HEL1OS FITS structure."""
from astropy.io import fits
import os

base = r"C:\Users\Admin\.mavis\sessions\mvs_677e6d7ce0a14fb0bae1c8744ba1a972\workspace\pradhan-solar-flare\data\hell1os_sample"
root = os.path.join(base, "2026", "04", "19")

for instrument in ["cdte", "czt"]:
    lc_path = os.path.join(root, f"HLS_20260419_000007_43180sec_lev1_V111", instrument, f"lightcurve_{instrument}1.fits")
    if not os.path.exists(lc_path):
        continue
    print(f"\n=== {instrument.upper()} Lightcurve ===")
    with fits.open(lc_path) as hdul:
        for i, h in enumerate(hdul):
            if h.data is not None and len(h.data) > 0:
                cols = [c.name for c in h.columns]
                print(f"  HDU[{i}] columns: {cols}")
                print(f"  rows: {len(h.data)}")
                for c in h.columns:
                    print(f"    {c.name}: first={h.data[c.name][0]}, last={h.data[c.name][-1]}")
                break

# Also check spectra briefly
spec_path = os.path.join(root, "HLS_20260419_000007_43180sec_lev1_V111", "cdte", "hel1os_cdte_spectra_cdte1.fits")
if os.path.exists(spec_path):
    print(f"\n=== CdTe Spectra ===")
    with fits.open(spec_path) as hdul:
        for i, h in enumerate(hdul):
            if h.data is not None and len(h.data) > 0:
                cols = [c.name for c in h.columns]
                print(f"  HDU[{i}] columns: {cols}")
                print(f"  rows: {len(h.data)}")
                for c in h.columns[:5]:
                    print(f"    {c.name}: first={h.data[c.name][0]}")
                break
