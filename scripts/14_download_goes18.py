"""
Download GOES-18 XRS data for Apr-Jun 2026.
This gives us simultaneous observations with HEL1OS.

GOES-18 XRS channels:
- XRS-A: 0.5-4 A (soft X-ray, ~3-25 keV) — equivalent to GOES XRS-A
- XRS-B: 1-8 A (hard X-ray, ~1.5-12 keV) — equivalent to GOES XRS-B

These are DIFFERENT from HEL1OS (10-150 keV) — they measure different
parts of the flare spectrum. This is the correct scientific approach:
use simultaneous multi-instrument observations, not cross-calibration.
"""
import sys
from pathlib import Path
import urllib.request
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("data/goes18_2026")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_goes18_xrs(start_date="2026-04-19", end_date="2026-06-20"):
    """
    Download GOES-18 XRS L2 data from NOAA NCEI.
    
    Uses the GOES-R data archive at data.ngdc.noaa.gov.
    Files are daily NetCDF with 1-minute cadence.
    """
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    downloaded = 0
    errors = 0
    
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        year = current.strftime("%Y")
        month = current.strftime("%m")
        day = current.strftime("%d")
        
        # GOES-18 XRS L2 data URL pattern
        # Try different version numbers
        for version in ["v0-0-0", "v0-0-1", "v0-0-2", "v0-0-3", "v1-0-0"]:
            url = (
                f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
                f"goes/goes18/l2/data/xrs-l2-flx1s/{year}/{month}/{day}/"
                f"sci_xrs-l2-flx1s_g18_d{date_str}_{version}.nc"
            )
            
            filename = f"sci_xrs-l2-flx1s_g18_d{date_str}_{version}.nc"
            filepath = OUTPUT_DIR / filename
            
            if filepath.exists():
                downloaded += 1
                break
            
            try:
                urllib.request.urlretrieve(url, str(filepath))
                downloaded += 1
                print(f"  Downloaded: {date_str}")
                break
            except Exception as e:
                if "404" in str(e) or "HTTP Error" in str(e):
                    continue
                else:
                    errors += 1
                    if errors <= 3:
                        print(f"  Warning: {date_str}: {e}")
                    break
        
        current += timedelta(days=1)
    
    print(f"\nDownloaded: {downloaded} files, {errors} errors")
    return downloaded


def try_sunpy_download():
    """Alternative: use sunpy to download GOES-18 data."""
    try:
        import sunpy
        from sunpy.net import Fido, attrs as a
        from astropy.time import Time
        
        print("Trying sunpy Fido download...")
        
        # Search for GOES-18 XRS data
        results = Fido.search(
            a.Time("2026-04-19", "2026-04-25"),  # Start with 1 week
            a.Instrument("XRS"),
            a.Goes("18"),
        )
        
        print(f"Found {len(results)} files")
        
        if len(results) > 0:
            # Download
            downloaded = Fido.fetch(results, path=str(OUTPUT_DIR))
            print(f"Downloaded {len(downloaded)} files")
            return True
            
    except ImportError:
        print("sunpy not available")
    except Exception as e:
        print(f"sunpy download failed: {e}")
    
    return False


if __name__ == "__main__":
    print("=" * 70)
    print("GOES-18 XRS Data Download for HEL1OS Cross-Instrument Analysis")
    print("=" * 70)
    
    # Try direct download first
    print("\n[1] Direct download from NOAA NCEI...")
    n = download_goes18_xrs()
    
    if n == 0:
        print("\n[2] Trying sunpy...")
        try_sunpy_download()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
