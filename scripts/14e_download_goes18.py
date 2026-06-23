"""
Download GOES-18 XRS data for Apr-Jun 2026.
Directory: data/xrsf-l2-flx1s/{YYYY}/{MM}/
File pattern: dn_xrsf-l2-flx1s_g18_d{YYYYMMDD}_v2-2-1.nc
Also try: sci_xrsf-l2-flx1s_g18_d{YYYYMMDD}_v2-2-1.nc (science version)
"""
import urllib.request
import os
import sys
from datetime import datetime, timedelta

OUTPUT_DIR = "data/goes18_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-flx1s"

start_date = datetime(2026, 4, 19)
end_date = datetime(2026, 6, 20)

current = start_date
downloaded = 0
failed = 0
skipped = 0

while current <= end_date:
    date_str = current.strftime("%Y%m%d")
    year = current.strftime("%Y")
    month = current.strftime("%m")
    
    # Try both dn_ and sci_ prefixes
    for prefix in ["sci_", "dn_"]:
        filename = f"{prefix}xrsf-l2-flx1s_g18_d{date_str}_v2-2-1.nc"
        url = f"{BASE_URL}/{year}/{month}/{filename}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            skipped += 1
            break
        
        try:
            urllib.request.urlretrieve(url, filepath)
            size = os.path.getsize(filepath)
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded} files so far (latest: {date_str}, {size} bytes)")
            break
        except Exception as e:
            if "404" in str(e):
                continue  # Try next prefix
            else:
                # File might not exist yet for this date
                pass
    
    current += timedelta(days=1)

print(f"\nDone! Downloaded: {downloaded}, Already existed: {skipped}, Total: {downloaded + skipped}")
print(f"Files in {OUTPUT_DIR}:")
count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".nc")])
print(f"  {count} .nc files")
