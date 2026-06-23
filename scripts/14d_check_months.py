"""Check month subdirs under xrsf-l2-flx1s/2024/ to find actual file naming."""
import urllib.request
import re

base = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-flx1s"

# Check 2024 since we know it has data (GOES-18 was operational)
for month in ["01", "04", "06", "12"]:
    url = f"{base}/2024/{month}/"
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        content = resp.read().decode("utf-8", errors="replace")
        links = re.findall(r'href="([^"]+)"', content)
        files = [l for l in links if ".nc" in l]
        print(f"2024/{month}: {len(files)} .nc files")
        for f in files[:3]:
            print(f"  {f}")
    except Exception as e:
        print(f"2024/{month}: {e}")
    print()

# Also check if 2025 exists
for path in ["2025", "2025/04", "2025/06", "2026", "2026/04"]:
    url = f"{base}/{path}/"
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        content = resp.read().decode("utf-8", errors="replace")
        links = re.findall(r'href="([^"]+)"', content)
        files = [l for l in links if ".nc" in l]
        print(f"{path}: {len(files)} .nc files")
        for f in files[:3]:
            print(f"  {f}")
    except Exception as e:
        print(f"{path}: {e}")
    print()
