"""Explore GOES-18 directory structure to find actual XRS data URLs."""
import urllib.request
import re

base = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2"

for subpath in ["data/", "data/xrs-l2-flx1s/", "data/xrs-l2-flx1s/2026/"]:
    url = f"{base}/{subpath}"
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        content = resp.read().decode("utf-8", errors="replace")
        links = re.findall(r'href="([^"]+)"', content)
        print(f"\n=== {subpath} ===")
        for f in links[:30]:
            print(f"  {f}")
    except Exception as e:
        print(f"\n=== {subpath} === FAILED: {e}")
