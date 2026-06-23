"""Check if 2025/2026 data exists in xrsf-l2-flx1s, and get actual file naming."""
import urllib.request
import re

base = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-flx1s"

for year in ["2024", "2025", "2026"]:
    url = f"{base}/{year}/"
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        content = resp.read().decode("utf-8", errors="replace")
        links = re.findall(r'href="([^"]+)"', content)
        files = [l for l in links if l.startswith("sci_") or l.startswith("dn_")]
        print(f"{year}: {len(files)} files, first few:")
        for f in files[:5]:
            print(f"  {f}")
        if files:
            # Check one month
            month_url = f"{url}04/"
            try:
                resp2 = urllib.request.urlopen(month_url, timeout=15)
                content2 = resp2.read().decode("utf-8", errors="replace")
                links2 = re.findall(r'href="([^"]+)"', content2)
                files2 = [l for l in links2 if "sci_" in l]
                print(f"  April {year}: {len(files2)} files")
                for f in files2[:3]:
                    print(f"    {f}")
            except:
                print(f"  April {year}: no month dir")
        print()
    except Exception as e:
        print(f"{year}: {e}\n")
