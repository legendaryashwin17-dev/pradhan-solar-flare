"""Find XRS data in GOES-18 directory — it might be named differently."""
import urllib.request
import re

base = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2"

# Get full listing of data/
url = f"{base}/data/"
resp = urllib.request.urlopen(url, timeout=15)
content = resp.read().decode("utf-8", errors="replace")
links = re.findall(r'href="([^"]+)"', content)

print("All entries in data/:")
for f in links:
    if f.endswith("/"):
        name = f.rstrip("/")
        # Filter out navigation
        if name and not name.startswith("?") and not name.startswith("/") and not name.startswith("http") and name not in [".", "..", "Parent Directory"]:
            print(f"  {name}/")
            # Check if it contains XRS
            if "xrs" in name.lower() or "xray" in name.lower():
                try:
                    sub_resp = urllib.request.urlopen(f"{url}{name}/", timeout=10)
                    sub_content = sub_resp.read().decode("utf-8", errors="replace")
                    sub_links = re.findall(r'href="([^"]+)"', sub_content)
                    print(f"    -> contents:")
                    for s in sub_links[:10]:
                        print(f"       {s}")
                except:
                    pass
