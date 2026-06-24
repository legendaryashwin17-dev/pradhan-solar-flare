"""
PRADHAN Auto-Update: Selenium-based data downloader.

Downloads fresh data from:
  1. GOES-18 XRS — NOAA NCEI (urllib, no Selenium needed)
  2. HEL1OS — ISRO SDC portal (Selenium)
  3. SOLEXS — ISRO Aditya-L1 portal (Selenium)
  4. SHARP — JSOC (drms Python package)

Usage:
  python scripts/30_auto_update.py --source all
  python scripts/30_auto_update.py --source goes
  python scripts/30_auto_update.py --source hel1os
  python scripts/30_auto_update.py --source solexs
  python scripts/30_auto_update.py --source sharp

Environment:
  JSOC_EMAIL — registered JSOC email (for SHARP)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent

# Output directories
GOES_DIR = WORKSPACE / "data" / "goes18_2026"
HEL1OS_DIR = WORKSPACE / "data" / "raw" / "hel1os" / "2026"
SOLEXS_DIR = WORKSPACE / "data" / "pradan_solexs"
SHARP_DIR = WORKSPACE / "data" / "raw" / "sharp"

UPDATE_LOG = WORKSPACE / "data" / "update_log.json"


def log_update(source, status, details):
    """Append to update log."""
    log = []
    if UPDATE_LOG.exists():
        with open(UPDATE_LOG) as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "status": status,
        "details": details,
    })
    with open(UPDATE_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ── GOES-18 XRS Download ──────────────────────────────────────────────
def download_goes(start_date=None, end_date=None):
    """Download GOES-18 XRS L2 data from NOAA NCEI."""
    print("\n" + "=" * 70)
    print("GOES-18 XRS DATA DOWNLOAD")
    print("=" * 70)

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    GOES_DIR.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    downloaded = 0
    skipped = 0
    errors = 0
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        year = current.strftime("%Y")
        month = current.strftime("%m")
        day = current.strftime("%d")

        for version in ["v0-0-0", "v0-0-1", "v0-0-2", "v0-0-3", "v1-0-0"]:
            filename = f"sci_xrs-l2-flx1s_g18_d{date_str}_{version}.nc"
            filepath = GOES_DIR / filename

            if filepath.exists():
                skipped += 1
                break

            url = (
                f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
                f"goes/goes18/l2/data/xrs-l2-flx1s/{year}/{month}/{day}/{filename}"
            )

            try:
                urllib.request.urlretrieve(url, str(filepath))
                downloaded += 1
                print(f"  Downloaded: {date_str}")
                break
            except Exception as e:
                if "404" in str(e) or "HTTP Error" in str(e):
                    continue
                errors += 1
                if errors <= 5:
                    print(f"  Warning: {date_str}: {e}")
                break

        current += timedelta(days=1)

    result = {"downloaded": downloaded, "skipped": skipped, "errors": errors}
    log_update("goes", "ok", result)
    print(f"\nGOES: {downloaded} downloaded, {skipped} skipped, {errors} errors")
    return result


# ── HEL1OS Download (Selenium) ────────────────────────────────────────
def download_hel1os(start_date=None, end_date=None):
    """Download HEL1OS FITS files from ISRO SDC portal using Selenium."""
    print("\n" + "=" * 70)
    print("HEL1OS DATA DOWNLOAD (ISRO SDC Portal)")
    print("=" * 70)

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        print("ERROR: selenium not installed. pip install selenium")
        log_update("hel1os", "error", "selenium not installed")
        return {"downloaded": 0, "error": "selenium not installed"}

    HEL1OS_DIR.mkdir(parents=True, exist_ok=True)

    # ISRO SDC portal URL for HEL1OS data
    SDC_URL = "https://sdc-pdc.aditya-l1.isro.gov.in/"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--download-default-directory=" + str(HEL1OS_DIR))
    options.add_experimental_option("prefs", {
        "download.default_directory": str(HEL1OS_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    })

    driver = None
    downloaded = 0
    errors = []

    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 20)

        print(f"  Navigating to {SDC_URL}")
        driver.get(SDC_URL)
        time.sleep(3)

        # Look for HEL1OS data section
        # The ISRO SDC portal typically has instrument selection
        try:
            # Try to find HEL1OS link/button
            hel1os_link = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'HEL1OS')]"))
            )
            hel1os_link.click()
            time.sleep(2)
            print("  Navigated to HEL1OS section")
        except Exception:
            print("  HEL1OS section not found on main page, trying direct URL...")
            # Try direct instrument page
            driver.get(SDC_URL + "?instrument=HEL1OS")
            time.sleep(3)

        # Set date range
        if start_date and end_date:
            try:
                date_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='date']")
                if len(date_inputs) >= 2:
                    date_inputs[0].clear()
                    date_inputs[0].send_keys(start_date)
                    date_inputs[1].clear()
                    date_inputs[1].send_keys(end_date)
                    print(f"  Set date range: {start_date} to {end_date}")
            except Exception as e:
                print(f"  Could not set date range: {e}")

        # Look for download buttons/links
        try:
            download_buttons = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Download') or contains(text(), 'download')]"
            )
            print(f"  Found {len(download_buttons)} download elements")

            for btn in download_buttons[:10]:  # Limit to first 10
                try:
                    href = btn.get_attribute("href")
                    if href and ("fits" in href.lower() or "lc" in href.lower()):
                        filename = href.split("/")[-1]
                        filepath = HEL1OS_DIR / filename
                        if not filepath.exists():
                            urllib.request.urlretrieve(href, str(filepath))
                            downloaded += 1
                            print(f"  Downloaded: {filename}")
                except Exception as e:
                    errors.append(str(e))

        except Exception as e:
            errors.append(f"Download search failed: {e}")

        # Also try direct file listing if available
        try:
            links = driver.find_elements(By.TAG_NAME, "a")
            fits_links = [l for l in links if l.get_attribute("href", "")
                         and (".fits" in l.get_attribute("href", "").lower()
                              or "lightcurve" in l.get_attribute("href", "").lower())]
            print(f"  Found {len(fits_links)} FITS links on page")

            for link in fits_links[:20]:
                try:
                    href = link.get_attribute("href")
                    filename = href.split("/")[-1]
                    filepath = HEL1OS_DIR / filename
                    if not filepath.exists():
                        urllib.request.urlretrieve(href, str(filepath))
                        downloaded += 1
                        print(f"  Downloaded: {filename}")
                except Exception as e:
                    errors.append(str(e))

        except Exception as e:
            errors.append(f"Link search failed: {e}")

    except Exception as e:
        errors.append(f"Selenium error: {e}")
        print(f"  ERROR: {e}")
    finally:
        if driver:
            driver.quit()

    result = {"downloaded": downloaded, "errors": errors[:10]}
    log_update("hel1os", "ok" if downloaded > 0 else "partial", result)
    print(f"\nHEL1OS: {downloaded} downloaded, {len(errors)} errors")
    return result


# ── SOLEXS Download (Selenium) ────────────────────────────────────────
def download_solexs(start_date=None, end_date=None):
    """Download SOLEXS data from ISRO Aditya-L1 portal using Selenium."""
    print("\n" + "=" * 70)
    print("SOLEXS DATA DOWNLOAD (ISRO Aditya-L1 Portal)")
    print("=" * 70)

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        print("ERROR: selenium not installed. pip install selenium")
        log_update("solexs", "error", "selenium not installed")
        return {"downloaded": 0, "error": "selenium not installed"}

    SOLEXS_DIR.mkdir(parents=True, exist_ok=True)

    # ISRO Aditya-L1 science data portal
    PORTAL_URL = "https://aditya-l1.isro.gov.in/"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("prefs", {
        "download.default_directory": str(SOLEXS_DIR),
        "download.prompt_for_download": False,
    })

    driver = None
    downloaded = 0
    errors = []

    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 20)

        print(f"  Navigating to {PORTAL_URL}")
        driver.get(PORTAL_URL)
        time.sleep(3)

        # Look for SOLEXS / Solar X-ray data section
        try:
            solexs_link = wait.until(
                EC.element_to_be_clickable((By.XPATH,
                    "//*[contains(text(), 'SOLEXS') or contains(text(), 'X-Ray') or contains(text(), 'Solar X')]"
                ))
            )
            solexs_link.click()
            time.sleep(2)
            print("  Navigated to SOLEXS/X-ray section")
        except Exception:
            print("  SOLEXS section not found, trying direct URL...")
            driver.get(PORTAL_URL + "?payload=SOLEXS")
            time.sleep(3)

        # Set date range if available
        if start_date and end_date:
            try:
                date_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='date']")
                if len(date_inputs) >= 2:
                    date_inputs[0].clear()
                    date_inputs[0].send_keys(start_date)
                    date_inputs[1].clear()
                    date_inputs[1].send_keys(end_date)
                    print(f"  Set date range: {start_date} to {end_date}")
            except Exception as e:
                print(f"  Could not set date range: {e}")

        # Look for data file links
        try:
            links = driver.find_elements(By.TAG_NAME, "a")
            data_links = [l for l in links if l.get_attribute("href", "")
                         and (".fits" in l.get_attribute("href", "").lower()
                              or ".csv" in l.get_attribute("href", "").lower()
                              or ".parquet" in l.get_attribute("href", "").lower()
                              or "solexs" in l.get_attribute("href", "").lower())]
            print(f"  Found {len(data_links)} data links")

            for link in data_links[:20]:
                try:
                    href = link.get_attribute("href")
                    filename = href.split("/")[-1]
                    filepath = SOLEXS_DIR / filename
                    if not filepath.exists():
                        urllib.request.urlretrieve(href, str(filepath))
                        downloaded += 1
                        print(f"  Downloaded: {filename}")
                except Exception as e:
                    errors.append(str(e))

        except Exception as e:
            errors.append(f"Link search failed: {e}")

        # Try clicking download buttons
        try:
            download_buttons = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Download') or contains(text(), 'download') or contains(@class, 'download')]"
            )
            for btn in download_buttons[:5]:
                try:
                    btn.click()
                    time.sleep(2)
                except Exception:
                    pass
        except Exception:
            pass

    except Exception as e:
        errors.append(f"Selenium error: {e}")
        print(f"  ERROR: {e}")
    finally:
        if driver:
            driver.quit()

    result = {"downloaded": downloaded, "errors": errors[:10]}
    log_update("solexs", "ok" if downloaded > 0 else "partial", result)
    print(f"\nSOLEXS: {downloaded} downloaded, {len(errors)} errors")
    return result


# ── SHARP Download (JSOC drms) ────────────────────────────────────────
def download_sharp(start_date=None, end_date=None):
    """Download SHARP magnetic features from JSOC."""
    print("\n" + "=" * 70)
    print("SHARP MAGNETIC FEATURE DOWNLOAD (JSOC)")
    print("=" * 70)

    try:
        import drms
    except ImportError:
        print("ERROR: drms not installed. pip install drms")
        log_update("sharp", "error", "drms not installed")
        return {"downloaded": 0, "error": "drms not installed"}

    email = os.getenv("JSOC_EMAIL", "legendaryashwin17@gmail.com")
    SHARP_DIR.mkdir(parents=True, exist_ok=True)

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    print(f"  Connecting to JSOC as {email}...")
    try:
        client = drms.Client(email=email)
        print("  JSOC auth OK")
    except Exception as e:
        print(f"  JSOC auth failed: {e}")
        log_update("sharp", "error", str(e))
        return {"downloaded": 0, "error": str(e)}

    series = "hmi.sharp_cea_720s"
    keys = "USFLUX,TOTUSJH,TOTUSJZ,TOTPOT,R_VALUE,SAVNCPP,MEANPOT"

    # Query SHARP records in date range
    query_str = (
        f"{series}[]["
        f"{start.strftime('%Y.%m.%d')}_TAI-"
        f"{end.strftime('%Y.%m.%d')}_TAI]"
        f"{{{keys}}}"
    )

    print(f"  Querying: {query_str[:80]}...")
    try:
        result = client.query(query_str)
        print(f"  Retrieved {len(result)} records")
    except Exception as e:
        print(f"  Query failed: {e}")
        log_update("sharp", "error", str(e))
        return {"downloaded": 0, "error": str(e)}

    if len(result) > 0:
        out_path = SHARP_DIR / "sharp_real.csv"
        # Append to existing if present
        if out_path.exists():
            existing = pd.read_csv(out_path)
            result = pd.concat([existing, result], ignore_index=True)
            result = result.drop_duplicates()

        result.to_csv(out_path, index=False)
        print(f"  Saved {len(result)} records to {out_path}")

        detail = {"records": len(result), "columns": list(result.columns)}
        log_update("sharp", "ok", detail)
        return {"downloaded": len(result)}

    log_update("sharp", "ok", {"records": 0})
    return {"downloaded": 0}


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PRADHAN Auto-Update Data Downloader")
    parser.add_argument("--source", choices=["all", "goes", "hel1os", "solexs", "sharp"],
                        default="all", help="Data source to update")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"PRADHAN AUTO-UPDATE: {args.source.upper()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}

    if args.source in ("all", "goes"):
        results["goes"] = download_goes(args.start, args.end)

    if args.source in ("all", "hel1os"):
        results["hel1os"] = download_hel1os(args.start, args.end)

    if args.source in ("all", "solexs"):
        results["solexs"] = download_solexs(args.start, args.end)

    if args.source in ("all", "sharp"):
        results["sharp"] = download_sharp(args.start, args.end)

    print("\n" + "=" * 70)
    print("UPDATE SUMMARY")
    print("=" * 70)
    for source, res in results.items():
        print(f"  {source.upper():10s}: {res}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
