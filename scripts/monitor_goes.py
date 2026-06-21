"""
GOES Download Monitor
======================
Watches the GOES data directory and reports new files.
Run in a separate terminal while download is in progress.
"""

import time
import os
from pathlib import Path
from datetime import datetime


def monitor_goes(
    data_dir: str = r"C:\Users\Admin\aditya-flare-forecast\data\goes_historical",
    interval: int = 30,
):
    """
    Monitor GOES download progress.

    Parameters
    ----------
    data_dir : str
        Directory to watch
    interval : int
        Check interval in seconds
    """
    data_path = Path(data_dir)

    print("=" * 60)
    print("GOES Download Monitor")
    print(f"Watching: {data_dir}")
    print(f"Refresh interval: {interval}s")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    prev_nc = set()
    prev_parquet = set()
    start_time = time.time()

    try:
        while True:
            current_nc = set(f.name for f in data_path.glob("*.nc"))
            current_parquet = set(f.name for f in data_path.glob("*.parquet"))

            new_nc = current_nc - prev_nc
            new_parquet = current_parquet - prev_parquet

            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            total_size = sum(f.stat().st_size for f in data_path.glob("*"))
            total_gb = total_size / (1024 ** 3)

            now = datetime.now().strftime("%H:%M:%S")

            print(f"\r[{now}] NC: {len(current_nc):,} (+{len(new_nc)}) | "
                  f"PQ: {len(current_parquet):,} (+{len(new_parquet)}) | "
                  f"Size: {total_gb:.2f} GB | "
                  f"Elapsed: {elapsed_str}", end="", flush=True)

            if new_nc:
                print(f"\n  New NC files: {', '.join(sorted(new_nc)[:5])}{'...' if len(new_nc) > 5 else ''}")

            prev_nc = current_nc
            prev_parquet = current_parquet

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\nMonitor stopped. Final count: {len(prev_nc):,} NC files, {len(prev_parquet):,} parquet files")


if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    monitor_goes(interval=interval)
