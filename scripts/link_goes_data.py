"""
Link GOES parquet data into workspace
======================================
Creates symlinks from external GOES data to workspace/data/goes/.
Run this once to set up data access.
"""

import os
import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GOES_PARQUET_DIR, DATA_DIR


def link_goes_data():
    """Create symlinks for GOES parquet files."""
    goes_link_dir = DATA_DIR / "goes"
    goes_link_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(GOES_PARQUET_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {GOES_PARQUET_DIR}")

    linked = 0
    skipped = 0
    for src in parquet_files:
        dst = goes_link_dir / src.name
        if dst.exists():
            skipped += 1
            continue

        # Try symlink first (requires admin or dev mode on Windows)
        try:
            os.symlink(src, dst)
            linked += 1
        except OSError:
            # Fallback: create a .txt pointer file
            pointer = dst.with_suffix(".ptr")
            pointer.write_text(str(src))
            linked += 1

    print(f"Linked: {linked}, Already exists: {skipped}")
    print(f"GOES data accessible at: {goes_link_dir}")

    # Verify
    files = list(goes_link_dir.glob("*.parquet")) + list(goes_link_dir.glob("*.ptr"))
    print(f"Total files in link dir: {len(files)}")
    return goes_link_dir


if __name__ == "__main__":
    link_goes_data()
