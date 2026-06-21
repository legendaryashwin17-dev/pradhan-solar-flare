"""
PRADHAN Visualization — GOES X-ray Light Curves
=================================================
Generate publication-quality plots of GOES data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, FIGURES_DIR, FLUX_THRESHOLDS, VIZ_CONFIG

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LogLocator, LogFormatterSciNotation


def load_goes():
    """Load GOES data from linked parquet files."""
    goes_dir = DATA_DIR / "goes"
    parquet_files = sorted(goes_dir.glob("*.parquet"))

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)

    goes = pd.concat(dfs, ignore_index=False)
    goes = goes.sort_index()

    # Handle both naming conventions
    if "xrsa" in goes.columns and "xrs_a_flux" not in goes.columns:
        goes = goes.rename(columns={"xrsa": "xrs_a_flux", "xrsb": "xrs_b_flux"})
    goes = goes.loc[:, ~goes.columns.duplicated()]

    return goes


def plot_full_timeline(goes, save_path=None):
    """Plot the full GOES X-ray timeline (2003-2017)."""
    fig, ax = plt.subplots(figsize=(16, 6))

    flux_b = goes["xrs_b_flux"].clip(lower=1e-9)
    flux_a = goes["xrs_a_flux"].clip(lower=1e-9)

    ax.semilogy(goes.index, flux_b, color="#F44336", linewidth=0.3, alpha=0.7, label="XRS-B (0.1-0.8 nm)")
    ax.semilogy(goes.index, flux_a, color="#2196F3", linewidth=0.3, alpha=0.5, label="XRS-A (0.05-0.4 nm)")

    # Flare classification lines
    colors = {"C": "#FF9800", "M": "#E91E63", "X": "#9C27B0"}
    for cls in ["C", "M", "X"]:
        ax.axhline(y=FLUX_THRESHOLDS[cls], color=colors[cls], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(goes.index[-1], FLUX_THRESHOLDS[cls] * 1.2, cls, color=colors[cls], fontsize=8, fontweight="bold")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("X-ray Flux (W/m2)", fontsize=12)
    ax.set_title("GOES X-ray Flux (2003-2017) - PRADHAN Dataset", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(1e-9, 1e-2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "goes_full_timeline.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_flare_events(goes, year=2003, save_path=None):
    """Plot a single year with flare events highlighted."""
    mask = goes.index.year == year
    goes_year = goes[mask]

    if len(goes_year) == 0:
        print(f"No data for year {year}")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    flux_b = goes_year["xrs_b_flux"].clip(lower=1e-9)

    # Color by classification
    colors = np.where(flux_b >= FLUX_THRESHOLDS["X"], "#9C27B0",
             np.where(flux_b >= FLUX_THRESHOLDS["M"], "#E91E63",
             np.where(flux_b >= FLUX_THRESHOLDS["C"], "#FF9800", "#F44336")))

    ax.semilogy(goes_year.index, flux_b, color="#F44336", linewidth=0.5, alpha=0.8)

    # Mark major flares
    major_mask = flux_b >= FLUX_THRESHOLDS["M"]
    if major_mask.sum() > 0:
        ax.scatter(goes_year.index[major_mask], flux_b[major_mask],
                   c="#9C27B0", s=10, zorder=5, label="M/X class flares")

    # Classification lines
    for cls in ["C", "M", "X"]:
        ax.axhline(y=FLUX_THRESHOLDS[cls], color="#FF9800", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("XRS-B Flux (W/m2)", fontsize=12)
    ax.set_title(f"GOES X-ray Flux - {year}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / f"goes_flare_events_{year}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_flare_distribution(goes, save_path=None):
    """Plot flare class distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    flux_b = goes["xrs_b_flux"].clip(lower=1e-9)
    log_flux = np.log10(flux_b)

    bins = np.linspace(-9, -2, 100)
    ax.hist(log_flux, bins=bins, color="#F44336", alpha=0.7, edgecolor="white", linewidth=0.3)

    # Classification lines
    labels = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
    for cls, thresh in labels.items():
        ax.axvline(x=np.log10(thresh), color="#FF9800", linestyle="--", linewidth=1)
        ax.text(np.log10(thresh), ax.get_ylim()[1] * 0.9, cls, ha="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("log10(XRS-B Flux) [W/m2]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("GOES X-ray Flux Distribution (2003-2017)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "goes_flux_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_monthly_event_rate(goes, save_path=None):
    """Plot monthly flare event rate over time."""
    flux_b = goes["xrs_b_flux"]

    monthly = flux_b.resample("ME").agg(
        mean_flux="mean",
        max_flux="max",
        c_count=lambda x: (x >= FLUX_THRESHOLDS["C"]).sum(),
        m_count=lambda x: (x >= FLUX_THRESHOLDS["M"]).sum(),
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Mean and max flux
    ax1.semilogy(monthly.index, monthly["mean_flux"], color="#2196F3", linewidth=1, label="Mean flux")
    ax1.semilogy(monthly.index, monthly["max_flux"], color="#F44336", linewidth=0.8, alpha=0.7, label="Max flux")
    ax1.set_ylabel("XRS-B Flux (W/m2)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Monthly GOES X-ray Statistics", fontsize=14, fontweight="bold")

    # Bottom: Event counts
    ax2.bar(monthly.index, monthly["c_count"], width=25, color="#FF9800", alpha=0.7, label="C-class")
    ax2.bar(monthly.index, monthly["m_count"], width=25, color="#E91E63", alpha=0.7, label="M-class")
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Event Count", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "goes_monthly_stats.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    print("Loading GOES data...")
    goes = load_goes()
    print(f"Loaded {len(goes):,} records\n")

    print("Generating plots...")
    plot_full_timeline(goes)
    plot_flare_events(goes, year=2003)
    plot_flare_events(goes, year=2014)  # Near solar max
    plot_flare_distribution(goes)
    plot_monthly_event_rate(goes)

    print(f"\nAll plots saved to: {FIGURES_DIR}")
