"""
Generate Feature Importance Plot from Training Results
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_feature_importance(save_path=None):
    """Plot feature importance from saved results."""
    results_path = Path("results/training_results.json")

    with open(results_path) as f:
        results = json.load(f)

    importance = results["feature_importance"]
    features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    names = [f[0] for f in features]
    values = [f[1] for f in features]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#F44336", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("PRADHAN Feature Importance (XGBoost)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "feature_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def plot_metrics_summary(save_path=None):
    """Plot training metrics summary."""
    results_path = Path("results/training_results.json")

    with open(results_path) as f:
        results = json.load(f)

    metrics = results["metrics"]
    ablation = results["ablation_raw"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Main metrics
    ax = axes[0]
    main_metrics = {k: metrics[k] for k in ["tss", "hss", "auc", "brier"]}
    names = list(main_metrics.keys())
    values = list(main_metrics.values())
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    bars = ax.bar(names, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Set Metrics", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=10)

    # Right: Ablation comparison
    ax = axes[1]
    x = np.arange(4)
    width = 0.35

    raw_vals = [ablation[k] for k in ["tss", "hss", "auc", "brier"]]
    all_vals = [metrics[k] for k in ["tss", "hss", "auc", "brier"]]

    ax.bar(x - width/2, raw_vals, width, label="Raw Flux", color="#90CAF9", alpha=0.8)
    ax.bar(x + width/2, all_vals, width, label="All Features", color="#F44336", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["TSS", "HSS", "AUC", "Brier"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation: Raw Flux vs All Features", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "metrics_summary.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_feature_importance()
    plot_metrics_summary()
    print(f"\nPlots saved to {FIGURES_DIR}")
