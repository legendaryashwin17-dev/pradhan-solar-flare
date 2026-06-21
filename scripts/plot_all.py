"""
Generate All Comparison and Analysis Plots
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
import matplotlib.ticker as mticker


def load_multi_config():
    with open("results/multi_config_results.json") as f:
        return json.load(f)

def load_best_model():
    with open("results/best_model_results.json") as f:
        return json.load(f)


def plot_tss_comparison(results, save_path=None):
    """Bar chart comparing TSS across all configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r["label"] for r in results]
    tss_vals = [r["metrics"]["tss"] for r in results]

    # Sort by TSS
    sorted_data = sorted(zip(labels, tss_vals), key=lambda x: x[1], reverse=True)
    labels, tss_vals = zip(*sorted_data)

    colors = ["#4CAF50" if t >= 0.65 else "#FF9800" if t >= 0.50 else "#F44336" for t in tss_vals]

    bars = ax.barh(range(len(labels)), tss_vals, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("TSS (True Skill Statistic)", fontsize=12)
    ax.set_title("PRADHAN — TSS Comparison Across Configurations", fontsize=14, fontweight="bold")
    ax.axvline(x=0.65, color="#2196F3", linestyle="--", linewidth=1.5, label="ISRO Target (0.65)")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, tss_vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "tss_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_metrics_radar(results, save_path=None):
    """Radar chart of top 3 configs."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    metrics_keys = ["tss", "hss", "auc", "pod", "csi"]
    metrics_labels = ["TSS", "HSS", "AUC", "POD", "CSI"]

    # Top 3 configs
    top3 = sorted(results, key=lambda x: x["metrics"]["tss"], reverse=True)[:3]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    angles = np.linspace(0, 2 * np.pi, len(metrics_keys), endpoint=False).tolist()
    angles += angles[:1]

    for idx, r in enumerate(top3):
        values = [r["metrics"][k] for k in metrics_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx], label=r["label"])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Top 3 Configs — Metrics Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "metrics_radar.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_event_rate_vs_tss(results, save_path=None):
    """Scatter: event rate vs TSS."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        er = r["event_rate"] * 100
        tss = r["metrics"]["tss"]
        auc = r["metrics"]["auc"]
        color = "#4CAF50" if tss >= 0.65 else "#FF9800" if tss >= 0.50 else "#F44336"
        size = auc * 300
        ax.scatter(er, tss, s=size, c=color, alpha=0.8, edgecolors="white", linewidth=1.5)
        ax.annotate(r["label"], (er, tss), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    ax.axhline(y=0.65, color="#2196F3", linestyle="--", linewidth=1, label="ISRO Target")
    ax.set_xlabel("Event Rate (%)", fontsize=12)
    ax.set_ylabel("TSS", fontsize=12)
    ax.set_title("Event Rate vs TSS (bubble size = AUC)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.9)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "event_rate_vs_tss.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_importance_best(best, save_path=None):
    """Feature importance from best model."""
    fig, ax = plt.subplots(figsize=(10, 8))

    imp = best["feature_importance"]
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    y_pos = np.arange(len(names))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.9, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — Best Model ({best['config']})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "best_feature_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_model_comparison_table(results, best, save_path=None):
    """Summary table as image."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    headers = ["Config", "TSS", "HSS", "AUC", "POD", "POFD", "CSI", "Brier", "Event%"]
    cell_data = []
    row_colors = []

    for r in sorted(results, key=lambda x: x["metrics"]["tss"], reverse=True):
        m = r["metrics"]
        row = [
            r["label"],
            f"{m['tss']:.4f}",
            f"{m['hss']:.4f}",
            f"{m['auc']:.4f}",
            f"{m['pod']:.4f}",
            f"{m['pofd']:.4f}",
            f"{m['csi']:.4f}",
            f"{m['brier']:.4f}",
            f"{r['event_rate']:.1%}",
        ]
        cell_data.append(row)

        if r["label"] == best["config"]:
            row_colors.append(["#C8E6C9"] * len(headers))
        elif m["tss"] >= 0.65:
            row_colors.append(["#E8F5E9"] * len(headers))
        else:
            row_colors.append(["#FFFFFF"] * len(headers))

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellColours=row_colors,
        colColours=["#1565C0"] * len(headers),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("PRADHAN — All Training Configurations", fontsize=14,
                 fontweight="bold", pad=20)

    plt.tight_layout()
    save_path = save_path or FIGURES_DIR / "config_comparison_table.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    results = load_multi_config()
    best = load_best_model()

    print("Generating all plots...")
    plot_tss_comparison(results)
    plot_metrics_radar(results)
    plot_event_rate_vs_tss(results)
    plot_feature_importance_best(best)
    plot_model_comparison_table(results, best)
    print(f"\nAll plots saved to {FIGURES_DIR}")
