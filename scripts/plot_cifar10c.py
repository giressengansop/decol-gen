"""Visualize CIFAR-10-C robustness results.

Reads results_v3/cifar10c/ outputs and produces:
  1. Heatmap: accuracy per (model × corruption), averaged over severities
  2. Bar chart: mean Corruption Accuracy (mCA) per model
  3. Luminosity analysis: LAB vs RGB on brightness/fog/contrast
  4. Severity curves: accuracy vs severity for luminosity corruptions

Usage:
    python -m scripts.plot_cifar10c \
        --results_dir results_v3/cifar10c \
        --output_dir  results_v3/cifar10c
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

LUMINOSITY_CORRUPTIONS = {"brightness", "fog", "contrast"}

CORRUPTION_GROUPS = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "Weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"],
}

COLORSPACE_COLORS = {
    "baseline_rgb":  "#4C72B0",
    "exp_lab":       "#DD8452",
    "exp_hsv":       "#55A868",
    "exp_grayscale": "#C44E52",
}

DISPLAY_NAMES = {
    "baseline_rgb":  "RGB",
    "exp_lab":       "LAB",
    "exp_hsv":       "HSV",
    "exp_grayscale": "Grayscale",
}


def load_data(results_dir: str):
    raw_path     = os.path.join(results_dir, "results_raw.csv")
    summary_path = os.path.join(results_dir, "corruption_summary.csv")
    json_path    = os.path.join(results_dir, "summary.json")

    raw      = pd.read_csv(raw_path)
    summary  = pd.read_csv(summary_path)
    with open(json_path) as f:
        global_stats = json.load(f)

    return raw, summary, global_stats


# ── 1. Heatmap accuracy per (model × corruption) ────────────────────────────
def plot_heatmap(summary: pd.DataFrame, output_dir: str):
    pivot = summary.pivot(index="corruption", columns="experiment", values="mean_acc")

    # reorder columns
    col_order = [c for c in ["baseline_rgb", "exp_lab", "exp_hsv", "exp_grayscale"]
                 if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.columns = [DISPLAY_NAMES.get(c, c) for c in pivot.columns]

    # reorder rows following group order
    row_order = [c for g in CORRUPTION_GROUPS.values() for c in g if c in pivot.index]
    pivot = pivot.loc[row_order]

    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=pivot.values.min() - 0.02,
        vmax=pivot.values.max() + 0.02,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )

    # group separators
    group_sizes = [len(v) for v in CORRUPTION_GROUPS.values()]
    cumulative  = 0
    for size in group_sizes[:-1]:
        cumulative += size
        ax.axhline(cumulative, color="black", lw=1.5)

    # group labels on y-axis
    cumulative = 0
    for group_name, members in CORRUPTION_GROUPS.items():
        n     = len(members)
        mid   = cumulative + n / 2
        ax.text(-0.35, mid, group_name, va="center", ha="right",
                fontsize=9, fontweight="bold",
                transform=ax.get_yaxis_transform())
        cumulative += n

    ax.set_xlabel("Color Space", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Accuracy on CIFAR-10-C\n(mean over 5 severities)", fontsize=13, pad=12)
    plt.tight_layout()

    path = os.path.join(output_dir, "heatmap_cifar10c.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 2. mCA bar chart ─────────────────────────────────────────────────────────
def plot_mca_bars(global_stats: dict, output_dir: str):
    experiments = list(global_stats.keys())
    mca         = [global_stats[e]["mCA"]              for e in experiments]
    mca_lum     = [global_stats[e]["mCA_luminosity"]   for e in experiments]
    mca_nolum   = [global_stats[e]["mCA_non_luminosity"] for e in experiments]

    labels = [DISPLAY_NAMES.get(e, e) for e in experiments]
    x      = np.arange(len(experiments))
    width  = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORSPACE_COLORS.get(e, "gray") for e in experiments]

    bars1 = ax.bar(x - width, mca,       width, label="All corruptions",           alpha=0.9)
    bars2 = ax.bar(x,         mca_lum,   width, label="Luminosity (brightness/fog/contrast)", alpha=0.9, hatch="//")
    bars3 = ax.bar(x + width, mca_nolum, width, label="Non-luminosity",             alpha=0.9, hatch="..")

    for bars, vals in [(bars1, mca), (bars2, mca_lum), (bars3, mca_nolum)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Corruption Accuracy (mCA)", fontsize=11)
    ax.set_title("Robustness on CIFAR-10-C\nby color space", fontsize=13, pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylim(max(0, min(mca + mca_lum + mca_nolum) - 0.03), 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "mca_bars_cifar10c.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 3. Luminosity delta: LAB vs RGB ─────────────────────────────────────────
def plot_luminosity_delta(summary: pd.DataFrame, output_dir: str):
    lum_df = summary[summary["luminosity"] == True].copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    x         = np.arange(len(LUMINOSITY_CORRUPTIONS))
    lum_list  = sorted(LUMINOSITY_CORRUPTIONS)
    width     = 0.18
    offsets   = np.linspace(-1.5 * width, 1.5 * width, 4)

    exp_order = ["baseline_rgb", "exp_lab", "exp_hsv", "exp_grayscale"]
    for i, exp_name in enumerate(exp_order):
        sub  = lum_df[lum_df["experiment"] == exp_name]
        vals = [sub.loc[sub["corruption"] == c, "mean_acc"].values[0]
                if len(sub.loc[sub["corruption"] == c]) > 0 else 0
                for c in lum_list]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=DISPLAY_NAMES.get(exp_name, exp_name),
                      color=COLORSPACE_COLORS.get(exp_name, "gray"),
                      alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in lum_list], fontsize=11)
    ax.set_ylabel("Mean Accuracy", fontsize=11)
    ax.set_title("Luminosity Corruptions — LAB vs Others\n"
                 "(Does separating L from a,b help?)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "luminosity_delta_cifar10c.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 4. Severity curves for luminosity corruptions ───────────────────────────
def plot_severity_curves(raw: pd.DataFrame, output_dir: str):
    lum_corrs = sorted(LUMINOSITY_CORRUPTIONS)
    fig, axes = plt.subplots(1, len(lum_corrs), figsize=(14, 4), sharey=False)

    exp_order = ["baseline_rgb", "exp_lab", "exp_hsv", "exp_grayscale"]
    for ax, corruption in zip(axes, lum_corrs):
        sub = raw[raw["corruption"] == corruption]
        for exp_name in exp_order:
            exp_sub = sub[sub["experiment"] == exp_name].sort_values("severity")
            if exp_sub.empty:
                continue
            ax.plot(
                exp_sub["severity"], exp_sub["accuracy"],
                marker="o", linewidth=1.8,
                label=DISPLAY_NAMES.get(exp_name, exp_name),
                color=COLORSPACE_COLORS.get(exp_name, "gray"),
            )
        ax.set_title(corruption.capitalize(), fontsize=11)
        ax.set_xlabel("Severity", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy vs Severity — Luminosity Corruptions", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "severity_curves_cifar10c.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_v3/cifar10c")
    parser.add_argument("--output_dir",  default="results_v3/cifar10c")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw, summary, global_stats = load_data(args.results_dir)

    print("Generating plots …")
    plot_heatmap(summary, args.output_dir)
    plot_mca_bars(global_stats, args.output_dir)
    plot_luminosity_delta(summary, args.output_dir)
    plot_severity_curves(raw, args.output_dir)
    print("All plots saved.")


if __name__ == "__main__":
    main()
