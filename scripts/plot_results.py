"""Generate val_acc_curves.png and val_acc_bars.png from results_v2/.

Usage:
    python -m scripts.plot_results --results_dir results_v2
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt


COLORS = {
    "baseline_rgb":  "#4C72B0",
    "exp_hsv":       "#DD8452",
    "exp_lab":       "#55A868",
    "exp_grayscale": "#C44E52",
}

LABELS = {
    "baseline_rgb":  "RGB",
    "exp_hsv":       "HSV",
    "exp_lab":       "LAB",
    "exp_grayscale": "Grayscale",
}


def load_metrics(exp_dir):
    csv_path = os.path.join(exp_dir, "metrics.csv")
    epochs, val_accs = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            val_accs.append(float(row["val_acc"]))
    return epochs, val_accs


def load_summary(exp_dir):
    with open(os.path.join(exp_dir, "summary.json")) as f:
        return json.load(f)


def plot_curves(experiments, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for exp_name, exp_dir in experiments.items():
        epochs, val_accs = load_metrics(exp_dir)
        ax.plot(epochs, val_accs,
                label=LABELS.get(exp_name, exp_name),
                color=COLORS.get(exp_name, None),
                linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy per Epoch — CIFAR-10 (ResNet-18)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_bars(experiments, out_path):
    names, accs, colors = [], [], []

    for exp_name, exp_dir in experiments.items():
        summary = load_summary(exp_dir)
        names.append(LABELS.get(exp_name, exp_name))
        accs.append(summary["best_val_acc"])
        colors.append(COLORS.get(exp_name, "#999999"))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Best Val Accuracy by Colorspace — CIFAR-10 (ResNet-18)")
    ax.set_ylim(0.85, 0.95)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_v2")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    order = ["baseline_rgb", "exp_hsv", "exp_lab", "exp_grayscale"]
    experiments = {}
    for name in order:
        path = os.path.join(args.results_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "metrics.csv")):
            experiments[name] = path
        else:
            print(f"Skipping {name} (not found in {args.results_dir})")

    if not experiments:
        print("No experiment results found.")
        return

    plot_curves(experiments, os.path.join(args.out_dir, "val_acc_curves.png"))
    plot_bars(experiments,   os.path.join(args.out_dir, "val_acc_bars.png"))


if __name__ == "__main__":
    main()
