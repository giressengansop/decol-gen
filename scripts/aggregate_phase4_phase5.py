"""Aggregate Phase 4 (multi-seed, z-score) + Phase 5 (normalization ablation)
results into a single mean +/- std table, grouped by (colorspace, normalization).

Phase 4 checkpoints (results_v4/<exp>_seed<N>/) have no "normalization" key in
their summary.json (they predate that field) -> implicitly "zscore".
Phase 5 checkpoints (results_v5/<exp>_norm_<norm>_seed<N>/) have it explicitly.

Usage:
    python -m scripts.aggregate_phase4_phase5
"""

import glob
import json
import os

import numpy as np

# (results_dir, training_glob, cifar10c_glob) per (colorspace, normalization) group.
GROUPS = [
    ("rgb",       "zscore",   "results_v4/baseline_rgb_seed*",              "results_v4/cifar10c_seed*"),
    ("lab",       "zscore",   "results_v4/exp_lab_seed*",                   "results_v4/cifar10c_seed*"),
    ("hsv",       "zscore",   "results_v4/exp_hsv_seed*",                   "results_v4/cifar10c_seed*"),
    ("grayscale", "zscore",   "results_v4/exp_grayscale_seed*",             "results_v4/cifar10c_seed*"),
    ("rgb",       "minmax",   "results_v5/baseline_rgb_norm_minmax_seed*",  "results_v5/cifar10c_norm_minmax_seed*"),
    ("lab",       "minmax",   "results_v5/exp_lab_norm_minmax_seed*",       "results_v5/cifar10c_norm_minmax_seed*"),
    ("rgb",       "centered", "results_v5/baseline_rgb_norm_centered_seed*","results_v5/cifar10c_norm_centered_seed*"),
    ("lab",       "centered", "results_v5/exp_lab_norm_centered_seed*",     "results_v5/cifar10c_norm_centered_seed*"),
]

# Key used inside each cifar10c summary.json to pick the right experiment block.
CIFAR10C_KEY = {"rgb": "baseline_rgb", "lab": "exp_lab", "hsv": "exp_hsv", "grayscale": "exp_grayscale"}


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def load_training(pattern):
    """seed -> best_val_acc, reading every <pattern>/summary.json."""
    out = {}
    for path in sorted(glob.glob(os.path.join(pattern, "summary.json"))):
        with open(path) as f:
            d = json.load(f)
        out[d["seed"]] = d["best_val_acc"]
    return out


def load_cifar10c(pattern, exp_key):
    """seed -> (mCA, mCA_luminosity, mCA_non_luminosity), reading every <pattern>/summary.json."""
    out = {}
    for path in sorted(glob.glob(os.path.join(pattern, "summary.json"))):
        with open(path) as f:
            d = json.load(f)
        if exp_key not in d:
            continue
        entry = d[exp_key]
        out[entry["seed"]] = (entry["mCA"], entry["mCA_luminosity"], entry["mCA_non_luminosity"])
    return out


def main():
    os.makedirs("results_aggregate", exist_ok=True)
    rows = []

    for colorspace, normalization, train_pattern, cifar10c_pattern in GROUPS:
        train = load_training(train_pattern)
        cifar10c = load_cifar10c(cifar10c_pattern, CIFAR10C_KEY[colorspace])

        val_accs = list(train.values())
        mcas = [v[0] for v in cifar10c.values()]
        lumis = [v[1] for v in cifar10c.values()]
        nonlumis = [v[2] for v in cifar10c.values()]

        val_mean, val_std = mean_std(val_accs) if val_accs else (None, None)
        mca_mean, mca_std = mean_std(mcas) if mcas else (None, None)
        lumi_mean, lumi_std = mean_std(lumis) if lumis else (None, None)
        nonlumi_mean, nonlumi_std = mean_std(nonlumis) if nonlumis else (None, None)

        rows.append({
            "colorspace": colorspace,
            "normalization": normalization,
            "n_train_seeds": len(train),
            "train_seeds": sorted(train.keys()),
            "val_acc_mean": val_mean, "val_acc_std": val_std,
            "n_cifar10c_seeds": len(cifar10c),
            "cifar10c_seeds": sorted(cifar10c.keys()),
            "mCA_mean": mca_mean, "mCA_std": mca_std,
            "mCA_luminosity_mean": lumi_mean, "mCA_luminosity_std": lumi_std,
            "mCA_non_luminosity_mean": nonlumi_mean, "mCA_non_luminosity_std": nonlumi_std,
        })

    print("=" * 112)
    print("PHASE 4 + PHASE 5 — combined multi-seed aggregation (mean +/- std)")
    print("=" * 112)
    header = (
        f"{'colorspace':<12}{'norm':<10}{'n_tr':>5}{'val_acc':>16}"
        f"{'n_c10c':>7}{'mCA':>16}{'mCA_lumi':>16}{'mCA_nonlumi':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        va = f"{r['val_acc_mean']:.4f}+-{r['val_acc_std']:.4f}" if r["val_acc_mean"] is not None else "n/a"
        mca = f"{r['mCA_mean']:.4f}+-{r['mCA_std']:.4f}" if r["mCA_mean"] is not None else "n/a"
        lum = f"{r['mCA_luminosity_mean']:.4f}+-{r['mCA_luminosity_std']:.4f}" if r["mCA_luminosity_mean"] is not None else "n/a"
        nlum = f"{r['mCA_non_luminosity_mean']:.4f}+-{r['mCA_non_luminosity_std']:.4f}" if r["mCA_non_luminosity_mean"] is not None else "n/a"
        print(f"{r['colorspace']:<12}{r['normalization']:<10}{r['n_train_seeds']:>5}{va:>16}{r['n_cifar10c_seeds']:>7}{mca:>16}{lum:>16}{nlum:>16}")
    print("=" * 112)

    # Key comparisons: RGB vs LAB on mCA_luminosity, per normalization.
    print("\nRGB vs LAB on mCA_luminosity, per normalization (mean+-std interval overlap == not distinguishable):")
    for normalization in ["zscore", "minmax", "centered"]:
        rgb = next((r for r in rows if r["colorspace"] == "rgb" and r["normalization"] == normalization), None)
        lab = next((r for r in rows if r["colorspace"] == "lab" and r["normalization"] == normalization), None)
        if not rgb or not lab or rgb["mCA_luminosity_mean"] is None or lab["mCA_luminosity_mean"] is None:
            continue
        rgb_lo, rgb_hi = rgb["mCA_luminosity_mean"] - rgb["mCA_luminosity_std"], rgb["mCA_luminosity_mean"] + rgb["mCA_luminosity_std"]
        lab_lo, lab_hi = lab["mCA_luminosity_mean"] - lab["mCA_luminosity_std"], lab["mCA_luminosity_mean"] + lab["mCA_luminosity_std"]
        overlap = not (rgb_hi < lab_lo or lab_hi < rgb_lo)
        winner = "RGB" if rgb["mCA_luminosity_mean"] > lab["mCA_luminosity_mean"] else "LAB"
        print(
            f"  {normalization:<10} RGB={rgb['mCA_luminosity_mean']:.4f}+-{rgb['mCA_luminosity_std']:.4f} (n={rgb['n_cifar10c_seeds']})  "
            f"LAB={lab['mCA_luminosity_mean']:.4f}+-{lab['mCA_luminosity_std']:.4f} (n={lab['n_cifar10c_seeds']})  "
            f"-> {winner} ahead, intervals {'OVERLAP (not distinguishable)' if overlap else 'DO NOT OVERLAP (distinguishable)'}"
        )

    out_path = "results_aggregate/phase4_phase5_summary.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
