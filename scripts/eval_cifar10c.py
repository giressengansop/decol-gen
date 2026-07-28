"""Evaluate trained ResNet-18 checkpoints on CIFAR-10-C corruptions.

For each of the 4 colorspace models (RGB, LAB, HSV, Grayscale), evaluates
all 15 corruption types × 5 severity levels without any re-training.

Scientific question: does LAB resist luminosity corruptions (brightness,
contrast, fog) better because L is decoupled from the a,b channels?

Usage:
    python -m scripts.eval_cifar10c \
        --cifar10c_root data/CIFAR-10-C \
        --results_dir  results_v3 \
        --output_dir   results_v3/cifar10c
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from color_dg.color_spaces.transforms import get_transforms
from color_dg.models.resnet import create_resnet18

# ── Corruption catalogue ────────────────────────────────────────────────────
# Standard 15 + extras present in the Zenodo distribution
CORRUPTIONS = [
    # Noise
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    # Blur
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    # Weather
    "snow", "frost", "fog", "brightness", "spatter",
    # Digital
    "contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate",
]

# Luminosity-sensitive corruptions — the core scientific hypothesis
LUMINOSITY_CORRUPTIONS = {"brightness", "fog", "contrast"}

SEVERITIES = [1, 2, 3, 4, 5]

# Experiments: name → (colorspace, model directory)
EXPERIMENTS = {
    "baseline_rgb":  ("rgb",       "baseline_rgb"),
    "exp_lab":       ("lab",       "exp_lab"),
    "exp_hsv":       ("hsv",       "exp_hsv"),
    "exp_grayscale": ("grayscale", "exp_grayscale"),
}


# ── Dataset ─────────────────────────────────────────────────────────────────
class CIFAR10C(Dataset):
    """Wraps a single (corruption, severity) slice of CIFAR-10-C.

    CIFAR-10-C stores 50 000 images per corruption file ordered by severity:
      severity 1 → indices [0,     10 000)
      severity 2 → indices [10000, 20 000)
      ...
      severity 5 → indices [40000, 50 000)
    """

    def __init__(self, data_root: str, corruption: str, severity: int, transform=None):
        assert 1 <= severity <= 5, f"severity must be 1–5, got {severity}"

        data_path   = os.path.join(data_root, f"{corruption}.npy")
        labels_path = os.path.join(data_root, "labels.npy")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Missing: {data_path}")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Missing: {labels_path}")

        all_data   = np.load(data_path)          # (50000, 32, 32, 3) uint8
        all_labels = np.load(labels_path)        # (50000,)

        start = (severity - 1) * 10_000
        end   = severity * 10_000

        self.data      = all_data[start:end]     # (10000, 32, 32, 3) uint8
        self.targets   = all_labels[start:end].astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img    = self.data[idx]           # (32, 32, 3) uint8 numpy
        target = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ── Evaluation helpers ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total   += targets.size(0)
    return correct / total


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = create_resnet18(num_classes=10, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate models on CIFAR-10-C")
    parser.add_argument("--cifar10c_root", default="data/CIFAR-10-C",
                        help="Path to extracted CIFAR-10-C directory")
    parser.add_argument("--results_dir",   default="results_v3",
                        help="Root of results_v3/ with best_model.pth files")
    parser.add_argument("--output_dir",    default="results_v3/cifar10c",
                        help="Where to write CSV/JSON outputs")
    parser.add_argument("--batch_size",    type=int, default=256)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--seed",          type=int, default=None,
                        help="Evaluate the <exp>_seed<N> checkpoints instead of <exp>")
    parser.add_argument("--normalization", default="zscore",
                        choices=["zscore", "minmax", "centered"],
                        help="Normalization scheme matching how the checkpoint was trained (Phase 5)")
    parser.add_argument("--subdir_suffix", default="",
                        help="Appended to each experiment's checkpoint subdir before _seed<N>, "
                             "e.g. '_norm_minmax' for Phase 5 checkpoints (baseline_rgb_norm_minmax_seed0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CIFAR-10-C root: {args.cifar10c_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    raw_csv_path     = os.path.join(args.output_dir, "results_raw.csv")
    summary_csv_path = os.path.join(args.output_dir, "corruption_summary.csv")
    summary_json     = os.path.join(args.output_dir, "summary.json")

    raw_rows     = []
    all_results  = {}   # {exp_name: {corruption: {severity: acc}}}

    for exp_name, (colorspace, subdir) in EXPERIMENTS.items():
        subdir = f"{subdir}{args.subdir_suffix}"
        if args.seed is not None:
            subdir = f"{subdir}_seed{args.seed}"
        checkpoint = os.path.join(args.results_dir, subdir, "best_model.pth")
        if not os.path.isfile(checkpoint):
            print(f"[SKIP] {exp_name}: checkpoint not found at {checkpoint}")
            continue

        print(f"\n{'='*60}")
        print(f"Experiment : {exp_name}  (colorspace={colorspace})")
        print(f"Checkpoint : {checkpoint}")

        model     = load_model(checkpoint, device)
        transform = get_transforms(colorspace, train=False, normalization=args.normalization)
        all_results[exp_name] = {}

        for corruption in CORRUPTIONS:
            all_results[exp_name][corruption] = {}
            accs = []

            for severity in SEVERITIES:
                try:
                    dataset = CIFAR10C(
                        args.cifar10c_root, corruption, severity, transform
                    )
                except FileNotFoundError as e:
                    print(f"  [WARN] {e}")
                    continue

                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )

                acc = evaluate(model, loader, device)
                accs.append(acc)
                all_results[exp_name][corruption][severity] = acc

                raw_rows.append({
                    "experiment": exp_name,
                    "colorspace": colorspace,
                    "corruption": corruption,
                    "severity":   severity,
                    "accuracy":   acc,
                    "luminosity": corruption in LUMINOSITY_CORRUPTIONS,
                })

                print(
                    f"  {corruption:<22} sev={severity}  acc={acc:.4f}",
                    flush=True,
                )

            if accs:
                mean_acc = np.mean(accs)
                tag = " [LUMINOSITY]" if corruption in LUMINOSITY_CORRUPTIONS else ""
                print(f"  {'':22} mean={mean_acc:.4f}{tag}")

    # ── Write raw CSV ────────────────────────────────────────────────────────
    if raw_rows:
        with open(raw_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"\nRaw results saved  → {raw_csv_path}")

    # ── Per-(experiment, corruption) mean accuracy ───────────────────────────
    summary_rows = []
    for exp_name, corruptions in all_results.items():
        colorspace = EXPERIMENTS[exp_name][0]
        for corruption, sevs in corruptions.items():
            if not sevs:
                continue
            mean_acc = float(np.mean(list(sevs.values())))
            summary_rows.append({
                "experiment":  exp_name,
                "colorspace":  colorspace,
                "corruption":  corruption,
                "luminosity":  corruption in LUMINOSITY_CORRUPTIONS,
                "mean_acc":    mean_acc,
            })

    if summary_rows:
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Corruption summary → {summary_csv_path}")

    # ── Global mCA + luminosity mCA ──────────────────────────────────────────
    final_summary = {}
    for exp_name, corruptions in all_results.items():
        all_accs  = []
        lum_accs  = []
        nolum_accs = []
        for corruption, sevs in corruptions.items():
            if not sevs:
                continue
            mean_acc = float(np.mean(list(sevs.values())))
            all_accs.append(mean_acc)
            if corruption in LUMINOSITY_CORRUPTIONS:
                lum_accs.append(mean_acc)
            else:
                nolum_accs.append(mean_acc)

        colorspace = EXPERIMENTS[exp_name][0]
        final_summary[exp_name] = {
            "colorspace":       colorspace,
            "normalization":    args.normalization,
            "seed":             args.seed,
            "mCA":              float(np.mean(all_accs))    if all_accs   else None,
            "mCA_luminosity":   float(np.mean(lum_accs))   if lum_accs   else None,
            "mCA_non_luminosity": float(np.mean(nolum_accs)) if nolum_accs else None,
        }

    with open(summary_json, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Global summary     → {summary_json}")

    # ── Print final table ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Experiment':<20} {'mCA':>6}  {'mCA_lum':>8}  {'mCA_nolum':>10}")
    print("-"*60)
    for exp_name, stats in final_summary.items():
        mca     = stats["mCA"]
        mca_l   = stats["mCA_luminosity"]
        mca_nl  = stats["mCA_non_luminosity"]
        print(
            f"{exp_name:<20} "
            f"{mca:.4f}  {mca_l:.4f}    {mca_nl:.4f}"
        )
    print("="*60)
    print("Done.")


if __name__ == "__main__":
    main()
