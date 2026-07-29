#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Ce script utilise exactement le même convertisseur que l'entraînement, donc les
constantes produites sont cohérentes par construction.

Usage :
    cd ~/decol-gen
    python compute_cifar10_stats.py --data_root ./data

Sortie : un bloc CIFAR10_STATS prêt à coller dans transforms.py.
"""

import argparse
import sys

import numpy as np

try:
    from torchvision import datasets
except ImportError:
    sys.exit("torchvision requis. Active l'environnement : conda activate color_dg")

try:
    from color_dg.color_spaces.converter import ColorspaceConverter
except ImportError:
    sys.exit("Lance ce script depuis la racine du projet (~/decol-gen), "
             "avec le package installé (pip install -e .)")


SPACES = ["rgb", "hsv", "lab", "grayscale"]


def convert(conv, img_uint8, space):
    """Reproduit exactement ColorspaceTransform.__call__ (sans le tenseur)."""
    if space == "rgb":
        return img_uint8.astype(np.float32) / 255.0
    if space == "hsv":
        return conv.rgb_to_hsv(img_uint8)
    if space == "lab":
        return conv.rgb_to_lab(img_uint8)
    if space == "grayscale":
        return conv.rgb_to_grayscale(img_uint8)
    raise ValueError(space)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--limit", type=int, default=None,
                    help="N'utiliser que les N premières images (pour un test rapide)")
    args = ap.parse_args()

    ds = datasets.CIFAR10(root=args.data_root, train=True, download=True)
    data = ds.data  # (50000, 32, 32, 3) uint8, ordre RGB
    if args.limit:
        data = data[:args.limit]
    n = len(data)
    print(f"CIFAR-10 train : {n} images\n")

    conv = ColorspaceConverter()
    results = {}

    for space in SPACES:
        # Accumulation en somme / somme des carrés pour éviter de tout charger
        s = np.zeros(3, dtype=np.float64)
        s2 = np.zeros(3, dtype=np.float64)
        count = 0

        for i in range(n):
            out = convert(conv, data[i], space)          # (32, 32, 3) float32
            flat = out.reshape(-1, 3).astype(np.float64)
            s += flat.sum(axis=0)
            s2 += (flat ** 2).sum(axis=0)
            count += flat.shape[0]

            if (i + 1) % 10000 == 0:
                print(f"  {space:<10} {i + 1}/{n}", flush=True)

        mean = s / count
        var = s2 / count - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        results[space] = (mean, std)
        print(f"  {space:<10} mean={np.round(mean, 4)}  std={np.round(std, 4)}\n")

    # ── Bloc prêt à coller ──────────────────────────────────────────────────
    print("=" * 72)
    print("À coller dans src/color_dg/color_spaces/transforms.py :")
    print("=" * 72)
    print("CIFAR10_STATS = {")
    for space in SPACES:
        mean, std = results[space]
        m = ", ".join(f"{v:.4f}" for v in mean)
        d = ", ".join(f"{v:.4f}" for v in std)
        print(f'    "{space}":{" " * (12 - len(space))}'
              f'{{"mean": [{m}], "std": [{d}]}},')
    print("}")
    print("=" * 72)

    # ── Contrôle de cohérence ───────────────────────────────────────────────
    print("\nContrôle de cohérence")
    print("-" * 72)
    lab_mean = results["lab"][0]
    ok = abs(lab_mean[1] - 0.5) < 0.05 and abs(lab_mean[2] - 0.5) < 0.05
    print(f"  LAB, canaux a et b centrés sur ~0.5 : "
          f"a={lab_mean[1]:.4f}  b={lab_mean[2]:.4f}  → {'OK' if ok else 'ANORMAL'}")
    if not ok:
        print("    (attendu ~0.50 : a et b valent 0 pour une couleur neutre,")
        print("     et (0 + 128) / 255 = 0.502)")

    old = [0.3277, 0.3277, 0.3277]
    print(f"\n  Anciennes valeurs LAB dans le code : {old}")
    print(f"  Nouvelles valeurs                  : {np.round(lab_mean, 4).tolist()}")
    shift = np.abs(np.array(old) - lab_mean)
    print(f"  Écart par canal                    : {np.round(shift, 4).tolist()}")
    print("\n  → Après correction, relance une graine de contrôle :")
    print("     python -m scripts.train_model --config configs/exp_lab.yaml \\")
    print("            --seed 0 --output_dir results_v6/exp_lab_fixedstats_seed0")
    print("     Si l'accuracy reste ~92.9 %, les conclusions sont inchangées.")


if __name__ == "__main__":
    main()
