#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse der gelernten Eingangsfilter — architekturuebergreifend.

Liest conv1 direkt aus dem state_dict. ResNet-18 fuehrt die Schicht unter
"conv1.weight", das hier verwendete VGG-11 unter "features.0.weight"; beide
haben die Form [64, 3, 3, 3], sodass dieselbe Auswertung gilt.

Zusaetzlich zum ueber die Startwerte gemittelten zweiten Moment wird der
effektive Rang JE STARTWERT berechnet, sodass Vergleiche zwischen den
Farbraeumen gepaart getestet werden koennen.
"""
import argparse, os
import numpy as np, torch
from scipy import stats

KEYS = ["conv1.weight", "features.0.weight"]
DIRS = {"rgb": "baseline_rgb", "lab": "exp_lab",
        "hsv": "exp_hsv", "grayscale": "exp_grayscale"}
LUMDIR = np.ones(3) / np.sqrt(3)


def second_moment(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    key = next((k for k in KEYS if k in sd), None)
    if key is None:
        raise KeyError(f"conv1 nicht gefunden in {path}: {list(sd)[:5]} ...")
    W = sd[key].numpy()                          # [64, 3, 3, 3]
    X = W.transpose(0, 2, 3, 1).reshape(-1, 3)   # 576 Kanalvektoren
    return X.T @ X / len(X)


def decompose(M):
    w, V = np.linalg.eigh(M)
    o = np.argsort(w)[::-1]
    lam = w[o] / w.sum()
    axes = V[:, o].T
    axes = np.array([a * np.sign(a[np.argmax(np.abs(a))]) for a in axes])
    return lam, axes, 1.0 / np.sum(lam ** 2)


ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
ap.add_argument("--label", default="")
ap.add_argument("--suffix", default="")
a = ap.parse_args()

print(f"\n{'='*74}\n  GELERNTE FARBRICHTUNGEN IN conv1 — {a.label or a.results_dir}\n{'='*74}")
print(f"{'Quelle':12s} {'l1':>7s} {'l2':>7s} {'l3':>7s}   {'Rang(oM)':>9s}   {'Rang je Seed':>16s}")

ranks, axes_store = {}, {}
for cs, d in DIRS.items():
    Ms = []
    for s in a.seeds:
        p = os.path.join(a.results_dir, f"{d}{a.suffix}_seed{s}", "best_model.pth")
        if os.path.isfile(p):
            Ms.append(second_moment(p))
    if not Ms:
        continue
    lam, axes, r = decompose(np.mean(Ms, axis=0))
    per = np.array([decompose(M)[2] for M in Ms])
    ranks[cs], axes_store[cs] = per, axes
    print(f"{cs:12s} {lam[0]:7.3f} {lam[1]:7.3f} {lam[2]:7.3f}   "
          f"{r:9.2f}   {per.mean():7.2f} +/- {per.std(ddof=1):.2f}  (n={len(per)})")

if "rgb" in axes_store:
    print(f"\nAchsen des RGB-Netzes (Vorzeichen normiert):")
    lam, axes, _ = decompose(np.mean(
        [second_moment(os.path.join(a.results_dir, f"baseline_rgb{a.suffix}_seed{s}", "best_model.pth"))
         for s in a.seeds if os.path.isfile(
             os.path.join(a.results_dir, f"baseline_rgb{a.suffix}_seed{s}", "best_model.pth"))], axis=0))
    for i, (v, ax) in enumerate(zip(lam, axes)):
        print(f"  Achse {i+1}  [{ax[0]:+.3f} {ax[1]:+.3f} {ax[2]:+.3f}]  "
              f"l={v:.3f}   |cos| zu (1,1,1) = {abs(ax @ LUMDIR):.3f}")

print("\nGepaarte Tests des effektiven Rangs (je Startwert):")
for x, y in [("lab", "rgb"), ("hsv", "rgb"), ("rgb", "grayscale")]:
    if x in ranks and y in ranks and len(ranks[x]) == len(ranks[y]):
        d = ranks[x] - ranks[y]
        t, p = stats.ttest_rel(ranks[x], ranks[y])
        print(f"  {x:9s} - {y:9s} {d.mean():+6.3f}   p={p:.4f}   {sum(d>0)}/{len(d)}")
