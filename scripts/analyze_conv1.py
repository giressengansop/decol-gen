#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse der gelernten Eingangsfilter (conv1).

Prüft die Kernaussage der Diskussion: „Das Netz lernt seine eigene
Farbentkopplung." conv1 hat die Form [64, 3, 3, 3]; die zweite Achse ist die
Eingangsfarbachse. Über alle Filter und Kernpositionen hinweg ergibt das
zweite Moment dieser 3-Vektoren die dominanten Farbrichtungen, die das Netz
tatsächlich nutzt.

Kontrolle: Das Graustufen-Netz erhält drei identische Kanäle und kann daher
nur eine Farbrichtung nutzen. Zeigt die Analyse das nicht, ist sie falsch.

Usage:  python -m scripts.analyze_conv1
"""

import os
import numpy as np
import torch

SEEDS = [0, 1, 2, 3, 42]
CONDS = {
    "BatchNorm": {"rgb": ("results_v6", "baseline_rgb", ""),
                  "lab": ("results_v6", "exp_lab", ""),
                  "hsv": ("results_v6", "exp_hsv", ""),
                  "grayscale": ("results_v4", "exp_grayscale", "")},
    "GroupNorm": {"rgb": ("results_v7", "baseline_rgb", "_gn"),
                  "lab": ("results_v7", "exp_lab", "_gn"),
                  "hsv": ("results_v7", "exp_hsv", "_gn"),
                  "grayscale": ("results_v7", "exp_grayscale", "_gn")},
}

# Referenzachsen im RGB-Raum
AXES = {
    "Luminanz  (1,1,1)":      np.array([1, 1, 1]) / np.sqrt(3),
    "Rot–Grün  (1,-1,0)":     np.array([1, -1, 0]) / np.sqrt(2),
    "Blau–Gelb (1,1,-2)":     np.array([1, 1, -2]) / np.sqrt(6),
}


def color_moment(path):
    """Zweites Moment der Kanal-Gewichtsvektoren von conv1.

    conv1.weight : [out=64, in=3, kh=3, kw=3]
    -> [64*3*3, 3] : ein 3-Vektor je (Filter, Kernposition)
    -> M = X^T X / N  (3x3, vorzeichen-invariant)
    """
    sd = torch.load(path, map_location="cpu")
    W = sd["conv1.weight"].numpy()              # [64, 3, 3, 3]
    X = W.transpose(0, 2, 3, 1).reshape(-1, 3)  # [576, 3]
    return X.T @ X / len(X)


def analyse(M):
    ev, evec = np.linalg.eigh(M)          # aufsteigend
    ev, evec = ev[::-1], evec[:, ::-1]    # absteigend
    return ev / ev.sum(), evec            # Anteile, Achsen als Spalten


print("=" * 76)
print("  GELERNTE FARBRICHTUNGEN IN conv1")
print("=" * 76)

store = {}
for cond, mapping in CONDS.items():
    print(f"\n### {cond}")
    print(f"  {'Farbraum':<12}{'λ1':>8}{'λ2':>8}{'λ3':>8}   "
          f"{'genutzte Farbdimensionen':<26}")
    print("  " + "-" * 70)
    for cs, (root, exp, suf) in mapping.items():
        Ms, ok = [], 0
        for s in SEEDS:
            p = f"{root}/{exp}{suf}_seed{s}/best_model.pth"
            if os.path.isfile(p):
                Ms.append(color_moment(p)); ok += 1
        if not Ms:
            print(f"  {cs:<12}  keine Checkpoints gefunden"); continue
        M = np.mean(Ms, axis=0)
        ev, evec = analyse(M)
        store[(cond, cs)] = (ev, evec, ok)
        # effektive Dimension (Partizipationsverhältnis)
        eff = 1.0 / np.sum(ev ** 2)
        print(f"  {cs:<12}{ev[0]:>8.3f}{ev[1]:>8.3f}{ev[2]:>8.3f}   "
              f"eff. Rang = {eff:.2f}  ({ok} Seeds)")

# ── Ausrichtung der Hauptachse (nur RGB und Graustufen sind im RGB-Raum) ──
print("\n\n### Ausrichtung der gelernten Achsen (nur RGB-Eingabe interpretierbar)")
print("=" * 76)
for cond in CONDS:
    for cs in ["rgb", "grayscale"]:
        if (cond, cs) not in store:
            continue
        ev, evec, _ = store[(cond, cs)]
        print(f"\n  {cond} · {cs}")
        for k in range(3):
            v = evec[:, k]
            best = max(AXES.items(), key=lambda kv: abs(v @ kv[1]))
            cos = abs(v @ best[1])
            comp = "  ".join(f"{x:+.2f}" for x in v)
            print(f"    Achse {k+1} (λ={ev[k]:.3f}) : [{comp}]   "
                  f"→ {best[0]:<20} |cos| = {cos:.3f}")

# ── Referenz: Hauptachsen der CIFAR-10-Pixelfarben ───────────────────────
print("\n\n### Referenz — Hauptachsen der Bilddaten selbst")
print("=" * 76)
try:
    from torchvision import datasets
    ds = datasets.CIFAR10(root="./data", train=True, download=False)
    px = (ds.data.reshape(-1, 3).astype(np.float64) / 255.0)
    px -= px.mean(0)
    ev, evec = np.linalg.eigh(px.T @ px / len(px))
    ev, evec = ev[::-1] / ev.sum(), evec[:, ::-1]
    for k in range(3):
        v = evec[:, k]
        best = max(AXES.items(), key=lambda kv: abs(v @ kv[1]))
        comp = "  ".join(f"{x:+.2f}" for x in v)
        print(f"  Datenachse {k+1} (λ={ev[k]:.3f}) : [{comp}]   "
              f"→ {best[0]:<20} |cos| = {abs(v @ best[1]):.3f}")
except Exception as e:
    print(f"  (übersprungen: {e})")

print("\n" + "=" * 76)
print("""  LESEHILFE

  λ1, λ2, λ3 : Anteil jeder Farbrichtung an der gesamten Kanalgewichtung.
  Der effektive Rang sagt, wie viele Farbdimensionen das Netz wirklich nutzt:
  3,00 = alle drei gleichermaßen, 1,00 = nur eine einzige.

  KONTROLLE: Das Graustufen-Netz bekommt drei identische Kanäle. Sein
  effektiver Rang muss deutlich unter dem der Farbnetze liegen. Ist das
  nicht der Fall, misst die Analyse nicht das, was sie messen soll.

  Nur bei RGB-Eingabe sind die Achsen direkt als Luminanz- bzw.
  Gegenfarbachsen lesbar. Bei LAB und HSV liegen die Achsen im jeweiligen
  Eingaberaum; dort ist der EFFEKTIVE RANG die vergleichbare Größe.
""")
