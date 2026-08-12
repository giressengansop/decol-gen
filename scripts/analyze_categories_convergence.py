#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse par catégorie de corruption + dynamique de convergence.

Répond à deux questions du Exposés:
  Phase 2 : « Does a model trained on CIELAB converge faster? »
  Phase 3 : robustesse aux corruptions d'illumination ET de météo

Sources :
  BatchNorm  : results_v6 (rgb/lab/hsv, constantes corrigées)
               + results_v4 (grayscale — dessen Konstanten waren bereits korrekt)
  GroupNorm  : results_v7 (alle vier)

Usage :  python -m scripts.analyze_categories_convergence
"""

import csv, json, os, statistics as st
from collections import defaultdict

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

SEEDS = [0, 1, 2, 3, 42]
CS = ["rgb", "lab", "hsv", "grayscale"]
EXP = {"rgb": "baseline_rgb", "lab": "exp_lab",
       "hsv": "exp_hsv", "grayscale": "exp_grayscale"}

# Taxonomie standard de Hendrycks & Dietterich (2019).
# Les 4 corruptions "extra" du dépôt Zenodo sont marquées d'une *.
CATEGORIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "digital": ["contrast", "elastic_transform", "pixelate",
                "jpeg_compression", "saturate"],
}
# Métrique spécifique à cette these : corruptions liées à la luminance.
# ATTENTION : chevauche "weather" (fog, brightness) und "digital" (contrast).
LUMINOSITY = ["brightness", "fog", "contrast"]

# ── (root, suffixe de dossier) pour chaque condition ─────────────────────
CONDITIONS = {
    "BatchNorm": {"rgb": ("results_v6", ""), "lab": ("results_v6", ""),
                  "hsv": ("results_v6", ""), "grayscale": ("results_v4", "")},
    "GroupNorm": {c: ("results_v7", "_gn") for c in CS},
}


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    p = float(sps.ttest_rel(a, b).pvalue) if HAVE_SCIPY else float("nan")
    return st.mean(d), p, sum(1 for x in d if x > 0)


# ══════════════════════════════════════════════════════════════════════
#  PARTIE 1 — mCA par catégorie
# ══════════════════════════════════════════════════════════════════════
def load_per_corruption(root, colorspace):
    """{corruption: [valeur par graine]} pour un espace donné."""
    out = defaultdict(list)
    for s in SEEDS:
        f = f"{root}/cifar10c_seed{s}/corruption_summary.csv"
        if not os.path.isfile(f):
            continue
        for r in csv.DictReader(open(f)):
            if r["colorspace"] == colorspace:
                out[r["corruption"]].append(float(r["mean_acc"]))
    return out


def category_means(per_corr, corruptions):
    """Moyenne par graine sur un sous-ensemble de corruptions."""
    present = [c for c in corruptions if c in per_corr and per_corr[c]]
    if not present:
        return []
    n = min(len(per_corr[c]) for c in present)
    return [st.mean([per_corr[c][i] for c in present]) for i in range(n)]


print("=" * 78)
print("  PARTIE 1 — ROBUSTHEIT NACH KORRUPTIONSKATEGORIE")
print("=" * 78)

results = {}
for cond, mapping in CONDITIONS.items():
    data = {}
    for cs in CS:
        root, _ = mapping[cs]
        pc = load_per_corruption(root, cs)
        if not pc:
            continue
        data[cs] = {k: category_means(pc, v) for k, v in CATEGORIES.items()}
        data[cs]["luminosity"] = category_means(pc, LUMINOSITY)
        data[cs]["gesamt"] = category_means(pc, list(pc.keys()))
    results[cond] = data

    if not data:
        print(f"\n  [{cond}] aucune donnée trouvée — dossiers manquants ?")
        continue

    print(f"\n### {cond}")
    cats = ["gesamt", "noise", "blur", "weather", "digital", "luminosity"]
    print(f"  {'Farbraum':<12}" + "".join(f"{c:>12}" for c in cats))
    print("  " + "-" * 74)
    for cs in CS:
        if cs not in data:
            continue
        row = f"  {cs:<12}"
        for c in cats:
            v = data[cs].get(c, [])
            row += f"{st.mean(v)*100:>11.2f}%" if v else f"{'—':>12}"
        print(row)

    # écart entre les trois espaces couleur (grayscale exclu)
    print("  " + "-" * 74)
    row = f"  {'Fächer':<12}"
    for c in cats:
        vals = [st.mean(data[cs][c]) for cs in ["rgb", "lab", "hsv"]
                if cs in data and data[cs].get(c)]
        row += f"{(max(vals)-min(vals))*100:>11.2f} " if len(vals) == 3 else f"{'—':>12}"
    print(row + "pp")

# ── tests appariés par catégorie ─────────────────────────────────────
print("\n\n### Gepaarte Vergleiche je Kategorie")
print("=" * 78)
for cond, data in results.items():
    if not data:
        continue
    print(f"\n  {cond}")
    print(f"    {'Kategorie':<12}" + "".join(
        f"{a.upper()+'-'+b.upper():>22}" for a, b in
        [("lab", "rgb"), ("hsv", "rgb")]))
    for cat in ["noise", "blur", "weather", "digital", "luminosity"]:
        row = f"    {cat:<12}"
        for a, b in [("lab", "rgb"), ("hsv", "rgb")]:
            if a in data and b in data and data[a].get(cat) and data[b].get(cat):
                m, p, w = paired(data[a][cat], data[b][cat])
                row += f"{m*100:>+9.2f} pp p={p:<6.3f}"
            else:
                row += f"{'—':>22}"
        print(row)


# ══════════════════════════════════════════════════════════════════════
#  PARTIE 2 — Konvergenzdynamik
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 78)
print("  PARTIE 2 — KONVERGENZDYNAMIK")
print("=" * 78)


def curve(root, exp, suffix, seed):
    f = f"{root}/{exp}{suffix}_seed{seed}/metrics.csv"
    if not os.path.isfile(f):
        return None
    return [float(r["val_acc"]) for r in csv.DictReader(open(f))]


def epoch_to(c, thr):
    for i, v in enumerate(c, 1):
        if v >= thr:
            return i
    return None


for cond, mapping in CONDITIONS.items():
    curves = {}
    for cs in CS:
        root, suf = mapping[cs]
        cc = [curve(root, EXP[cs], suf, s) for s in SEEDS]
        if all(c is not None for c in cc):
            curves[cs] = cc
    if not curves:
        print(f"\n  [{cond}] aucune courbe trouvée")
        continue

    print(f"\n### {cond} — Epoche bis zum Erreichen einer Schwelle")
    print(f"  {'Schwelle':>10}" + "".join(f"{cs:>12}" for cs in curves))
    print("  " + "-" * (10 + 12 * len(curves)))
    for thr in [0.70, 0.80, 0.85, 0.88, 0.90, 0.92]:
        row = f"  {thr*100:>9.0f}%"
        for cs in curves:
            eps = [epoch_to(c, thr) for c in curves[cs]]
            row += f"{st.mean(eps):>12.1f}" if None not in eps else f"{'nie':>12}"
        print(row)

    print(f"\n  Fläche unter der Kurve (Ø val_acc über 50 Epochen)")
    for cs in curves:
        v = [st.mean(c) * 100 for c in curves[cs]]
        print(f"    {cs:<12}{st.mean(v):>7.2f}% ± {st.stdev(v):.2f}")

    if "lab" in curves and "rgb" in curves:
        a = [st.mean(c) for c in curves["rgb"]]
        b = [st.mean(c) for c in curves["lab"]]
        m, p, w = paired(b, a)
        print(f"    LAB − RGB : {m*100:+.2f} pp   p = {p:.3f}   ({w}/{len(a)} Seeds)")

print("\n" + "=" * 78)
print("""  LESEHILFE

  Kategorien : "weather" folgt Hendrycks (snow, frost, fog, brightness,
  spatter). "luminosity" ist die arbeitsspezifische Metrik (brightness,
  fog, contrast) und ÜBERSCHNEIDET SICH mit weather und digital — beide
  Zahlen dürfen nicht addiert werden.

  Konvergenz : Antwortet auf die Exposé-Frage "Does CIELAB converge
  faster?". Entscheidend ist nicht ein einzelner Schwellenwert, sondern
  ob sich das Bild über alle Schwellen hinweg konsistent zeigt.
""")
