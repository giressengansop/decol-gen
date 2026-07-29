#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_stats.py — Statistical verification of Phase 4 / Phase 5 results.

To be run from the project root directory:
    cd ~/decol-gen
    python verify_stats.py

This script does not modify anything. It reads the `results_v4/` directory and produces:

  1. A summary table by colour space (mean ± standard deviation).
  2. Paired tests (t-test AND Wilcoxon signed-rank test) for all pairs.
  3. A correction for multiple tests (Holm-Bonferroni).
  4. The recalculation restricted to the 15 standard Hendrycks corruptions,
     to verify that the ranking does not depend on the 4 additional
     corruptions in the Zenodo repository.
  5. The deviation per corruption, sorted — the noise/blur trade-off.

DDependencies: numpy, scipy (scipy is optional, degradation proper without it).
"""

import argparse
import csv
import json
import os
import collections
import statistics as st
from math import sqrt

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

SEEDS = [0, 1, 2, 3, 42]
EXPERIMENTS = {
    "rgb": "baseline_rgb",
    "lab": "exp_lab",
    "hsv": "exp_hsv",
    "grayscale": "exp_grayscale",
}
ORDER = ["rgb", "lab", "hsv", "grayscale"]

# Les 15 corruptions standard de Hendrycks & Dietterich (2019).
# Les 4 autres (speckle_noise, gaussian_blur, spatter, saturate) sont fournies
# par le dépôt Zenodo comme jeu de validation.
HENDRYCKS_15 = {
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
}
LUMINOSITY = {"brightness", "fog", "contrast"}


# ── utilitaires ─────────────────────────────────────────────────────────────
def rule(char="─", n=78):
    print(char * n)


def holm_bonferroni(pvals):
    """Retourne les p-values ajustées selon Holm-Bonferroni."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def paired_test(a, b):
    """t-test apparié + Wilcoxon. Retourne un dict."""
    diff = [x - y for x, y in zip(a, b)]
    mean = st.mean(diff)
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    sem = sd / sqrt(len(diff)) if sd else 0.0
    t = mean / sem if sem else float("nan")
    p_t = float(sps.ttest_rel(a, b).pvalue) if HAVE_SCIPY else float("nan")
    try:
        p_w = float(sps.wilcoxon(a, b).pvalue) if HAVE_SCIPY else float("nan")
    except Exception:
        p_w = float("nan")
    # d de Cohen pour données appariées
    d = mean / sd if sd else float("nan")
    return {
        "mean": mean, "sd": sd, "t": t, "p_t": p_t, "p_w": p_w,
        "d": d, "wins": sum(1 for x in diff if x > 0), "n": len(diff),
    }


# ── chargement ──────────────────────────────────────────────────────────────
def load_summary(results_dir, metric):
    """{colorspace: [valeur par graine]} depuis les summary.json."""
    out = collections.defaultdict(list)
    missing = []
    for s in SEEDS:
        path = os.path.join(results_dir, f"cifar10c_seed{s}", "summary.json")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        d = json.load(open(path))
        for cs, exp in EXPERIMENTS.items():
            if exp in d and d[exp].get(metric) is not None:
                out[cs].append(d[exp][metric])
    if missing:
        print(f"  [!] {len(missing)} fichier(s) manquant(s), ex. {missing[0]}")
    return out


def load_per_corruption(results_dir):
    """{corruption: {colorspace: [valeur par graine]}}"""
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in SEEDS:
        path = os.path.join(results_dir, f"cifar10c_seed{s}", "corruption_summary.csv")
        if not os.path.isfile(path):
            continue
        for r in csv.DictReader(open(path)):
            data[r["corruption"]][r["colorspace"]].append(float(r["mean_acc"]))
    return data


def load_val_acc(results_dir):
    """Accuracy de validation depuis les dossiers d'entraînement."""
    out = collections.defaultdict(list)
    for s in SEEDS:
        for cs, exp in EXPERIMENTS.items():
            path = os.path.join(results_dir, f"{exp}_seed{s}", "summary.json")
            if os.path.isfile(path):
                out[cs].append(json.load(open(path))["best_val_acc"])
    return out


# ── sections du rapport ─────────────────────────────────────────────────────
def section_table(title, data, unit="%"):
    print(f"\n### {title}")
    rule()
    print(f"{'espace':<12}{'n':>3}{'moyenne':>11}{'écart-type':>13}{'min':>9}{'max':>9}")
    rule()
    ranked = sorted(ORDER, key=lambda c: -st.mean(data[c]) if data[c] else 0)
    for cs in ranked:
        v = data[cs]
        if not v:
            continue
        m = st.mean(v) * 100
        sd = st.stdev(v) * 100 if len(v) > 1 else 0.0
        print(f"{cs:<12}{len(v):>3}{m:>10.2f}{unit}{sd:>12.2f}"
              f"{min(v)*100:>9.2f}{max(v)*100:>9.2f}")
    rule()


def section_pairwise(title, data, pairs):
    print(f"\n### Tests appariés — {title}")
    rule()
    print(f"{'comparaison':<16}{'Δ (pp)':>9}{'d':>7}{'t':>8}{'p (t)':>9}"
          f"{'p (W)':>9}{'gagne':>9}")
    rule()
    results = []
    for a, b in pairs:
        if not data[a] or not data[b]:
            continue
        r = paired_test(data[a], data[b])
        results.append(((a, b), r))
        label = f"{a.upper()} − {b.upper()}"
        wins = f"{r['wins']}/{r['n']}"
        print(f"{label:<16}{r['mean']*100:>+9.2f}{r['d']:>7.2f}"
              f"{r['t']:>8.2f}{r['p_t']:>9.3f}{r['p_w']:>9.3f}{wins:>9}")
    rule()
    print("  Note : avec n = 5, la p-value minimale possible du test de Wilcoxon")
    print("  bilatéral est 0,0625. Une valeur de 0,0625 signifie donc « aussi")
    print("  significatif que possible à cette taille d'échantillon », pas « non")
    print("  significatif ».")
    return results


def section_holm(all_results):
    """Correction pour tests multiples sur l'ensemble des comparaisons."""
    if not HAVE_SCIPY:
        print("\n[scipy absent — correction Holm non calculée]")
        return
    print("\n### Correction pour tests multiples (Holm-Bonferroni)")
    print("Sur l'ensemble des comparaisons ci-dessus.")
    rule()
    labels = [f"{m} : {a.upper()} − {b.upper()}" for m, (a, b), _ in all_results]
    pvals = [r["p_t"] for _, _, r in all_results]
    adj = holm_bonferroni(pvals)
    print(f"{'test':<40}{'p brut':>10}{'p ajusté':>12}{'':>6}")
    rule()
    for lab, p, pa in sorted(zip(labels, pvals, adj), key=lambda x: x[1]):
        mark = "***" if pa < 0.01 else ("**" if pa < 0.05 else ("*" if pa < 0.10 else ""))
        print(f"{lab:<40}{p:>10.3f}{pa:>12.3f}{mark:>6}")
    rule()
    print("** p ajusté < 0,05   *** p ajusté < 0,01   * < 0,10 (tendance)")


def section_subset(per_corr, subset, label):
    """Recalcule le mCA sur un sous-ensemble de corruptions."""
    print(f"\n### mCA restreint — {label}")
    rule()
    means = {}
    for cs in ORDER:
        per_seed = []
        for i in range(len(SEEDS)):
            vals = [per_corr[c][cs][i] for c in per_corr
                    if c in subset and len(per_corr[c][cs]) > i]
            if vals:
                per_seed.append(st.mean(vals))
        means[cs] = per_seed
    for cs in sorted(ORDER, key=lambda c: -st.mean(means[c]) if means[c] else 0):
        if means[cs]:
            print(f"{cs:<12}{st.mean(means[cs])*100:>8.2f}%"
                  f"   ± {st.stdev(means[cs])*100:.2f}")
    rule()
    if means["hsv"] and means["lab"]:
        r = paired_test(means["hsv"], means["lab"])
        print(f"HSV − LAB : {r['mean']*100:+.2f} pp   p(t) = {r['p_t']:.3f}"
              f"   gagne {r['wins']}/{r['n']}")
    return means


def section_per_corruption(per_corr):
    print("\n### Écart à RGB par corruption (moyenne sur les graines, pp)")
    rule()
    rows = []
    for c, d in per_corr.items():
        if not d["rgb"]:
            continue
        rows.append((
            c,
            (st.mean(d["lab"]) - st.mean(d["rgb"])) * 100 if d["lab"] else float("nan"),
            (st.mean(d["hsv"]) - st.mean(d["rgb"])) * 100 if d["hsv"] else float("nan"),
            c in LUMINOSITY,
            c in HENDRYCKS_15,
        ))
    rows.sort(key=lambda r: r[1])
    print(f"{'corruption':<22}{'LAB−RGB':>10}{'HSV−RGB':>10}  {'lum':>4} {'std15':>6}")
    rule()
    for c, l, h, lum, std in rows:
        print(f"{c:<22}{l:>+10.2f}{h:>+10.2f}  {'oui' if lum else '':>4} "
              f"{'oui' if std else '':>6}")
    rule()


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_v4",
                    help="Dossier contenant cifar10c_seed*/ et <exp>_seed*/")
    args = ap.parse_args()

    print("=" * 78)
    print("  VÉRIFICATION STATISTIQUE — decol-gen")
    print(f"  Source : {args.results_dir}/   Graines : {SEEDS}")
    if not HAVE_SCIPY:
        print("  [!] scipy non installé — p-values indisponibles.")
        print("      pip install scipy")
    print("=" * 78)

    # 1. Accuracy
    val = load_val_acc(args.results_dir)
    if any(val.values()):
        section_table("Accuracy de validation (images propres)", val)

    # 2. Robustesse, 3 métriques
    all_results = []
    pairs = [("hsv", "lab"), ("hsv", "rgb"), ("lab", "rgb")]

    for metric, title in [
        ("mCA", "Robustesse globale (mCA, toutes corruptions)"),
        ("mCA_luminosity", "Robustesse luminosité (brightness, fog, contrast)"),
        ("mCA_non_luminosity", "Robustesse hors luminosité"),
    ]:
        data = load_summary(args.results_dir, metric)
        if not any(data.values()):
            continue
        section_table(title, data)
        res = section_pairwise(title, data, pairs)
        for (a, b), r in res:
            all_results.append((metric, (a, b), r))

    # 3. Correction tests multiples
    if all_results:
        section_holm(all_results)

    # 4. Sous-ensemble Hendrycks
    per_corr = load_per_corruption(args.results_dir)
    if per_corr:
        section_subset(per_corr, HENDRYCKS_15,
                       "15 corruptions standard de Hendrycks")
        section_subset(per_corr, set(per_corr) - HENDRYCKS_15,
                       "4 corruptions supplémentaires (Zenodo)")
        section_per_corruption(per_corr)

    print("\n" + "=" * 78)
    print("  Lecture rapide")
    print("=" * 78)
    print("""
  • Si HSV − LAB reste significatif après Holm ET sur les 15 corruptions
    standard, tu peux le présenter comme un résultat, pas seulement comme
    une observation.

  • Si la significativité disparaît après correction, garde la formulation
    exploratoire : « effet cohérent sur 5/5 graines, à confirmer sur un
    échantillon plus large ». C'est défendable et honnête.

  • Le d de Cohen aide à distinguer « petit effet bien mesuré » de
    « gros effet mal mesuré ». |d| > 0,8 = effet fort.
""")


if __name__ == "__main__":
    main()
