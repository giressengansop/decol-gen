# Robust Representation Learning via Decorrelated Color Spaces

Bachelor's thesis project — University of Bamberg, xAI Lab.

Does the choice of input color space (RGB, HSV, CIELAB, grayscale) change how
well a CNN learns and how robustly it generalizes to corrupted images?

---

## Research question and hypothesis

**Question.** RGB channels are strongly correlated: a change in illumination
shifts R, G and B together. Color spaces such as CIELAB separate luminance (L)
from chrominance (a, b) by construction. Does feeding a network a decorrelated
representation make it more robust — in particular to luminosity corruptions?

**Hypothesis (as formulated at the start).** CIELAB should improve robustness to
brightness, fog and contrast corruptions, because luminance is an explicit,
separate channel.

**Status.** This hypothesis is **not supported** by the experiments. See Results.

---

## Experimental setup

| | |
|---|---|
| Dataset | CIFAR-10 (50 000 train / 10 000 test) |
| Robustness benchmark | CIFAR-10-C — 19 corruptions × 5 severities = 95 test sets |
| Model | ResNet-18 (timm), adapted for 32×32: 3×3 stride-1 conv1, no maxpool |
| Training | 50 epochs, Adam, StepLR, batch 32, from scratch |
| Color spaces | RGB (baseline), HSV, CIELAB, grayscale (3-channel duplicate) |
| Normalization schemes | z-score (default), min-max, centered [-1, 1] |
| Seeds | 5 (0, 1, 2, 3, 42) → 20 full training runs |

Color conversions use `scikit-image`; all channels are rescaled to [0, 1] before
normalization (see `src/color_dg/color_spaces/converter.py`).

---

## Results

### Clean accuracy — mean ± std over 5 seeds

| Color space | Val. accuracy |
|---|---|
| CIELAB | 92.91 % ± 0.24 |
| RGB | 92.67 % ± 0.16 |
| HSV | 92.13 % ± 0.10 |
| Grayscale | 90.95 % ± 0.25 |

RGB and LAB are statistically indistinguishable (paired t-test, p = 0.87).

### Robustness — CIFAR-10-C, mean corruption accuracy (mCA)

| Color space | mCA (all 19) | mCA luminosity (3) | mCA non-luminosity (16) |
|---|---|---|---|
| HSV | **73.39 % ± 0.53** | 81.29 % ± 0.24 | **71.91 % ± 0.64** |
| RGB | 72.29 % ± 0.82 | 81.98 % ± 0.55 | 70.47 % ± 0.88 |
| CIELAB | 72.23 % ± 0.74 | **82.10 % ± 0.29** | 70.38 % ± 0.86 |
| Grayscale | 65.48 % ± 0.40 | 80.66 % ± 0.22 | 62.63 % ± 0.46 |

### Main findings

1. **The original hypothesis is refuted.** LAB provides no advantage over RGB on
   luminosity corruptions (+0.12 pp, p = 0.71, LAB wins 2 of 5 seeds). Verified
   independently across seeds (Phase 4) and across normalization schemes
   (Phase 5).

2. **HSV behaves differently, consistently in both directions.** It leads on
   global robustness (+1.16 pp over LAB, 5/5 seeds, p = 0.033 uncorrected) and
   trails on luminosity (−0.81 pp, 0/5 seeds, p = 0.011). After Holm correction
   across all 9 comparisons no result stays below 0.05 — this is reported as an
   exploratory observation, supported by large effect sizes (Cohen's d ≈ 1.4–2.0)
   and full consistency across seeds, not by the p-value alone.

3. **CIELAB hides a systematic trade-off.** It loses heavily on additive noise
   (gaussian −8.0 pp, shot −7.5 pp, speckle −6.3 pp vs. RGB) and gains
   systematically on blur (+2.0 to +3.0 pp). The two cancel in the average — the
   aggregate "no difference" masks two real, opposite effects. Plausible cause:
   the cube-root nonlinearity in the LAB transform has a steep derivative near
   zero and amplifies pixel noise in dark regions.

4. **Grayscale loses everywhere** — the only unambiguous, systematic effect, and
   a working negative control confirming the pipeline detects information loss.

Full statistical analysis: `docs/discussion_robustness_checks.md`.
Literature review: `docs/revue_litterature_phase6.md`.

---

## Installation

```bash
conda create -n color_dg python=3.11
conda activate color_dg
pip install -e ".[dev]"
```

CIFAR-10 downloads automatically. CIFAR-10-C must be fetched separately from
[Zenodo](https://zenodo.org/record/2535967) and extracted to `data/CIFAR-10-C/`.

---

## Usage

```bash
# Single training run
python -m scripts.train_model --config configs/baseline_rgb.yaml

# Override seed and output directory
python -m scripts.train_model --config configs/exp_lab.yaml \
    --seed 0 --output_dir results_v4/exp_lab_seed0

# All color spaces, all seeds (Phase 4)
bash scripts/run_phase4_multiseed.sh

# CIFAR-10-C evaluation
python -m scripts.eval_cifar10c \
    --cifar10c_root data/CIFAR-10-C \
    --results_dir results_v4 --seed 0 \
    --output_dir results_v4/cifar10c_seed0

# Normalization ablation (Phase 5)
bash scripts/run_phase5_normalization.sh

# Aggregate Phase 4 + Phase 5
python -m scripts.aggregate_phase4_phase5

# Statistical verification (paired tests, multiple-comparison correction)
python verify_stats.py
```

---

## Project structure

```
src/color_dg/
├── color_spaces/     conversions (converter.py) + torchvision transforms
├── data/             CIFAR-10 loaders, color-space aware
└── models/           ResNet-18 adapted for 32×32
configs/              one YAML per experiment (color space × normalization)
scripts/              training, CIFAR-10-C evaluation, plots, aggregation
tests/                unit tests for color conversions
notebooks/            color space visualization
docs/                 analysis, literature review, presentations
results_v4/           Phase 4 — multi-seed (reference results)
results_v5/           Phase 5 — normalization ablation
results_aggregate/    combined Phase 4 + Phase 5 summary
```

Results directories are versioned by phase; `results_v4/` holds the reference
numbers reported above.

---

## Known limitations

- Best-epoch selection uses the test set (no separate validation split).
  Absolute accuracies are therefore mildly optimistic; the bias is identical
  across all four color spaces and does not affect comparisons.
- Single architecture (ResNet-18) and single resolution (32×32).
- The HSV finding is post-hoc, not a pre-registered hypothesis.
- Local contrast normalization has not been tested.

---

## Author

Giresse N'Jinkap — University of Bamberg
`gnjinkap@uni-bamberg.de`

Licensed under MIT — see `LICENSE`.

## Ergebnisverzeichnisse

| Verzeichnis | Architektur | Normalisierung | Farbräume | Seeds |
|---|---|---|---|---|
| `results_v4` | ResNet-18 | BatchNorm | 4 | 5 |
| `results_v5` | ResNet-18 | BatchNorm | 2 | 3 |
| `results_v6` | ResNet-18 | BatchNorm | 3 | 5 |
| `results_v6_gray` | ResNet-18 | BatchNorm | Graustufen | 1 |
| `results_v7` | ResNet-18 | GroupNorm | 4 | 5 |
| `results_v8` | VGG-11 | keine | 4 | 5 |
| `results_v9` | VGG-11 | BatchNorm | 4 | 5 |

`results_v4` und `results_v5` entstanden vor der Neuberechnung der
Normalisierungskonstanten (Commit `e02b93e`); für RGB, CIELAB und HSV sind
`results_v6` und später maßgeblich. Der Graustufen-Eintrag war von der
Korrektur nicht betroffen.

In `results_v8` und `results_v9` ist `cifar10c4_seed*` die maßgebliche
Auswertung über alle vier Farbräume; `cifar10c_seed*` stammt aus einem
früheren Durchgang ohne HSV.

## Ergebnisverzeichnisse

| Verzeichnis | Architektur | Normalisierung | Farbräume | Seeds |
|---|---|---|---|---|
| `results_v4` | ResNet-18 | BatchNorm | 4 | 5 |
| `results_v5` | ResNet-18 | BatchNorm | 2 | 3 |
| `results_v6` | ResNet-18 | BatchNorm | 3 | 5 |
| `results_v6_gray` | ResNet-18 | BatchNorm | Graustufen | 1 |
| `results_v7` | ResNet-18 | GroupNorm | 4 | 5 |
| `results_v8` | VGG-11 | keine | 4 | 5 |
| `results_v9` | VGG-11 | BatchNorm | 4 | 5 |

`results_v4` und `results_v5` entstanden vor der Neuberechnung der
Normalisierungskonstanten (Commit `e02b93e`); für RGB, CIELAB und HSV sind
`results_v6` und später maßgeblich. Der Graustufen-Eintrag war von der
Korrektur nicht betroffen, weshalb die Graustufen-Bedingung aus `results_v4`
weiterhin gültig ist; `results_v6_gray` enthält den Kontrolllauf, der dies
bestätigt.

In `results_v8` und `results_v9` ist `cifar10c4_seed*` die maßgebliche
Auswertung über alle vier Farbräume; `cifar10c_seed*` stammt aus einem
früheren Durchgang ohne HSV.
