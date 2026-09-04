# Evaluating the Impact of Alternative Color Spaces on Image Classification

Bachelor's thesis project — University of Bamberg, Chair of Explainable Machine Learning.

Does the choice of input color space (RGB, HSV, CIELAB, grayscale) change how
well a CNN learns and how robustly it generalizes to corrupted images?

---

## Research question and hypothesis

**Question.** RGB channels are strongly correlated: a change in illumination
shifts R, G and B together. Color spaces such as CIELAB separate luminance (L\*)
from chrominance (a\*, b\*) by construction. Does feeding a network a
decorrelated representation make it more robust — in particular to
illumination-related corruptions?

**Hypothesis.** CIELAB should improve robustness to brightness, fog and
contrast corruptions, because luminance is an explicit, separate channel.

**Status.** Not supported. The difference is smaller than the spread between
two runs of the same condition, does not survive correction for multiple
testing, and does not appear on a second architecture. See Results.

---

## Experimental setup

| | |
|---|---|
| Dataset | CIFAR-10 (50 000 train / 10 000 test) |
| Robustness benchmark | CIFAR-10-C — 15 standard corruptions × 5 severities |
| Architectures | ResNet-18 (timm, adapted for 32×32) and VGG-11 |
| Normalization layers | BatchNorm, GroupNorm (32 groups), none |
| Input normalization | z-score, min-max, centered, unified scaling |
| Color spaces | RGB (baseline), CIELAB, HSV, grayscale (3-channel duplicate) |
| Training | 50 epochs, Adam, StepLR, batch 32, from scratch |
| Seeds | 5 (0, 1, 2, 3, 42) per condition |

Color conversions use `scikit-image`; all channels are rescaled to [0, 1]
before normalization (see `src/color_dg/color_spaces/converter.py`).

The four additional corruptions shipped with CIFAR-10-C (speckle noise,
gaussian blur, spatter, saturate) are evaluated separately and are not part of
the headline metric.

---

## Results

### Clean accuracy — ResNet-18 + BatchNorm, mean ± std over 5 seeds

| Color space | Val. accuracy |
|---|---|
| CIELAB | 92.76 % ± 0.14 |
| RGB | 92.58 % ± 0.13 |
| HSV | 92.40 % ± 0.08 |
| Grayscale | 90.95 % ± 0.22 |

### Robustness — CIFAR-10-C, 15 standard corruptions

| Color space | mCA | mCA illumination (3) | mCA rest (12) |
|---|---|---|---|
| CIELAB | 71.79 % ± 0.68 | **82.08 %** ± 0.51 | 69.22 % ± 0.74 |
| HSV | 71.73 % ± 0.68 | 81.40 % ± 0.36 | **69.32 %** ± 0.77 |
| RGB | 71.66 % ± 0.52 | 81.87 % ± 0.36 | 69.11 % ± 0.58 |
| Grayscale | 64.59 % ± 0.34 | 80.66 % ± 0.20 | 60.57 % ± 0.41 |

### Main findings

**1. The hypothesis is not supported.** CIELAB leads RGB by 0.21 pp on
illumination-related corruptions (p = 0.31, 3 of 5 seeds). The per-seed
differences change sign, and the standard deviation between five runs of the
*same* condition (0.36 pp) exceeds the measured effect. After Holm correction
across the nine pairwise comparisons, no robustness comparison remains
significant.

**2. The aggregate hides a trade-off.** Broken down by corruption type, CIELAB
loses 9.72 pp on noise and gains 4.64 pp on blur — consistently across all
seeds (0/5 and 5/5), and both surviving Holm correction. The two nearly cancel
in the mean. The same pattern appears on VGG-11 with almost twice the
amplitude.

**3. The cause is the input normalization, not the color transform.** CIELAB's
chroma channels have very small standard deviations (0.0398 and 0.0631 vs
0.2426 for L\*), so per-channel standardization amplifies them four- to sixfold
relative to an RGB channel. Unifying the scaling — dividing all channels by
σ(L\*), leaving the centering untouched — reduces the amplitude of the
trade-off from 14.35 to 5.67 pp, while the RGB control condition changes by
only −0.54 pp.

**4. Networks build their own luminance-chrominance decomposition.** An
eigenvalue analysis of the first-layer channel weights shows an effective rank
of 2.63 for RGB-trained networks versus 2.76 for CIELAB-trained ones, with the
principal axis at |cos| = 0.994 to the luminance direction (1,1,1) — against
1.20 for the raw pixel colors and 1.37 for the grayscale control. A
decorrelated input therefore saves the network work it does anyway.

**5. The normalization layer matters roughly ten times more than the color
space.** Switching ResNet-18 from BatchNorm to GroupNorm improves robustness by
6.4–6.9 pp for every color space; removing the normalization layer entirely
from VGG-11 costs 5.75–6.12 pp. Color space differences in the same setup stay
below one percentage point.

**6. Grayscale loses everywhere** — 6.97 pp on the 15 corruptions, every seed,
in every configuration. This negative control confirms that the design detects
a genuine difference in color content when one exists.

Full statistical analysis: `docs/discussion_robustness_checks.md`.
Literature review: `docs/revue_litterature_phase6.md`.
Pre-registered predictions for the VGG-11 replication: `PREREGISTRATION_vgg11.md`.

---

## Repository structure

```
src/color_dg/
├── color_spaces/     conversions + torchvision transforms
├── data/             CIFAR-10 loaders, color-space aware
└── models/           ResNet-18 and VGG-11, parameterized normalization
configs/              one YAML per experiment
scripts/              training, evaluation, analysis, plotting
tests/                unit tests for color conversions
notebooks/            color space visualization
docs/                 statistical analysis, literature review
logs/                 raw training logs
runs/                 experiment outputs (see below)
```

### Experiment directories

| Directory | Architecture | Normalization layer | Input normalization | Color spaces | Seeds |
|---|---|---|---|---|---|
| `runs/resnet18_bn` | ResNet-18 | BatchNorm | z-score | RGB, LAB, HSV | 5 |
| `runs/resnet18_bn_phase4` | ResNet-18 | BatchNorm | z-score | 4 | 5 |
| `runs/resnet18_bn_gray_control` | ResNet-18 | BatchNorm | z-score | grayscale | 1 |
| `runs/resnet18_bn_normschemes` | ResNet-18 | BatchNorm | min-max, centered | RGB, LAB | 3 |
| `runs/resnet18_bn_uniscale` | ResNet-18 | BatchNorm | unified scaling | RGB, LAB | 5 |
| `runs/resnet18_gn` | ResNet-18 | GroupNorm | z-score | 4 | 5 |
| `runs/vgg11_bn` | VGG-11 | BatchNorm | z-score | 4 | 5 |
| `runs/vgg11_nonorm` | VGG-11 | none | z-score | 4 | 5 |
| `runs/aggregate` | — | combined summaries and conv1 analysis | | | |
| `runs/legacy/` | — | early exploratory runs, superseded | | | |

The reference numbers reported above come from `runs/resnet18_bn` for RGB,
CIELAB and HSV, and from `runs/resnet18_bn_phase4` for grayscale. In
`runs/vgg11_*`, the `cifar10c4_seed*` subdirectories cover all four color
spaces and are the ones used.

Trained checkpoints (`*.pth`) are not versioned; rerun the training commands
below to regenerate them.

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
python -m scripts.train_model --config configs/exp_lab.yaml \
    --seed 0 --output_dir runs/resnet18_bn/exp_lab_seed0

# CIFAR-10-C evaluation
python -m scripts.eval_cifar10c \
    --cifar10c_root data/CIFAR-10-C \
    --results_dir runs/resnet18_bn --seed 0 \
    --output_dir runs/resnet18_bn/cifar10c_seed0

# Recompute the normalization constants
python scripts/compute_cifar10_stats.py --data_root ./data

# Statistical verification (paired tests, Holm correction, per-corruption breakdown)
python scripts/verify_stats.py

# Analysis of the learned first-layer filters
python -m scripts.analyze_conv1_multi
```

---

## Known limitations

- Best-epoch selection uses the test set (no separate validation split).
  Absolute accuracies are therefore mildly optimistic; the bias is identical
  across all conditions and does not affect comparisons.
- Both architectures are convolutional; nothing is established for models
  without convolution.
- Single resolution (32×32) and single dataset (CIFAR-10).
- The per-corruption breakdown is a post-hoc analysis, not a pre-registered
  hypothesis. The VGG-11 replication is pre-registered.
- Under unified scaling, CIELAB leads RGB on illumination corruptions at all
  five seeds (+0.91 pp, p = 0.071). This is below significance but consistent,
  and suggests the null result may be tied to the standard preprocessing rather
  than to the color space itself.

---

## Author

Giresse Ngansop — University of Bamberg
`giresse-ginola.ngansop-njinkap@stud.uni-bamberg.de`

Licensed under MIT — see `LICENSE`.