---
name: project-progress
description: Project progress — which stages have been completed, what has been coded, and what remains to be done
metadata:
  type: project
---

Bachelor’s thesis project: A comparison of RGB, HSV and CIELAB colour spaces for ResNet-18 on CIFAR-10.

**Why:** Testing whether uncorrelated channels improve the generalisation of a neural network.
**How to apply:**Always pick up where you left off; do not regenerate code that has already been written by the user.

---

## Step 1 — COMPLETED (4 June 2026)

### Files written by the user
- `src/color_dg/color_spaces/converter.py` — `ColorspaceConverter` class with:
  - `rgb_to_hsv(rgb)` — RGB → normalised HSV [0,1]
  - `rgb_to_lab(rgb)` — RGB → normalised CIELAB [0,1]
  - `rgb_to_grayscale(rgb)` — RGB → normalised 3-channel greyscale [0,1]
- `notebooks/01_visualize_colorspaces.ipynb` — visualisation of the 4 colour spaces on CIFAR-10

### Corrections made
- `src/color_dg/__init__.py` — removal of the `training` import (module not yet created)
- Conda environment: `thesis` (miniforge3)

### User observations
- Visually confirmed correlation of RGB channels
- L (LAB) and V (HSV) channels isolate brightness → decorrelation confirmed

---

## Step 2 — TO DO: Data pipeline

### Objective
Create a CIFAR-10 PyTorch `DataLoader` that applies colour conversions as transformations.

### Files to write
- `src/color_dg/color_spaces/transforms.py` — `torchvision.transforms`-compatible `ColorspaceTransform` class
- `src/color_dg/data/loader.py` — `get_cifar10_loaders(colorspace, batch_size)` function

### Resources for this step
- PyTorch DataLoader: pytorch.org/tutorials/beginner/basics/data_tutorial.html
- Torchvision transforms: pytorch.org/vision/stable/transforms.html

---

## Next steps (not yet started)
- Step 3: Adapting ResNet-18 for CIFAR-10 (32×32)
- Step 4: Train/eval training loop
- Step 5: Comparative experiments: RGB vs HSV vs LAB
- Step 6: Analysis and visualisation of results
