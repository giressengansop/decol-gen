"""PyTorch transforms for color space conversion"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from .converter import ColorspaceConverter

# Per-channel normalization statistics for CIFAR-10.
#
# Recalculated on 28 July 2026 over the full 50,000-image training set with
# compute_cifar10_stats.py, which reuses this project's own converter.
#
# PREVIOUS VALUES (incorrect — used for results_v2 through results_v5):
#   "rgb": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
#   "hsv": {"mean": [0.3096, 0.2223, 0.3277], "std": [0.2418, 0.2047, 0.3050]},
#   "lab": {"mean": [0.3277, 0.3277, 0.3277], "std": [0.2836, 0.1478, 0.1788]},
#
# The LAB standard deviations of the chroma channels were overestimated by a
# factor of 3.7 (a) and 2.8 (b). After z-score normalization, a and b reached
# the network with 0.32 and 0.41 times the variance of L instead of 1.0 — LAB
# was the only color space whose channels were not equally weighted. The LAB
# means were also identical across all three channels and equal to the V
# channel of HSV (copy-paste error); after (a+128)/255, a and b must be ~0.50.
#
# Verification (Phase 5 bis, results_v6): RGB and LAB retrained and re-evaluated
# on 5 seeds with the corrected constants. LAB - RGB is unchanged on all four
# metrics — accuracy +0.18 pp (p = 0.14), mCA -0.05 pp (p = 0.67), mCA
# luminosity +0.21 pp (p = 0.31). The correction reveals no latent LAB
# advantage; the conclusions did not depend on this error.
CIFAR10_STATS = {
    "rgb":         {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    "hsv":         {"mean": [0.3254, 0.2738, 0.5393], "std": [0.2765, 0.2184, 0.2472]},
    "lab":         {"mean": [0.5085, 0.5035, 0.5244], "std": [0.2426, 0.0398, 0.0631]},
    "grayscale":   {"mean": [0.4809, 0.4809, 0.4809], "std": [0.2392, 0.2392, 0.2392]},
}




class ColorspaceTransform:
    """Torchvision-compatible transform: PIL image → CHW float tensor."""

    def __init__(self, colorspace: str = "rgb"):
        self.colorspace = colorspace.lower()
        self.converter = ColorspaceConverter()


    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)   # shape: (H, W, 3), dtype: uint8

        # conversion to the target color space (call our converter)
        if self.colorspace == "rgb":
            image = image.astype(np.float32) / 255.0
        elif self.colorspace == "hsv":
            image = self.converter.rgb_to_hsv(image)
        elif self.colorspace == "lab":
            image = self.converter.rgb_to_lab(image)
        elif self.colorspace == "grayscale":
            image = self.converter.rgb_to_grayscale(image)
        else:
            raise ValueError(f"Unknown colorspace: '{self.colorspace}'")

        return torch.from_numpy(image).permute(2, 0, 1)


def _build_normalize(colorspace: str, normalization: str):
    """Phase 5 — normalization scheme ablation (RGB/LAB only, see docs/plan_recherche_suite.md).

    - "zscore"  : per-channel mean/std from CIFAR10_STATS (reference, phases 2-4).
    - "minmax"  : no extra normalization — channels stay in [0, 1] as produced
                  by the colorspace converter.
    - "centered": maps [0, 1] -> [-1, 1], i.e. Normalize(mean=0.5, std=0.5).
    """
    if normalization == "zscore":
        stats = CIFAR10_STATS.get(colorspace, CIFAR10_STATS["rgb"])
        return transforms.Normalize(mean=stats["mean"], std=stats["std"])
    if normalization == "minmax":
        return None
    if normalization == "centered":
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if normalization == "uniscale":
        # Alle Kanaele mit der Streuung des ERSTEN Kanals skalieren (bei CIELAB: L*).
        # Die Zentrierung bleibt kanalweise, nur die Skalierung wird vereinheitlicht.
        # Damit ist die Kanalskalierung isoliert von der Zentrierung variiert.
        stats = CIFAR10_STATS.get(colorspace, CIFAR10_STATS["rgb"])
        s0 = stats["std"][0]
        return transforms.Normalize(mean=stats["mean"], std=[s0, s0, s0])
    raise ValueError(f"Unknown normalization: '{normalization}'")


def get_transforms(colorspace: str = "rgb", train: bool = True, normalization: str = "zscore"):
    normalize = _build_normalize(colorspace, normalization)
    steps = []
    if train:
        steps += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    steps.append(ColorspaceTransform(colorspace=colorspace))
    if normalize is not None:
        steps.append(normalize)
    return transforms.Compose(steps)

