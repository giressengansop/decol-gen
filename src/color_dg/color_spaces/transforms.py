"""PyTorch transforms for color space conversion"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .converter import ColorspaceConverter

# Normalization stats for CIFAR-10 per colorspace.
# RGB: exact values — see github.com/kuangliu/pytorch-cifar
# HSV / LAB / grayscale: placeholder [0.5, 0.5, 0.5] until computed on the dataset.
CIFAR10_STATS = {
    "rgb":       {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "hsv":       {"mean": [0.5, 0.5, 0.5],          "std": [0.5, 0.5, 0.5]},
    "lab":       {"mean": [0.5, 0.5, 0.5],          "std": [0.5, 0.5, 0.5]},
    "grayscale": {"mean": [0.5, 0.5, 0.5],          "std": [0.5, 0.5, 0.5]},
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


def get_transforms(colorspace: str = "rgb", train: bool = True):
    stats = CIFAR10_STATS.get(colorspace, CIFAR10_STATS["rgb"])
    normalize = transforms.Normalize(mean=stats["mean"], std=stats["std"])

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # ← increase
            transforms.RandomHorizontalFlip(),       # ← inrease
            ColorspaceTransform(colorspace=colorspace),
            normalize,
        ])
    return transforms.Compose([
        ColorspaceTransform(colorspace=colorspace),
        normalize,
    ])

