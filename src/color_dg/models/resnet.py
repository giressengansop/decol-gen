"""ResNet-18 for CIFAR-10 — built with timm (https://huggingface.co/docs/timm)"""

from functools import partial

import torch.nn as nn
import timm

try:                                    # timm >= 0.9
    from timm.layers import GroupNorm
except ImportError:                     # older timm
    from timm.models.layers import GroupNorm


def create_resnet18(num_classes: int = 10, pretrained: bool = False,
                    norm: str = "bn", gn_groups: int = 32) -> nn.Module:
    """ResNet-18 adapted to CIFAR-10.

    norm:
      "bn" — BatchNorm2d, timm's default. Used for phases 2-6.
      "gn" — GroupNorm (Wu & He, 2018), num_groups=32.

    Why "gn" exists: the discussion chapter argues that BatchNorm realigns the
    channels, so the network learns its own decorrelation and the input color
    space becomes irrelevant. GroupNorm normalizes over channel groups WITHIN
    each image — no batch statistics, no running averages. If the BatchNorm
    explanation holds, color spaces should separate under GroupNorm.

    NOTE: timm's GroupNorm takes num_channels FIRST (drop-in for BatchNorm2d),
    and the group count must be passed as a KEYWORD: timm's get_norm_layer()
    only preserves partial keywords and silently drops positional args.
    """
    if norm == "bn":
        kwargs = {}                                      # timm default
    elif norm == "gn":
        kwargs = {"norm_layer": partial(GroupNorm, num_groups=gn_groups)}
    else:
        raise ValueError(f"Unknown norm: {norm!r} (expected 'bn' or 'gn')")

    model = timm.create_model("resnet18", pretrained=pretrained,
                              num_classes=num_classes, **kwargs)

    # CIFAR-10 adaptation: 3x3 stride-1 conv1 instead of 7x7 stride-2,
    # and no maxpool — otherwise 32x32 collapses to 8x8 immediately.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model
