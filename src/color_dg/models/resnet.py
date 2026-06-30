"""ResNet-18 for CIFAR-10 — built with timm (https://huggingface.co/docs/timm)"""

import torch.nn as nn
import timm


def create_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    # 1. Load ResNet-18 via timm (pretrained=True → ImageNet weights)
    model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)

    # 2. CIFAR-10 adaptation: images are 32×32, not 224×224.
    #    The default 7×7 conv1 with stride 2 would reduce 32×32 → 16×16 immediately.
    #    Replace with 3×3 / stride 1 to keep the full spatial resolution.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 3. Remove MaxPool: on 32×32 it would further reduce to 8×8 too early.
    model.maxpool = nn.Identity()

    return model
 