"""ResNet-18 for CIFAR-10"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def create_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    # 1. Load the weights (new TorchVision API >= 0.13)
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # 2. Replace conv1: 7×7/stride-2 → 3×3/stride-1
    # Sur 32×32 : l'original sortirait 16×16, on garde 32×32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # 3. Remove the MaxPool (to avoid switching from 32×32 to 8×8 too early)
    model.maxpool = nn.Identity()

    # 4. Replace the head : 1000 classes ImageNet → num_classes(Adjust the final layer (10 classes instead of 1,000))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
 