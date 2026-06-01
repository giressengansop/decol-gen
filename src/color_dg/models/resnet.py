"""ResNet-18 for CIFAR-10"""

import torch.nn as nn
from torchvision import models

def create_resnet18(num_classes: int = 10, pretrained: bool = False):
    """Create ResNet-18 adapted for CIFAR-10"""
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
