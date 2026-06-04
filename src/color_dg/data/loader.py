"""CIFAR-10 data loaders with colorspace-aware transforms."""

from torch.utils.data import DataLoader
from torchvision import datasets
from ..color_spaces.transforms import get_transforms

def get_cifar10_loaders(
    colorspace: str = "rgb",
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
):
    train_loader = DataLoader(
        datasets.CIFAR10(root=data_root, train=True,  download=True,
                         transform=get_transforms(colorspace, train=True)),
        batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.CIFAR10(root=data_root, train=False, download=True,
                         transform=get_transforms(colorspace, train=False)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader

    