"""CIFAR-10 data loader"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from color_dg.color_spaces import ColorspaceTransform

def get_cifar10_loaders(colorspace: str = "rgb", batch_size: int = 32, num_workers: int = 4):
    """Load CIFAR-10 dataset in specified color space"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        ColorspaceTransform(colorspace=colorspace),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
