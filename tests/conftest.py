"""Pytest configuration"""

import pytest
import torch

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_image():
    """Create a sample CIFAR-10 like image"""
    return torch.randn(3, 32, 32)
