"""
Color Domain Generalization Package

A package for investigating decorrelated color spaces in deep learning.
"""

__version__ = "0.1.0"
__author__ = "Giresse N'Jinkap"

from . import color_spaces, data, models, losses, utils

__all__ = [
    "color_spaces",
    "data",
    "models",
    "losses",
    "utils",
]
