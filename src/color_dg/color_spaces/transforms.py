"""PyTorch transforms for color space conversion"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from .converter import ColorspaceConverter

class ColorspaceTransform:
    """PyTorch transform for color space conversion"""
    
    def __init__(self, colorspace: str = "rgb"):
        self.colorspace = colorspace.lower()
        self.converter = ColorspaceConverter()
    
    def __call__(self, image):
        """Transform image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert colorspace
        if self.colorspace == "rgb":
            image = image.astype(np.float32) / 255.0
        elif self.colorspace == "hsv":
            image = self.converter.rgb_to_hsv(image)
        elif self.colorspace == "lab":
            image = self.converter.rgb_to_lab(image)
        elif self.colorspace == "grayscale":
            image = self.converter.rgb_to_grayscale(image)
        
        # Standardize
        image = self.converter.standardize(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

def get_transforms(colorspace: str = "rgb"):
    """Get composition of transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        ColorspaceTransform(colorspace=colorspace),
    ])
