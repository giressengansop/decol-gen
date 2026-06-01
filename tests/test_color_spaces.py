"""Test color space conversions"""

import numpy as np
from color_dg.color_spaces import ColorspaceConverter

def test_rgb_to_hsv():
    """Test RGB to HSV conversion"""
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    converter = ColorspaceConverter()
    hsv = converter.rgb_to_hsv(image)
    
    assert hsv.shape == (32, 32, 3)
    assert hsv.min() >= 0
    assert hsv.max() <= 1

def test_standardize():
    """Test image standardization"""
    image = np.random.randn(32, 32, 3)
    converter = ColorspaceConverter()
    standardized = converter.standardize(image)
    
    assert abs(standardized.mean()) < 1e-5
    assert abs(standardized.std() - 1.0) < 1e-5
