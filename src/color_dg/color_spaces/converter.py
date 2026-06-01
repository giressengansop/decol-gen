"""Color space conversion functions"""

import numpy as np
import cv2
from typing import Tuple

class ColorspaceConverter:
    """Converts images between different color spaces"""
    
    @staticmethod
    def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV"""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return image_hsv.astype(np.float32) / 255.0
    
    @staticmethod
    def rgb_to_lab(image: np.ndarray) -> np.ndarray:
        """Convert RGB to CIELAB"""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return image_lab.astype(np.float32) / 255.0
    
    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and expand to 3 channels"""
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_gray_3ch = np.stack([image_gray] * 3, axis=-1)
        return image_gray_3ch.astype(np.float32) / 255.0
    
    @staticmethod
    def standardize(image: np.ndarray) -> np.ndarray:
        """Normalize to mean=0, std=1"""
        mean = image.mean(axis=(0, 1), keepdims=True)
        std = image.std(axis=(0, 1), keepdims=True)
        return (image - mean) / (std + 1e-5)
