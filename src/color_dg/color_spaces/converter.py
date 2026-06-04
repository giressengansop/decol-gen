"""Color space conversion functions"""

import numpy as np
import cv2

class ColorspaceConverter:
    """Converts images between different color spaces"""
    
    @staticmethod
    def rgb_to_hsv(rgb):
        """Convert RGB to HSV color space"""
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return image_hsv.astype(np.float32) / 255.0
    
    @staticmethod
    def rgb_to_lab(rgb):
        """Convert RGB to LAB color space"""
        image_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_lab=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        return image_lab.astype(np.float32) / 255.0
    
    @staticmethod
    def rgb_to_grayscale(rgb):
        """Convert RGB to Grayscale color space"""
        image_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Create 3 identical channels : Duplicate the channel three times → produces a 3D array (H, W, 3)
        image_gray_3ch = np.stack((image_gray,)*3, axis=-1)
        return image_gray_3ch.astype(np.float32) / 255