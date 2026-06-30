"""Color space conversion functions"""

import numpy as np
import cv2
from skimage.color import rgb2hsv, rgb2lab


class ColorspaceConverter:
    """Converts images between different color spaces"""

    @staticmethod
    def rgb_to_hsv(rgb):
        """Convert RGB to HSV using scikit-image.

        skimage.color.rgb2hsv expects float input in [0, 1] and returns
        H, S, V all in [0, 1] — unlike OpenCV which returns H in [0, 179].
        """
        rgb_float = rgb.astype(np.float32) / 255.0   # uint8 → [0, 1]
        return rgb2hsv(rgb_float).astype(np.float32)  # H,S,V in [0, 1]

    @staticmethod
    def rgb_to_lab(rgb):
        """Convert RGB to CIE-LAB using scikit-image.

        skimage.color.rgb2lab returns:
          L  in [0, 100]      → divided by 100        → [0, 1]
          a  in [-128, 127]   → (a + 128) / 255       → [0, 1]
          b  in [-128, 127]   → (b + 128) / 255       → [0, 1]
        This preserves the perceptual meaning of each channel.
        """
        rgb_float = rgb.astype(np.float32) / 255.0   # uint8 → [0, 1]
        lab = rgb2lab(rgb_float).astype(np.float32)   # L:[0,100] a,b:[-128,127]
        lab[:, :, 0] = lab[:, :, 0] / 100.0               # L  → [0, 1]
        lab[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0     # a  → [0, 1]
        lab[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0     # b  → [0, 1]
        return lab

    @staticmethod
    def rgb_to_grayscale(rgb):
        """Convert RGB to Grayscale (3-channel duplicate for ResNet compatibility)."""
        image_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Duplicate the single channel 3 times → shape (H, W, 3)
        image_gray_3ch = np.stack((image_gray,) * 3, axis=-1)
        return image_gray_3ch.astype(np.float32) / 255.0