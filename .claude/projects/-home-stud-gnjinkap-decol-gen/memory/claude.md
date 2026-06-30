1. The project in a sentence

Bachelor’s thesis: “Evaluating the Impact of Alternative Colour Spaces on Image
Classification” (formal framework: Robust Representation Learning via Decorrelated
Colour Spaces for Clinical Domain Generalisation).

Research question: does changing the input colour space (RGB → HSV /
CIELAB / greyscale) improve the accuracy and, ABOVE ALL, the robustness of a
neural network in the face of image corruption?

Hypothesis: separating luminance from colour (as CIELAB does) forces the
network to learn structure/shape rather than colour → making it more robust to
illumination artefacts (brightness, contrast, fog).


2. Type of ML problem


Supervised multi-class classification (10 classes, CIFAR-10).
Unstructured data (32×32×3 images).
The actual subject of study = robustness / domain generalisation, not just
classification. Classification is the tool; robustness is what is measured.



3. Technical stack


Language: Python 3
Main framework: PyTorch (NOT TensorFlow)
Colour conversions: scikit-image (skimage.color) — rgb2hsv, rgb2lab, rgb2gray
Model: ResNet-18 (via torchvision.models)
Datasets: CIFAR-10 (train/test), CIFAR-10-C (robustness, 15 corruptions × 5 severity levels)
Experiment tracking: wandb
Editor: VS Code + Claude Code
Computation: university GPU server tiny-delli01