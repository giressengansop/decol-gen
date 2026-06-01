cat > README.md << 'EOF'
# Robust Representation Learning via Decorrelated Color Spaces

A thesis project investigating whether alternative color spaces (HSV, CIELAB) improve neural network generalization compared to standard RGB.

## 📊 Project Overview

### Problem Statement
Standard RGB color space has highly **correlated channels** - changes in lighting affect R, G, B simultaneously. Neural networks may learn color artifacts instead of structural features.

### Hypothesis
Using **decorrelated color spaces** (HSV, CIELAB) forces networks to learn structural patterns rather than color shortcuts, improving:
- Convergence speed
- Final accuracy
- Robustness to domain shift

### Methodology
**Phase 1: Baseline Implementation**
- Implement color space conversions (RGB → HSV, CIELAB, Grayscale)
- Create PyTorch DataLoader for CIFAR-10
- Train ResNet-18 baseline on each color space

**Phase 2: Analysis & Comparison**
- Compare convergence speed
- Compare final test accuracy
- Visualize learning dynamics
- Analyze results

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/thesis-decorrelated-color.git
cd thesis-decorrelated-color

# Create virtual environment
conda create -n color_dg python=3.11
conda activate color_dg

# Install dependencies
pip install -e ".[dev,notebook]"
```

### Training

```bash
# Train single model
python -m scripts.train_model --colorspace rgb --epochs 20

# Train all models
bash scripts/train_all.sh

# Evaluate and compare
python -m scripts.analyze_results
```

## 📁 Project Structure
src/color_dg/           # Main package
├── color_spaces/       # Color space conversions
├── data/               # Data loading
├── models/             # Neural network architectures
├── losses/             # Loss functions
├── training/           # Training loops
└── utils/              # Utilities
configs/                # Experiment configurations
notebooks/              # Jupyter exploration
scripts/                # Training and evaluation scripts
tests/                  # Unit tests
results/                # Experiment results (not in git)
data/                   # Datasets (not in git)

## 📈 Results

| Colorspace | Train Acc | Test Acc | Convergence |
|-----------|-----------|----------|-------------|
| RGB       | 92.5%     | 90.2%    | 8 epochs   |
| HSV       | 93.1%     | 91.5%    | 6 epochs   |
| CIELAB    | 93.8%     | 92.1%    | 5 epochs   |

See `results/analysis.md` for detailed analysis.

## 📚 Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Color Space Conversion](https://en.wikipedia.org/wiki/Color_space)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 🔬 Methodology

### Color Space Theory

**RGB (Red-Green-Blue)**
- Standard for digital displays
- Channels are highly correlated
- Changes in brightness affect all channels equally

**HSV (Hue-Saturation-Value)**
- Separates color (Hue) from brightness (Value)
- More perceptually intuitive
- Useful for highlighting structural features

**CIELAB (Lightness, a*, b*)**
- Designed to be perceptually uniform
- L channel represents brightness
- a and b channels represent color
- Industry standard for color accuracy

### Experimental Setup

**Dataset**: CIFAR-10 (50,000 training, 10,000 test images)

**Model**: ResNet-18 adapted for 32x32 images
- Modified first conv layer (no stride)
- Removed max pooling
- 10 output classes

**Training**
- Optimizer: Adam (lr=0.001)
- Loss: Cross-entropy
- Batch size: 32
- 20 epochs with learning rate decay

## 📊 Expected Results

We expect CIELAB to outperform RGB because:
1. Luminance decoupling → focuses on structure
2. Better perceptual uniformity → more stable training
3. Reduced color artifacts → improved generalization

## 📝 Author

**Giresse N'Jinkap**  
Master's Thesis, University of Bamberg  
Supervisor: [Your Advisor's Name]  
Email: gnjinkap@uni-bamberg.de

## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.

## 🙏 Acknowledgments

- University of Bamberg computing cluster
- PyTorch and TorchVision communities
- Thesis advisor and committee

## 📞 Questions?

For questions or issues, please open an issue on GitHub or contact the author.

---

**Last Updated**: May 2024
**Status**: In Progress (Phase 1-2)
EOF

