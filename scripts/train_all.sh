#!/bin/bash

echo "Training all models..."
python -m scripts.train_model --config configs/baseline_rgb.yaml
python -m scripts.train_model --config configs/exp_hsv.yaml
python -m scripts.train_model --config configs/exp_lab.yaml
echo "Done!"
