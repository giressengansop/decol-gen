#!/bin/bash
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate thesis
cd ~/decol-gen
mkdir -p results_v7

echo "[$(date +%H:%M)] env=$CONDA_DEFAULT_ENV"

for s in 0 1 42; do
  for c in baseline_rgb exp_lab exp_hsv exp_grayscale; do
    d="results_v7/${c}_gn_seed${s}"
    if [ -f "$d/summary.json" ]; then
      echo "[$(date +%H:%M)] SKIP $d"; continue
    fi
    echo "[$(date +%H:%M)] START $d"
    python -m scripts.train_model --config "configs/${c}_gn.yaml" \
        --seed "$s" --output_dir "$d"
  done
done

echo
echo "[$(date +%H:%M)] === Entraînements terminés ==="
python - << 'PY'
import json, glob, os
for p in sorted(glob.glob("results_v7/*_gn_seed*/summary.json")):
    d = json.load(open(p))
    print(f"{os.path.basename(os.path.dirname(p)):<28}{d['best_val_acc']*100:>8.2f}%")
PY
