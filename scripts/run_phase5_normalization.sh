#!/bin/bash
# Phase 5 - normalization ablation (RGB vs LAB only, minmax + centered).
# Z-score is not retrained here: it's the existing Phase 4 reference
# (results_v4/baseline_rgb_seed*, results_v4/exp_lab_seed*).
# Usage: ./scripts/run_phase5_normalization.sh [seed1 seed2 ...]
set -e

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 0 1)
fi

CONFIGS=(baseline_rgb_norm_minmax baseline_rgb_norm_centered exp_lab_norm_minmax exp_lab_norm_centered)

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="results_v5/${cfg}_seed${seed}"
        if [ -f "${out}/summary.json" ]; then
            echo "=== ${cfg} seed=${seed} -> ${out} (already done, skipping) ==="
            continue
        fi
        echo "=== ${cfg} seed=${seed} -> ${out} ==="
        python -u -m scripts.train_model \
            --config "configs/${cfg}.yaml" \
            --seed "${seed}" \
            --output_dir "${out}"
    done
done
