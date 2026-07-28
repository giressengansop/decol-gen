#!/bin/bash
# Phase 4 - multi-seed verification runs.
# Usage: ./scripts/run_phase4_multiseed.sh [seed1 seed2 ...]
# Default: 5 seeds (42 0 1 2 3). Trim the list to 3 seeds if GPU time is tight
# (see docs/plan_recherche_suite.md, Phase 4).
set -e

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 0 1 2 3)
fi

EXPERIMENTS=(baseline_rgb exp_hsv exp_lab exp_grayscale)

for exp in "${EXPERIMENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="results_v4/${exp}_seed${seed}"
        if [ -f "${out}/summary.json" ]; then
            echo "=== ${exp} seed=${seed} -> ${out} (already done, skipping) ==="
            continue
        fi
        echo "=== ${exp} seed=${seed} -> ${out} ==="
        python -u -m scripts.train_model \
            --config "configs/${exp}.yaml" \
            --seed "${seed}" \
            --output_dir "${out}"
    done
done
