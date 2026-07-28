#!/bin/bash
# Phase 5 - CIFAR-10-C re-evaluation of the normalization-ablation checkpoints.
# Usage: ./scripts/run_phase5_cifar10c.sh [seed1 seed2 ...]
set -e

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 0 1)
fi

NORMS=(minmax centered)

for norm in "${NORMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="results_v5/cifar10c_norm_${norm}_seed${seed}"
        if [ -f "${out}/summary.json" ]; then
            echo "=== norm=${norm} seed=${seed} -> ${out} (already done, skipping) ==="
            continue
        fi
        echo "=== CIFAR-10-C eval norm=${norm} seed=${seed} -> ${out} ==="
        python -u -m scripts.eval_cifar10c \
            --cifar10c_root data/CIFAR-10-C \
            --results_dir results_v5 \
            --subdir_suffix "_norm_${norm}" \
            --normalization "${norm}" \
            --seed "${seed}" \
            --output_dir "${out}"
    done
done
