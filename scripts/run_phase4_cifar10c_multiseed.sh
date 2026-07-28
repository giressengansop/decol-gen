#!/bin/bash
# Phase 4 - CIFAR-10-C re-evaluation of the multi-seed checkpoints in results_v4/.
# Usage: ./scripts/run_phase4_cifar10c_multiseed.sh [seed1 seed2 ...]
set -e

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 0 1 2 3)
fi

for seed in "${SEEDS[@]}"; do
    out="results_v4/cifar10c_seed${seed}"
    if [ -f "${out}/summary.json" ]; then
        echo "=== seed=${seed} -> ${out} (already done, skipping) ==="
        continue
    fi
    echo "=== CIFAR-10-C eval seed=${seed} -> ${out} ==="
    python -u -m scripts.eval_cifar10c \
        --cifar10c_root data/CIFAR-10-C \
        --results_dir results_v4 \
        --seed "${seed}" \
        --output_dir "${out}"
done
