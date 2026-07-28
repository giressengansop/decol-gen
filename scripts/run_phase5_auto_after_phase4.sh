#!/bin/bash
# Waits for the Phase 4 CIFAR-10-C re-evaluation (seeds 42,0,1,2,3) to finish,
# then automatically launches the 12 Phase 5 normalization-ablation training
# runs followed by their CIFAR-10-C re-evaluation.
set -u

WAIT_TARGET="results_v4/cifar10c_seed3/summary.json"
MAX_WAIT_ITERS=360   # 360 * 30s = 3 hours safety cap
i=0

echo "$(date '+%F %T') Waiting for ${WAIT_TARGET} ..."
while [ ! -f "${WAIT_TARGET}" ]; do
    i=$((i+1))
    if [ "${i}" -ge "${MAX_WAIT_ITERS}" ]; then
        echo "$(date '+%F %T') ERROR: timed out waiting for ${WAIT_TARGET} after $((MAX_WAIT_ITERS*30/60)) minutes."
        exit 1
    fi
    sleep 30
done
echo "$(date '+%F %T') Phase 4 CIFAR-10-C re-evaluation done. Starting Phase 5."

echo "$(date '+%F %T') === Phase 5: training (12 runs) ==="
./scripts/run_phase5_normalization.sh
train_status=$?
if [ "${train_status}" -ne 0 ]; then
    echo "$(date '+%F %T') ERROR: run_phase5_normalization.sh exited with ${train_status}"
    exit "${train_status}"
fi

echo "$(date '+%F %T') === Phase 5: CIFAR-10-C re-evaluation ==="
./scripts/run_phase5_cifar10c.sh
eval_status=$?
if [ "${eval_status}" -ne 0 ]; then
    echo "$(date '+%F %T') ERROR: run_phase5_cifar10c.sh exited with ${eval_status}"
    exit "${eval_status}"
fi

echo "$(date '+%F %T') Phase 5 fully done (training + CIFAR-10-C re-evaluation)."
