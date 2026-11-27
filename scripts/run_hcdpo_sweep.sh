#!/usr/bin/env bash
# HC-DPO sweep over four KL budgets (epsilon, in nats per response), one seed.
# Choose the epsilons to span the final KL range of the DPO beta sweep.
#
#   EPSILONS="1 2 4 8" bash scripts/run_hcdpo_sweep.sh
#   EPSILONS="1 2 4 8" bash scripts/run_hcdpo_sweep.sh dual_lr=0.1   # extra key=value overrides
#
# Run the DPO sweep first (it prepares the data and evaluates the base model).
# Safe to rerun after a disconnect: finished runs are skipped, interrupted runs resume.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/hcdpo.yaml}"
SEED="${SEED:-42}"
EPSILONS="${EPSILONS:?Set EPSILONS to four KL budgets, e.g. EPSILONS=\"1 2 4 8\"}"

for EPSILON in $EPSILONS; do
  echo "=== HC-DPO epsilon=$EPSILON seed=$SEED ==="
  python -m hcdpo.train --config "$CONFIG" --set epsilon="$EPSILON" seed="$SEED" "$@"
  python -m hcdpo.evaluate --config "$CONFIG" --set epsilon="$EPSILON" seed="$SEED" "$@"
done
