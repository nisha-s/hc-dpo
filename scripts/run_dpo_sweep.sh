#!/usr/bin/env bash
# Standard DPO baseline sweep: beta in {0.05, 0.1, 0.25, 0.5}, one seed.
#
#   bash scripts/run_dpo_sweep.sh                                  # defaults
#   bash scripts/run_dpo_sweep.sh batch_size=4 grad_accum_steps=8  # extra key=value overrides
#
# Safe to rerun after a disconnect: finished runs are skipped and an interrupted
# run resumes from its latest checkpoint.
set -euo pipefail

# Lets PyTorch reuse freed GPU memory for differently sized batches instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/dpo.yaml}"
SEED="${SEED:-42}"

python -m hcdpo.data --config "$CONFIG" --set "$@"
python -m hcdpo.evaluate --config "$CONFIG" --base --set "$@"

for BETA in 0.05 0.1 0.25 0.5; do
  echo "=== DPO beta=$BETA seed=$SEED ==="
  python -m hcdpo.train --config "$CONFIG" --set beta="$BETA" seed="$SEED" "$@"
  python -m hcdpo.evaluate --config "$CONFIG" --set beta="$BETA" seed="$SEED" "$@"
done
