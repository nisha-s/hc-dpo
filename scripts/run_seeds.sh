#!/usr/bin/env bash
# Three seeds (42, 43, 44) of the best DPO beta and of HC-DPO at the matching KL budget.
#
#   bash scripts/run_seeds.sh <beta> <epsilon>
#
# <beta>:    the DPO sweep run with the highest mean reward-model score.
# <epsilon>: that run's final evaluation KL (kl_mean in its metrics.json), so the two
#            methods are compared at matched KL.
# Seed 42 runs that already exist from the sweeps are skipped.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BETA="${1:?usage: bash scripts/run_seeds.sh <beta> <epsilon>}"
EPSILON="${2:?usage: bash scripts/run_seeds.sh <beta> <epsilon>}"

for SEED in 42 43 44; do
  echo "=== DPO beta=$BETA seed=$SEED ==="
  python -m hcdpo.train --config configs/dpo.yaml --set beta="$BETA" seed="$SEED"
  python -m hcdpo.evaluate --config configs/dpo.yaml --set beta="$BETA" seed="$SEED"
  echo "=== HC-DPO epsilon=$EPSILON seed=$SEED ==="
  python -m hcdpo.train --config configs/hcdpo.yaml --set epsilon="$EPSILON" seed="$SEED"
  python -m hcdpo.evaluate --config configs/hcdpo.yaml --set epsilon="$EPSILON" seed="$SEED"
done
