#!/bin/bash
# Heuristic (Tier 1) + Voyager-hosted Tier 3 baselines on one dataset. No GPU: the LLM runs on
# ASU Voyager (VOYAGER_API_KEY in .env).
#
#   DATASET=bird sbatch scripts/slurm/cpu_baselines.sbatch          (from the repo root)
#SBATCH --job-name=t2s-cpu-baselines
#SBATCH --partition=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
set -a; [ -f .env ] && source .env; set +a
DATASET=${DATASET:-spider}
MODELS=(${MODELS:-qwen3-coder-30b-a3b-instruct llama4-scout-17b})
SHOTS=(${SHOTS:-0 3})
source .venv/bin/activate

echo "--- heuristic $DATASET start $(date)"
python scripts/run_eval.py --system heuristic --dataset "$DATASET" 2>&1 | grep -v 'it/s\]$' | tail -30
echo "--- heuristic exit ${PIPESTATUS[0]} $(date)"

for m in "${MODELS[@]}"; do
  for s in "${SHOTS[@]}"; do
    echo "--- $m ${s}-shot $DATASET start $(date)"
    python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml \
        --model "$m" --set backend=voyager dataset="$DATASET" shots="$s" 2>&1 | grep -v 'it/s\]$' | tail -30
    echo "--- $m ${s}-shot exit ${PIPESTATUS[0]} $(date)"
  done
done
echo "done $(date)"
