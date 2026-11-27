#!/bin/bash
# Runs on models served locally by vLLM (the same HF weights the QLoRA comparison uses).
# One GPU; each model is served in turn, then every dataset x shot setting runs against it.
#
#   sbatch scripts/slurm/baselines_vllm.sbatch                         Tier 3 baselines, Spider
#   DATASETS="spider bird" sbatch scripts/slurm/baselines_vllm.sbatch  ... on both datasets
#   SYSTEM=pipeline CONFIG=configs/pipeline.yaml SHOTS=0 \
#     MODELS=Qwen/Qwen2.5-Coder-7B-Instruct DATASETS="spider bird" \
#     sbatch scripts/slurm/baselines_vllm.sbatch                       our pipeline, untuned base
#   (run from the repo root; LIMIT=200 for a partial run)
#
# Needs: .venv set up (requirements.txt), data in data/spider (and data/bird), and for Llama an
# HF_TOKEN in .env from an account with access to Meta's Llama 3.1 repo. vLLM is installed on
# first run into $VLLM_ENV (separate from .venv because it pins its own torch).
#SBATCH --job-name=t2s-vllm
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
set -a; [ -f .env ] && source .env; set +a

SYSTEM=${SYSTEM:-llm}
CONFIG=${CONFIG:-configs/baseline_llm.yaml}
DATASETS=(${DATASETS:-${DATASET:-spider}})
MODELS=(${MODELS:-Qwen/Qwen2.5-Coder-7B-Instruct meta-llama/Llama-3.1-8B-Instruct})
SHOTS=(${SHOTS:-0 3})
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
PORT=${PORT:-8000}
VLLM_ENV=${VLLM_ENV:-/scratch/$USER/vllm-env}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export HF_TOKEN=${HF_TOKEN:-}

echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "system=$SYSTEM config=$CONFIG datasets=${DATASETS[*]} models=${MODELS[*]} shots=${SHOTS[*]} ${LIMIT_ARG}"

if [ ! -x "$VLLM_ENV/bin/vllm" ]; then
  echo "installing vLLM into $VLLM_ENV"
  /usr/bin/python3.12 -m venv "$VLLM_ENV" && "$VLLM_ENV/bin/pip" install -q --upgrade pip vllm || exit 1
fi
"$VLLM_ENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

wait_ready() {  # up to 30 min: the first run downloads ~16 GB of weights
  for _ in $(seq 360); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

source .venv/bin/activate
for m in "${MODELS[@]}"; do
  if [[ "$m" == meta-llama/* && -z "$HF_TOKEN" ]]; then
    echo "SKIP $m: no HF_TOKEN in .env (Llama is gated)"; continue
  fi
  vlog="logs/vllm_${SLURM_JOB_ID:-local}_${m//\//_}.log"
  echo "=== serving $m  $(date)"
  "$VLLM_ENV/bin/vllm" serve "$m" --port "$PORT" --dtype bfloat16 --max-model-len 16384 \
      --gpu-memory-utilization 0.9 > "$vlog" 2>&1 &
  srv=$!
  if ! wait_ready "$srv"; then
    echo "FAILED to start $m, see $vlog"
    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; continue
  fi
  for d in "${DATASETS[@]}"; do
    for s in "${SHOTS[@]}"; do
      echo "--- $SYSTEM $m $d ${s}-shot start $(date)"
      python scripts/run_eval.py --system "$SYSTEM" --config "$CONFIG" $LIMIT_ARG \
          --model "$m" --set backend=vllm dataset="$d" shots="$s" 2>&1 | grep -v 'it/s\]$' | tail -40
      echo "--- $SYSTEM $m $d ${s}-shot exit ${PIPESTATUS[0]} $(date)"
    done
  done
  kill "$srv"; wait "$srv" 2>/dev/null
done
echo "done $(date)"
