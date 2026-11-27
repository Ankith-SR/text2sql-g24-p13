#!/bin/bash
# Download BIRD into data/bird/{dev,train}/ in the layout loaders.py expects:
#   dev/dev.json  dev/dev_tables.json  dev/dev_databases/<db>/<db>.sqlite
#   train/train.json  train/train_tables.json   (train databases only if BIRD_TRAIN_DBS=1)
# Train JSON is the few-shot pool for 3-shot runs; train databases (~30 GB unpacked) are only
# needed for QLoRA training on BIRD.
#
#   sbatch scripts/slurm/download_bird.sbatch            (from the repo root)
#SBATCH --job-name=t2s-bird-download
#SBATCH --partition=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
BASE=https://bird-bench.oss-cn-beijing.aliyuncs.com
mkdir -p data/bird/_zips && cd data/bird

for s in dev train; do
  echo "=== $s: download $(date)"
  curl -fL --retry 5 -C - -o "_zips/$s.zip" "$BASE/$s.zip"
  tmp="_tmp_$s"; rm -rf "$tmp" && mkdir "$tmp"
  unzip -q "_zips/$s.zip" -d "$tmp" -x '__MACOSX/*' '*.DS_Store'
  mkdir -p "$s"
  for f in "$s.json" "${s}_tables.json"; do
    cp "$(find "$tmp" -name "$f" -not -path '*/__MACOSX/*' | head -1)" "$s/$f"
  done
  if [ "$s" = dev ] || [ "${BIRD_TRAIN_DBS:-0}" = 1 ]; then
    dbzip=$(find "$tmp" -name "${s}_databases.zip" | head -1)
    if [ -n "$dbzip" ]; then
      unzip -q "$dbzip" -d "$s" -x '__MACOSX/*' '*.DS_Store'
    else
      mv "$(find "$tmp" -type d -name "${s}_databases" | head -1)" "$s/"
    fi
  fi
  rm -rf "$tmp"
  echo "=== $s: $(python3 -c "import json;print(len(json.load(open('$s/$s.json'))))") questions"
done

n=$(ls dev/dev_databases | wc -l)
echo "dev databases: $n"
[ "$n" -gt 0 ] && [ -f dev/dev.json ] && [ -f train/train.json ]
echo "done $(date)"
