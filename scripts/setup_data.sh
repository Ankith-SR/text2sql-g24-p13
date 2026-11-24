#!/usr/bin/env bash
# Datasets are too big for git. Each teammate downloads them once.
# Spider and BIRD are distributed from their official sites (links may change; check them):
#   Spider 1.0 : https://yale-lily.github.io/spider
#   BIRD       : https://bird-bench.github.io/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/spider data/bird

cat <<MSG
1) Download Spider from https://yale-lily.github.io/spider and unzip so you have:
     data/spider/train_spider.json  dev.json  tables.json  database/<db>/<db>.sqlite
2) Download BIRD from https://bird-bench.github.io/ and unzip train/ and dev/ so you have:
     data/bird/train/train.json  train_tables.json  train_databases/
     data/bird/dev/dev.json      dev_tables.json    dev_databases/
   (BIRD is ~33 GB total; on the Sol cluster keep it in scratch and symlink it to data/bird.)
3) For the official Exact-Match and test-suite scripts, clone (outside this repo):
     git clone https://github.com/taoyds/spider.git
     git clone https://github.com/taoyds/test-suite-sql-eval.git
MSG

for f in data/spider/dev.json data/spider/tables.json; do
  [ -f "$f" ] && echo "OK  $f" || echo "MISSING $f"
done
