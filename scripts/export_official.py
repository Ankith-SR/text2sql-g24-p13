"""Export a run for the official Spider scripts: Exact-set Match (EM) and test-suite accuracy.

  python scripts/export_official.py --run runs/pipeline-spider-qwen2.5-coder_7b
  # then, in a clone of https://github.com/taoyds/test-suite-sql-eval :
  python evaluation.py --gold <repo>/runs/<name>/official/gold.txt --pred <repo>/runs/<name>/official/pred.txt \
      --db <repo>/data/spider/database --table <repo>/data/spider/tables.json --etype all
  #   --etype match  -> EM only;  --etype exec with --db database/ of test-suite DBs -> test-suite accuracy

Copy the printed table into evaluations/official/<name>.txt so it is committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    rec = json.loads((run / "run_record.json").read_text())["config"]
    if rec.get("dataset", "spider") != "spider":
        sys.exit("official EM / test-suite scripts are for Spider only")
    preds = json.loads((run / "preds.json").read_text())
    examples, _ = load_split("spider", rec.get("split", "dev"))
    examples = [ex for ex in examples if ex.qid in preds]
    out = run / "official"
    out.mkdir(exist_ok=True)
    (out / "pred.txt").write_text("\n".join((preds[ex.qid] or "SELECT 1").replace("\n", " ")
                                            for ex in examples) + "\n")
    (out / "gold.txt").write_text("\n".join(f"{ex.gold_sql}\t{ex.db_id}" for ex in examples) + "\n")
    print(f"Wrote {len(examples)} lines to {out}/pred.txt and gold.txt")


if __name__ == "__main__":
    main()
