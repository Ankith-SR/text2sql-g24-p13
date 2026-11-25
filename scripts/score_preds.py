"""Score predictions made outside this repo (e.g. LGESQL, Tier 2) with the same metrics, so they
land in the same results table as every other system.

  python scripts/score_preds.py --preds lgesql_dev.txt --name lgesql --system lgesql
  python scripts/score_preds.py --preds preds.json --name flan-t5 --system seq2seq

--preds accepts either
  * a JSON {qid: sql} with qids like "spider-dev-0" (what run_eval.py writes), or
  * a text file with one SQL per line in dev-set order (what the official Spider scripts and
    most public text2sql repos, including LGESQL, write).
Writes runs/<name>/{preds.json,metrics.json,run_record.json}; then `make summary`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import load_split, score  # noqa: E402

from text2sql.utils.run_record import write_run_record  # noqa: E402


def read_preds(path: Path, examples) -> dict[str, str]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    lines = [ln.split("\t")[0].strip() for ln in text.splitlines()]
    if len(lines) != len(examples):
        sys.exit(f"{path} has {len(lines)} lines but the dev set has {len(examples)} questions; "
                 "the text format needs one prediction per question, in order.")
    return {ex.qid: sql for ex, sql in zip(examples, lines)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True, help="run name, e.g. lgesql")
    ap.add_argument("--system", default="external", help="label in the results table")
    ap.add_argument("--dataset", choices=["spider", "bird"], default="spider")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--note", default="", help="free text stored in run_record.json")
    args = ap.parse_args()

    examples, _ = load_split(args.dataset, args.split)
    preds = read_preds(Path(args.preds), examples)
    metrics = score(preds, examples, args.dataset)
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds.json").write_text(json.dumps(preds, indent=2))
    write_run_record(out, {"system": args.system, "dataset": args.dataset, "split": args.split,
                           "model": args.name, "source": str(args.preds), "note": args.note},
                     metrics=metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
