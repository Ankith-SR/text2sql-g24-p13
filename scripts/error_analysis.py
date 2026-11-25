"""Stage 2 error analysis: label every wrong prediction of a run (owner: Hui Yu).

  python scripts/error_analysis.py --run runs/llm-spider-qwen2.5-coder_7b

Writes evaluations/error_analysis/<run>/errors.csv (one row per wrong prediction, with an
auto_label and an empty human_label column for review) and error_counts.json (per category,
and per category x Spider difficulty).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.analysis.error_taxonomy import CATEGORIES, label, summarize  # noqa: E402
from text2sql.evaluation.execution import execute, same_result  # noqa: E402
from text2sql.evaluation.hardness import hardness  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> folder with preds.json")
    ap.add_argument("--out", default="evaluations/error_analysis")
    args = ap.parse_args()

    run = Path(args.run)
    rec = json.loads((run / "run_record.json").read_text())
    dataset = rec["config"].get("dataset", "spider")
    split = rec["config"].get("split", "dev")
    preds = json.loads((run / "preds.json").read_text())
    examples, schemas = load_split(dataset, split)

    rows, by_diff = [], Counter()
    n = 0
    for ex in examples:
        if ex.qid not in preds:
            continue
        n += 1
        pred_res = execute(ex.db_path, preds[ex.qid])
        gold_res = execute(ex.db_path, ex.gold_sql)
        if same_result(pred_res, gold_res, ex.gold_sql):
            continue
        lab = label(preds[ex.qid], ex.gold_sql, pred_res, schemas[ex.db_id], dataset == "bird")
        rows.append({"qid": ex.qid, "question": ex.question, "gold": ex.gold_sql,
                     "pred": preds[ex.qid], "auto_label": lab})
        if dataset == "spider":
            by_diff[f"{lab}|{hardness(ex.gold_sql)}"] += 1

    out = Path(args.out) / run.name
    out.mkdir(parents=True, exist_ok=True)
    counts = summarize(rows, out / "errors.csv")
    result = {
        "run": run.name, "n_evaluated": n, "n_wrong": len(rows),
        "counts": {c: counts.get(c, 0) for c in CATEGORIES},
        "share": {c: round(counts.get(c, 0) / len(rows), 3) if rows else 0.0 for c in CATEGORIES},
        "by_category_and_difficulty": dict(sorted(by_diff.items())),
    }
    (out / "error_counts.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nReview {out/'errors.csv'}: fill human_label where the auto label is wrong.")


if __name__ == "__main__":
    main()
