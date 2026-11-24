"""Run one system on a dev set and write runs/<name>/{preds.json,metrics.json,run_record.json}.

Examples:
  python scripts/run_eval.py --system heuristic --dataset spider --limit 100
  python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml --model qwen2.5-coder:7b
  python scripts/run_eval.py --system pipeline --config configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.baselines import heuristic  # noqa: E402
from text2sql.baselines.llm_prompt import build_prompt, generate, make_client  # noqa: E402
from text2sql.data.loaders import load_bird, load_spider  # noqa: E402
from text2sql.data.schema import load_schemas  # noqa: E402
from text2sql.evaluation.execution import evaluate  # noqa: E402
from text2sql.pipeline.run_pipeline import answer  # noqa: E402
from text2sql.pruning.schema_pruning import prune, retrieve_values  # noqa: E402
from text2sql.utils.run_record import write_run_record  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["heuristic", "llm", "pipeline"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", choices=["spider", "bird"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first K examples (smoke test)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    dataset = args.dataset or cfg.get("dataset", "spider")
    model = args.model or cfg.get("model") or (cfg.get("models") or [None])[0]
    split = cfg.get("split", "dev")

    root = Path("data") / dataset
    if dataset == "spider":
        examples = load_spider(root, split)
        schemas = load_schemas(root / "tables.json")
    else:
        examples = load_bird(root, split)
        schemas = load_schemas(root / split / f"{split}_tables.json")
    if args.limit:
        examples = examples[: args.limit]

    client = make_client(cfg.get("backend", "ollama")) if args.system != "heuristic" else None
    use_prune = cfg.get("prune", args.system == "pipeline")
    preds: dict[str, str] = {}

    for ex in tqdm(examples, desc=f"{args.system}/{dataset}"):
        sch = schemas[ex.db_id]
        if args.system == "heuristic":
            preds[ex.qid] = heuristic.predict(ex.question, sch)
            continue
        keep = prune(ex.question, sch) if use_prune else None
        schema_text = sch.to_prompt(keep)
        hints = ex.evidence
        if cfg.get("values") and dataset == "bird":
            hints += " " + "; ".join(retrieve_values(ex.question, ex.db_path, sch))
        prompt = build_prompt(schema_text, ex.question, hints)

        def gen(p, n, t):
            return generate(client, model, p, n=n, temperature=t)

        if args.system == "llm":
            preds[ex.qid] = gen(prompt, 1, 0.0)[0][0]
        else:
            preds[ex.qid], _, _ = answer(prompt, ex.db_path, gen, n=cfg.get("n_candidates", 5),
                                         r=cfg.get("max_repairs", 2),
                                         temperature=cfg.get("temperature", 0.7))

    metrics = evaluate(preds, examples)
    name = args.name or f"{args.system}-{dataset}-{(model or 'none').replace(':', '_')}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds.json").write_text(json.dumps(preds, indent=2))
    write_run_record(out, {"system": args.system, "dataset": dataset, "model": model, **cfg},
                     metrics=metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
