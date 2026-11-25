"""Run one system on a dev set and write runs/<name>/{preds.json,details.json,metrics.json,run_record.json}.

Examples:
  python scripts/run_eval.py --system heuristic --dataset spider
  python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml            # every model in the config
  python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml --set shots=3
  python scripts/run_eval.py --system pipeline --config configs/pipeline.yaml --limit 200
  python scripts/run_eval.py --system pipeline --config configs/pipeline.yaml --set n_candidates=1 --name abl-n1

`--set key=value` overrides any config key (used by scripts/run_ablations.py).
Metrics: execution accuracy, validity, EX by Spider difficulty, latency and model calls per question,
and (pipeline) how often each selection rule decided.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.baselines import heuristic  # noqa: E402
from text2sql.baselines.llm_prompt import (  # noqa: E402
    CallStats,
    ShotSelector,
    build_prompt,
    generate,
    make_client,
)
from text2sql.data.loaders import load_bird, load_spider  # noqa: E402
from text2sql.data.schema import load_schemas  # noqa: E402
from text2sql.evaluation.execution import evaluate  # noqa: E402
from text2sql.evaluation.hardness import hardness  # noqa: E402
from text2sql.pipeline.run_pipeline import answer_detailed  # noqa: E402
from text2sql.pruning.schema_pruning import prune, retrieve_values  # noqa: E402
from text2sql.utils.run_record import write_run_record  # noqa: E402


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k.strip()] = yaml.safe_load(v)  # "3" -> 3, "false" -> False, "0.7" -> 0.7
    return out


def load_split(dataset: str, split: str):
    root = Path("data") / dataset
    if dataset == "spider":
        return load_spider(root, split), load_schemas(root / "tables.json")
    return load_bird(root, split), load_schemas(root / split / f"{split}_tables.json")


def shot_selector(dataset: str, k: int) -> ShotSelector | None:
    """Few-shot pool = the train split of the same dataset (never dev)."""
    if k <= 0:
        return None
    train, _ = load_split(dataset, "train")
    return ShotSelector([(ex.question, ex.gold_sql) for ex in train])


def run_one(system: str, dataset: str, model: str | None, cfg: dict, limit: int | None,
            name: str | None) -> dict:
    split = cfg.get("split", "dev")
    examples, schemas = load_split(dataset, split)
    if limit:
        examples = examples[:limit]

    backend = cfg.get("backend", "ollama")
    client = make_client(backend) if system != "heuristic" else None
    use_prune = cfg.get("prune", system == "pipeline")
    k_shots = int(cfg.get("shots", 0) or 0)
    shots = shot_selector(dataset, k_shots) if system != "heuristic" else None
    stats = CallStats()
    preds: dict[str, str] = {}
    details: dict[str, dict] = {}
    latencies: list[float] = []
    calls: list[int] = []
    decided = Counter()
    repairs = Counter()

    for ex in tqdm(examples, desc=f"{system}/{dataset}/{model or '-'}"):
        sch = schemas[ex.db_id]
        t0 = time.perf_counter()
        stats.reset()
        if system == "heuristic":
            preds[ex.qid] = heuristic.predict(ex.question, sch)
            latencies.append(time.perf_counter() - t0)
            calls.append(0)
            continue
        keep = prune(ex.question, sch) if use_prune else None
        schema_text = sch.to_prompt(keep)
        hints = ex.evidence
        if cfg.get("values") and dataset == "bird":
            hints += " " + "; ".join(retrieve_values(ex.question, ex.db_path, sch))
        prompt = build_prompt(schema_text, ex.question, hints.strip(),
                              shots.select(ex.question, k_shots) if shots else None)

        def gen(p, n, t):
            return generate(client, model, p, n=n, temperature=t, stats=stats)

        try:
            if system == "llm":
                preds[ex.qid] = gen(prompt, 1, 0.0)[0][0]
            else:
                res = answer_detailed(prompt, ex.db_path, gen, n=int(cfg.get("n_candidates", 5)),
                                      r=int(cfg.get("max_repairs", 2)),
                                      temperature=float(cfg.get("temperature", 0.7)))
                preds[ex.qid] = res.sql
                decided[res.selection.decided_by] += 1
                repairs[res.repairs_used] += 1
                details[ex.qid] = {
                    "decided_by": res.selection.decided_by, "repairs_used": res.repairs_used,
                    "n_valid": res.selection.n_valid,
                    "candidates": [c.sql for c in res.candidates],
                }
        except Exception as e:  # noqa: BLE001  one bad call must not kill a 1,000-question run
            preds[ex.qid] = ""
            details.setdefault(ex.qid, {})["error"] = str(e)[:300]
        latencies.append(time.perf_counter() - t0)
        calls.append(stats.calls)

    difficulty = {ex.qid: hardness(ex.gold_sql) for ex in examples} if dataset == "spider" else None
    metrics = evaluate(preds, examples, difficulty)
    if difficulty:
        metrics["n_by_difficulty"] = dict(Counter(difficulty.values()))
    metrics["avg_latency_s"] = sum(latencies) / len(latencies) if latencies else 0.0
    metrics["avg_model_calls"] = sum(calls) / len(calls) if calls else 0.0
    metrics["n_errors"] = sum(1 for d in details.values() if "error" in d)
    if system == "pipeline":
        metrics["decided_by"] = dict(decided)
        metrics["repairs_used"] = {str(k): v for k, v in sorted(repairs.items())}

    tag = (model or "none").replace(":", "_").replace("/", "_")
    run_name = name or f"{system}-{dataset}-{tag}" + (f"-{k_shots}shot" if k_shots else "")
    out = Path("runs") / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds.json").write_text(json.dumps(preds, indent=2))
    if details:
        (out / "details.json").write_text(json.dumps(details, indent=2))
    config = {**cfg, "system": system, "dataset": dataset, "model": model, "limit": limit}
    config.pop("models", None)
    write_run_record(out, config, metrics=metrics)
    print(f"[{run_name}]", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["heuristic", "llm", "pipeline"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", choices=["spider", "bird"], default=None)
    ap.add_argument("--model", default=None, help="one model; default: every model in the config")
    ap.add_argument("--limit", type=int, default=None, help="first K examples (smoke test)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    args = ap.parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    cfg.update(parse_overrides(args.set))
    dataset = args.dataset or cfg.get("dataset", "spider")

    if args.system == "heuristic":
        models = [None]
    elif args.model:
        models = [args.model]
    else:
        models = cfg.get("models") or [cfg.get("model")]
    for m in models:
        name = args.name if len(models) == 1 else None
        run_one(args.system, dataset, m, cfg, args.limit, name)


if __name__ == "__main__":
    main()
