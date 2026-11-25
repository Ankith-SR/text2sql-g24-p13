"""Stage 4 ablations: switch one pipeline component at a time, sweep N and R.

  python scripts/run_ablations.py --limit 200               # core table (6 runs)
  python scripts/run_ablations.py --limit 200 --sweep       # + N in {1,3,5,8}, R in {0,1,2,3}
  python scripts/run_ablations.py --dry-run                 # print the commands only

Each run lands in runs/abl-<name>/. Then: python scripts/summarize_results.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE = {
    # name: overrides on top of configs/pipeline.yaml
    "full": [],
    "no-prune": ["prune=false"],
    "single-greedy": ["n_candidates=1", "max_repairs=0"],   # one query, no vote, no repair
    "vote-no-repair": ["max_repairs=0"],
    "repair-no-vote": ["n_candidates=1"],
    "full-3shot": ["shots=3"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sweep", action="store_true", help="also sweep N and R")
    ap.add_argument("--bird-values", action="store_true", help="add BIRD +/- value retrieval")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = dict(CORE)
    if args.sweep:
        for n in (1, 3, 5, 8):
            runs[f"N{n}-R2"] = [f"n_candidates={n}", "max_repairs=2"]
        for r in (0, 1, 3):
            runs[f"N5-R{r}"] = ["n_candidates=5", f"max_repairs={r}"]
    if args.bird_values:
        runs["bird-values"] = ["dataset=bird", "values=true"]
        runs["bird-no-values"] = ["dataset=bird", "values=false"]

    for name, sets in runs.items():
        cmd = [sys.executable, "scripts/run_eval.py", "--system", "pipeline", "--config", args.config,
               "--limit", str(args.limit), "--name", f"abl-{name}"]
        if sets:
            cmd += ["--set", *sets]
        print("$", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
