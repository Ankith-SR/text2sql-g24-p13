"""Collect every runs/*/metrics.json into the EVALUATIONS folder.

  python scripts/summarize_results.py

Writes:
  evaluations/runs/<name>/{metrics.json,run_record.json}   copies that are safe to commit
  evaluations/results_summary.csv / results_summary.md      one row per run
  evaluations/figures/ex_by_system.png, ex_by_difficulty.png (if matplotlib is installed)
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

RUNS = Path("runs")
OUT = Path("evaluations")
DIFF = ["easy", "medium", "hard", "extra"]


def pct(x) -> str:
    return f"{100 * x:.1f}" if isinstance(x, (int, float)) else ""


def main() -> None:
    rows = []
    for mpath in sorted(RUNS.glob("*/metrics.json")):
        run = mpath.parent
        m = json.loads(mpath.read_text())
        rec_path = run / "run_record.json"
        cfg = json.loads(rec_path.read_text()).get("config", {}) if rec_path.exists() else {}
        dst = OUT / "runs" / run.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(mpath, dst / "metrics.json")
        if rec_path.exists():
            shutil.copy(rec_path, dst / "run_record.json")
        exd = m.get("ex_by_difficulty", {})
        rows.append({
            "run": run.name, "system": cfg.get("system", ""), "dataset": cfg.get("dataset", ""),
            "model": cfg.get("model") or "", "shots": cfg.get("shots", 0) or 0,
            "prune": cfg.get("prune", ""), "N": cfg.get("n_candidates", ""),
            "R": cfg.get("max_repairs", ""), "n": m.get("n", 0),
            "EX": pct(m.get("execution_accuracy")), "validity": pct(m.get("validity_rate")),
            **{f"EX_{d}": pct(exd.get(d)) for d in DIFF},
            "latency_s": f"{m.get('avg_latency_s', 0):.2f}",
            "model_calls": f"{m.get('avg_model_calls', 0):.2f}",
        })
    if not rows:
        print("No runs/*/metrics.json found. Run scripts/run_eval.py first.")
        return
    OUT.mkdir(exist_ok=True)
    with open(OUT / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    md = ["# Results summary", "", "EX / validity / EX_* in %, latency in seconds per question.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (OUT / "results_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    _plots(rows)


def _plots(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipping figures)")
        return
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r["EX"]]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(rows)), 4))
    ax.bar([r["run"] for r in rows], [float(r["EX"]) for r in rows], color="#4C72B0")
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "ex_by_system.png", dpi=200)
    plt.close(fig)

    diff_rows = [r for r in rows if any(r[f"EX_{d}"] for d in DIFF)]
    if diff_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in diff_rows:
            ax.plot(DIFF, [float(r[f"EX_{d}"] or 0) for d in DIFF], marker="o", label=r["run"])
        ax.set_ylabel("Execution accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / "ex_by_difficulty.png", dpi=200)
        plt.close(fig)
    print(f"Figures in {figs}/")


if __name__ == "__main__":
    main()
