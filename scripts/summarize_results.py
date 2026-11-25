"""Collect every runs/*/metrics.json into the EVALUATIONS folder.

  python scripts/summarize_results.py

Writes:
  evaluations/runs/<name>/{metrics.json,run_record.json}   copies that are safe to commit
  evaluations/results_summary.csv / results_summary.md      one row per run
  evaluations/figures/ (if matplotlib is installed):
    ex_by_system.png         EX of every run
    ex_by_difficulty.png     EX on easy / medium / hard / extra
    accuracy_vs_latency.png  EX vs seconds per question (RQ3: what voting + repair cost)
    error_types.png          wrong predictions per error category, from scripts/error_analysis.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

RUNS = Path("runs")
OUT = Path("evaluations")
DIFF = ["easy", "medium", "hard", "extra"]          # Spider
BIRD_DIFF = ["simple", "moderate", "challenging"]  # BIRD


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
            **{f"EX_{d}": pct(exd.get(d)) for d in DIFF + BIRD_DIFF},
            "VES": f"{m['ves']:.3f}" if "ves" in m else "",
            "latency_s": f"{m.get('avg_latency_s', 0):.3f}",
            "model_calls": f"{m.get('avg_model_calls', 0):.2f}",
        })
    if not rows:
        print("No runs/*/metrics.json found. Run scripts/run_eval.py first.")
        return
    # drop columns no run uses (e.g. BIRD columns when only Spider was run)
    empty = [k for k in rows[0] if k.startswith(("EX_", "VES")) and not any(r[k] for r in rows)]
    rows = [{k: v for k, v in r.items() if k not in empty} for r in rows]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    md = ["# Results summary", "",
          "EX / validity / EX_* in %, VES (BIRD) as a ratio, latency in seconds per question.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md += _selection_section()
    (OUT / "results_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    _plots(rows)


def _selection_section() -> list[str]:
    """Pipeline runs: which selection rule decided, and how many repairs were used (RQ3)."""
    rules = ["unique", "non_empty", "confidence", "simplicity", "greedy", "all_empty", "all_failed"]
    lines = []
    for mpath in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        if "decided_by" not in m:
            continue
        n = max(m.get("n", 0), 1)
        dec = m["decided_by"]
        rep = m.get("repairs_used", {})
        lines.append("| " + " | ".join([mpath.parent.name]
                                       + [pct(dec.get(r, 0) / n) for r in rules]
                                       + [pct(1 - rep.get("0", 0) / n)]) + " |")
    if not lines:
        return []
    head = ["", "## Candidate selection (pipeline runs)", "",
            "Share of questions (%) decided by each rule; last column = questions that used at least one repair.", "",
            "| run | " + " | ".join(rules) + " | used repair |", "|" + "---|" * (len(rules) + 2)]
    return head + lines


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

    diff_rows = [r for r in rows if any(r.get(f"EX_{d}") for d in DIFF)]
    if diff_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in diff_rows:
            ax.plot(DIFF, [float(r.get(f"EX_{d}") or 0) for d in DIFF], marker="o", label=r["run"])
        ax.set_ylabel("Execution accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / "ex_by_difficulty.png", dpi=200)
        plt.close(fig)
    _plot_accuracy_vs_latency(plt, rows, figs)
    _plot_error_types(plt, figs)
    print(f"Figures in {figs}/")


def _plot_accuracy_vs_latency(plt, rows: list[dict], figs: Path) -> None:
    pts = [r for r in rows if r["system"] in ("llm", "pipeline")]  # systems with model latency
    if not pts:
        return
    colors = {"heuristic": "#8C8C8C", "llm": "#DD8452", "pipeline": "#4C72B0"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in pts:
        ax.scatter(float(r["latency_s"]), float(r["EX"]), s=40,
                   color=colors.get(r["system"], "#55A868"), zorder=3)
        label = r["run"] if r["system"] != "pipeline" or r["N"] == "" else f"{r['run']} (N={r['N']}, R={r['R']})"
        ax.annotate(label, (float(r["latency_s"]), float(r["EX"])), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    for system, color in colors.items():
        if any(r["system"] == system for r in pts):
            ax.scatter([], [], color=color, label=system)
    ax.set_xlim(left=0)
    ax.set_xlabel("Seconds per question")
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "accuracy_vs_latency.png", dpi=200)
    plt.close(fig)


def _plot_error_types(plt, figs: Path) -> None:
    files = sorted((OUT / "error_analysis").glob("*/error_counts.json"))
    if not files:
        return
    data = [json.loads(f.read_text()) for f in files]
    cats = [c for c in data[0]["counts"] if any(d["counts"].get(c, 0) for d in data)]
    if not cats:
        return
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cats)), 4))
    for i, d in enumerate(data):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(cats))]
        ax.bar(xs, [d["counts"].get(c, 0) for c in cats], width,
               label=f"{d['run']} ({d['n_wrong']} wrong / {d['n_evaluated']})")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace("_", " ") for c in cats])
    ax.set_ylabel("Wrong predictions")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "error_types.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
