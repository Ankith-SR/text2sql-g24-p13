"""Execution-based evaluation (owner: Hui Yu).

EX  : predicted and gold SQL return the same result set.
Validity : predicted SQL parses and executes without error.
Exact set match (EM) and test-suite accuracy use the official scripts; export a run with
scripts/export_official.py (instructions in its docstring and docs/EVALUATION.md).
Difficulty levels for "EX by difficulty" come from evaluation/hardness.py.
"""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

TIMEOUT_S = 30


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] | None = None
    error: str | None = None


def execute(db_path: str, sql: str, timeout: int = TIMEOUT_S) -> ExecResult:
    """Run SQL read-only. Runaway queries are interrupted after `timeout` seconds.

    Runs in-process with SQLite's progress handler instead of one subprocess per query:
    spawning a process costs ~0.5 s on macOS, which dominated latency when every question
    executes N candidates plus the gold query.
    """
    if not sql or not sql.strip():
        return ExecResult(False, error="empty query")
    deadline = time.monotonic() + timeout
    con = None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode(errors="ignore")
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        rows = con.execute(sql).fetchall()
        return ExecResult(True, rows)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e) and time.monotonic() > deadline:
            return ExecResult(False, error="timeout")
        return ExecResult(False, error=str(e))
    except Exception as e:  # noqa: BLE001
        return ExecResult(False, error=str(e))
    finally:
        if con is not None:
            con.close()


def result_key(rows: list[tuple], ordered: bool = False):
    """Hashable canonical form of a result set. Unordered unless the query orders."""
    return tuple(rows) if ordered else tuple(sorted(Counter(map(repr, rows)).items()))


def same_result(pred: ExecResult, gold: ExecResult, gold_sql: str) -> bool:
    if not (pred.ok and gold.ok):
        return False
    ordered = "order by" in gold_sql.lower()
    if ordered:
        return list(map(repr, pred.rows)) == list(map(repr, gold.rows))
    return result_key(pred.rows) == result_key(gold.rows)


def evaluate(preds: dict[str, str], examples, difficulty: dict[str, str] | None = None) -> dict:
    """preds: {qid: sql}. difficulty: optional {qid: easy|medium|hard|extra}."""
    n = ok_cnt = valid_cnt = 0
    by_diff: dict[str, list[int]] = {}
    for ex in examples:
        if ex.qid not in preds:
            continue
        n += 1
        pred = execute(ex.db_path, preds[ex.qid])
        gold = execute(ex.db_path, ex.gold_sql)
        valid_cnt += pred.ok
        hit = same_result(pred, gold, ex.gold_sql)
        ok_cnt += hit
        if difficulty and ex.qid in difficulty:
            by_diff.setdefault(difficulty[ex.qid], []).append(int(hit))
    return {
        "n": n,
        "execution_accuracy": ok_cnt / n if n else 0.0,
        "validity_rate": valid_cnt / n if n else 0.0,
        "ex_by_difficulty": {k: sum(v) / len(v) for k, v in by_diff.items()},
    }
