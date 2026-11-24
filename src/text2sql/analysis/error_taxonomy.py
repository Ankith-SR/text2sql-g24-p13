"""Stage 2 error analysis (owner: Hui Yu).

Auto-label wrong predictions with a first-pass category; humans then confirm/override in
the CSV. Categories match the proposal: schema_linking, join, nested_or_aggregation,
value_grounding (BIRD), syntax.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from text2sql.data.schema import Schema
from text2sql.evaluation.execution import ExecResult

CATEGORIES = ["syntax", "schema_linking", "join", "nested_or_aggregation", "value_grounding", "other"]


def _tables(sql: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", sql, re.I)}


def _agg(sql: str) -> bool:
    return bool(re.search(r"\b(group by|having|count\(|sum\(|avg\(|max\(|min\(|\(\s*select)", sql, re.I))


def label(pred_sql: str, gold_sql: str, pred_res: ExecResult, schema: Schema,
          is_bird: bool = False) -> str:
    if not pred_res.ok:
        err = (pred_res.error or "").lower()
        if "no such table" in err or "no such column" in err:
            return "schema_linking"
        return "syntax"
    pt, gt = _tables(pred_sql), _tables(gold_sql)
    known = {t.name.lower() for t in schema.tables}
    if not pt <= known or (pt != gt and len(pt) == len(gt)):
        return "schema_linking"
    if len(pt) != len(gt):
        return "join"
    if _agg(gold_sql) != _agg(pred_sql) or pred_sql.lower().count("select") != gold_sql.lower().count("select"):
        return "nested_or_aggregation"
    if is_bird:
        return "value_grounding"
    return "other"


def summarize(rows: list[dict], out_csv: str | Path) -> Counter:
    """rows: [{qid, question, gold, pred, auto_label}] -> CSV with a blank human_label column."""
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "question", "gold", "pred", "auto_label", "human_label"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "human_label": ""})
    return Counter(r["auto_label"] for r in rows)
