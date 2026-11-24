"""Tier-1 baseline: keyword / schema-matching heuristic (owner: Hui Yu).

Deliberately simple and non-learned. Picks the table whose name/columns best match the
question, then handles a few patterns (count, max/min). Its job is to set a floor so we
can say how much each tier of system adds.
"""
from __future__ import annotations

import re

from text2sql.data.schema import Schema, Table

STOP = {"the", "a", "an", "of", "in", "for", "is", "are", "what", "which", "how", "many",
        "show", "list", "give", "me", "all", "with", "and", "to", "by", "that", "have",
        "has", "whose", "who", "than", "number", "total"}


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower().replace("_", " ")) if w not in STOP]


def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def _stems(s: str) -> set[str]:
    return {_stem(w) for w in _tokens(s)}


def score_table(question: str, t: Table) -> float:
    q = _stems(question)
    name_hits = len(q & _stems(t.name))
    col_hits = sum(1 for c in t.columns if q & _stems(c))
    return 2.0 * name_hits + 0.5 * col_hits


def best_table(question: str, schema: Schema) -> Table:
    return max(schema.tables, key=lambda t: score_table(question, t))


def best_columns(question: str, t: Table, limit: int = 2) -> list[str]:
    q = _stems(question)
    scored = [(len(q & _stems(c)), c) for c in t.columns]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


def predict(question: str, schema: Schema) -> str:
    q = question.lower()
    t = best_table(question, schema)
    cols = best_columns(question, t)
    if re.search(r"\b(how many|number of|count)\b", q):
        return f"SELECT COUNT(*) FROM {t.name}"
    most = re.search(r"\b(most|highest|largest|maximum|max|oldest)\b", q)
    least = re.search(r"\b(least|lowest|smallest|minimum|min|youngest)\b", q)
    if (most or least) and cols:
        order = "DESC" if most else "ASC"
        return f"SELECT * FROM {t.name} ORDER BY {cols[0]} {order} LIMIT 1"
    sel = ", ".join(cols) if cols else "*"
    return f"SELECT {sel} FROM {t.name}"
