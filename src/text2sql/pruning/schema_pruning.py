"""Stage 3, component 1: schema pruning + BIRD value retrieval (owner: Divyesh).

v0 is a lexical (BM25) ranker so everyone has something working. Upgrade path: swap
`rank_items` for a trained or embedding-based ranker and compare in the ablation
(full schema vs pruned schema).
"""
from __future__ import annotations

import re
import sqlite3

from rank_bm25 import BM25Okapi

from text2sql.data.schema import Schema


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))


def rank_items(question: str, schema: Schema) -> list[tuple[float, str, str]]:
    """Score every (table, column) against the question. Returns sorted desc."""
    items = [(t.name, c) for t in schema.tables for c in t.columns]
    docs = [_tok(f"{t} {c}") for t, c in items]
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(_tok(question))
    return sorted(((float(s), t, c) for s, (t, c) in zip(scores, items)), reverse=True)


def prune(question: str, schema: Schema, top_tables: int = 4, top_cols: int = 6) -> dict[str, set[str]]:
    """Keep top tables, always keep primary keys and FK columns so joins stay possible."""
    ranked = rank_items(question, schema)
    tables: list[str] = []
    for _, t, _ in ranked:
        if t not in tables:
            tables.append(t)
        if len(tables) == top_tables:
            break
    keep: dict[str, set[str]] = {t: set() for t in tables}
    for _, t, c in ranked:
        if t in keep and len(keep[t]) < top_cols:
            keep[t].add(c)
    by_name = {t.name: t for t in schema.tables}
    for t in tables:
        keep[t].update(by_name[t].primary_keys)
    for tbl, col, rt, rc in schema.foreign_keys:
        if tbl in keep and rt in keep:
            keep[tbl].add(col)
            keep[rt].add(rc)
    return keep


def retrieve_values(question: str, db_path: str, schema: Schema, per_col: int = 2,
                    max_rows: int = 5000) -> list[str]:
    """BIRD helper: find text cell values that overlap question words (BM25 over distinct values).

    Returns hints like 'table.column = "value"'. Capped by max_rows per column for speed.
    """
    q = set(_tok(question))
    hints: list[str] = []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for t in schema.tables:
            for c, ty in zip(t.columns, t.types):
                if ty != "text":
                    continue
                try:
                    vals = [r[0] for r in con.execute(
                        f'SELECT DISTINCT "{c}" FROM "{t.name}" WHERE "{c}" IS NOT NULL LIMIT {max_rows}')]
                except sqlite3.Error:
                    continue
                matches = [v for v in vals if isinstance(v, str) and q & set(_tok(v))]
                matches.sort(key=lambda v: -len(q & set(_tok(v))))
                hints += [f'{t.name}.{c} = "{v}"' for v in matches[:per_col]]
    finally:
        con.close()
    return hints
