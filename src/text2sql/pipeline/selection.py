"""Stage 3, component 3: execution-based candidate selection (owner: Harshan).

Vote on result sets. When groups tie on votes, break ties in this fixed order
(matches slide 7 / proposal feedback):
  1. prefer non-empty result sets over empty ones
  2. prefer the group with the highest average model confidence (token log-prob)
  3. prefer the simpler query: fewer joins, less nesting, shorter SQL
  4. fall back to the greedy (temperature 0) candidate, so output is deterministic
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from text2sql.evaluation.execution import ExecResult, result_key


@dataclass
class Candidate:
    sql: str
    result: ExecResult
    avg_logprob: float | None = None
    is_greedy: bool = False


@dataclass
class Selection:
    chosen: Candidate | None
    decided_by: str  # unique | non_empty | confidence | simplicity | greedy | all_failed | all_empty
    n_valid: int = 0
    groups: list[list[Candidate]] = field(default_factory=list)


def complexity(sql: str) -> tuple[int, int]:
    s = sql.lower()
    joins = len(re.findall(r"\bjoin\b", s))
    nesting = len(re.findall(r"\(\s*select\b", s))
    return (joins + nesting, len(s))


def _group(cands: list[Candidate]) -> list[list[Candidate]]:
    groups: dict = {}
    for c in cands:
        groups.setdefault(result_key(c.result.rows), []).append(c)
    return list(groups.values())


def _mean_lp(g: list[Candidate]) -> float:
    vals = [c.avg_logprob for c in g if c.avg_logprob is not None]
    return sum(vals) / len(vals) if vals else float("-inf")


def _representative(g: list[Candidate]) -> Candidate:
    return min(g, key=lambda c: (-(c.avg_logprob if c.avg_logprob is not None else float("-inf")),
                                 complexity(c.sql)))


def select(cands: list[Candidate]) -> Selection:
    valid = [c for c in cands if c.result.ok]
    if not valid:
        return Selection(None, "all_failed", 0)
    groups = _group(valid)
    if all(not g[0].result.rows for g in groups):
        best = max(groups, key=lambda g: (len(g), _mean_lp(g)))
        return Selection(_representative(best), "all_empty", len(valid), groups)

    top = max(len(g) for g in groups)
    tied = [g for g in groups if len(g) == top]
    if len(tied) == 1:
        return Selection(_representative(tied[0]), "unique", len(valid), groups)

    # Rule 1: among tied groups, drop empty result sets if any non-empty group tied.
    non_empty = [g for g in tied if g[0].result.rows]
    if 0 < len(non_empty) < len(tied):
        tied = non_empty
        if len(tied) == 1:
            return Selection(_representative(tied[0]), "non_empty", len(valid), groups)
    return Selection(_pick_among(tied), _tie_reason(tied), len(valid), groups)


def _tie_reason(tied: list[list[Candidate]]) -> str:
    lps = sorted((_mean_lp(g) for g in tied), reverse=True)
    if lps[0] != lps[1]:
        return "confidence"
    simp = sorted(min(complexity(c.sql) for c in g) for g in tied)
    if simp[0] != simp[1]:
        return "simplicity"
    return "greedy"


def _pick_among(tied: list[list[Candidate]]) -> Candidate:
    # Rule 2: confidence
    best_lp = max(_mean_lp(g) for g in tied)
    tied = [g for g in tied if _mean_lp(g) == best_lp]
    if len(tied) == 1:
        return _representative(tied[0])
    # Rule 3: simplicity
    best_c = min(min(complexity(c.sql) for c in g) for g in tied)
    tied = [g for g in tied if min(complexity(c.sql) for c in g) == best_c]
    if len(tied) == 1:
        return _representative(tied[0])
    # Rule 4: greedy candidate, else first
    for g in tied:
        for c in g:
            if c.is_greedy:
                return c
    return _representative(tied[0])
