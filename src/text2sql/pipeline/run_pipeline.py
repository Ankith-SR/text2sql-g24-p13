"""Stage 3 orchestration: prune -> sample N -> execute -> vote -> repair up to R (owner: Harshan).

`generate_fn(prompt, n, temperature) -> [(sql, avg_logprob)]` is injected so the loop can be
unit-tested without a model. In real runs, wrap baselines.llm_prompt.generate or a local
HF/vLLM model that loads the QLoRA adapter.
"""
from __future__ import annotations

from collections.abc import Callable

from text2sql.evaluation.execution import execute
from text2sql.pipeline.selection import Candidate, Selection, select

GenerateFn = Callable[[str, int, float], list[tuple[str, float | None]]]


def repair_prompt(base_prompt: str, bad_sql: str, problem: str) -> str:
    return (
        f"{base_prompt}\n\n/* Previous attempt: */\n{bad_sql}\n"
        f"/* It {problem}. Write a corrected SQLite query. */"
    )


def answer(prompt: str, db_path: str, generate_fn: GenerateFn, n: int = 5, r: int = 2,
           temperature: float = 0.7) -> tuple[str, Selection, int]:
    """Return (final_sql, selection, repairs_used)."""
    greedy = generate_fn(prompt, 1, 0.0)
    cands = [Candidate(sql, execute(db_path, sql), lp, is_greedy=True) for sql, lp in greedy]
    if n > 1:
        cands += [Candidate(sql, execute(db_path, sql), lp)
                  for sql, lp in generate_fn(prompt, n - 1, temperature)]
    sel = select(cands)
    used = 0
    while used < r and (sel.chosen is None or not sel.chosen.result.rows):
        used += 1
        if sel.chosen is None:
            worst = cands[0]
            problem = f"failed with error: {worst.result.error}"
        else:
            worst = sel.chosen
            problem = "executed but returned no rows"
        fixed = generate_fn(repair_prompt(prompt, worst.sql, problem), 1, 0.0)
        for sql, lp in fixed:
            cands.append(Candidate(sql, execute(db_path, sql), lp, is_greedy=False))
        sel = select(cands)
    final = sel.chosen.sql if sel.chosen else cands[0].sql
    return final, sel, used
