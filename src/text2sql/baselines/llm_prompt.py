"""Tier-3 baseline: prompted open LLMs via any OpenAI-compatible endpoint (owner: Ankith).

Backends (all OpenAI-compatible):
  ollama  : local Ollama            (OLLAMA_BASE_URL)
  voyager : ASU Voyager API         (VOYAGER_BASE_URL / VOYAGER_API_KEY)
  vllm    : vLLM server, e.g. serving the QLoRA adapter (VLLM_BASE_URL); see docs/QLORA_SERVING.md
Prompt follows DAIL-SQL's code representation: CREATE TABLE schema + question, with optional
few-shot (question, SQL) pairs picked by question similarity (`select_shots`).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM = (
    "You are a text-to-SQL engine. Return exactly one SQLite query and nothing else. "
    "No explanation, no markdown."
)


def make_client(backend: str) -> OpenAI:
    if backend == "ollama":
        return OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                      api_key="ollama")
    if backend == "voyager":
        return OpenAI(base_url=os.environ["VOYAGER_BASE_URL"],
                      api_key=os.environ["VOYAGER_API_KEY"])
    if backend == "vllm":
        return OpenAI(base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                      api_key=os.environ.get("VLLM_API_KEY", "EMPTY"))
    raise ValueError(f"unknown backend {backend}")


def build_prompt(schema_text: str, question: str, evidence: str = "",
                 shots: list[tuple[str, str]] | None = None) -> str:
    parts = [f"/* Question: {q} */\n{s}\n" for q, s in (shots or [])]
    parts.append(schema_text)
    if evidence:
        parts.append(f"/* Hint: {evidence} */")
    parts.append(f"/* Question: {question} */")
    return "\n".join(parts)


class ShotSelector:
    """DAIL-SQL-style example selection: the k train questions most similar to the test
    question (BM25 over question tokens). Deterministic, so every system sees the same shots."""

    def __init__(self, pairs: list[tuple[str, str]]):
        from rank_bm25 import BM25Okapi

        self.pairs = pairs
        self.bm25 = BM25Okapi([_tok(q) for q, _ in pairs]) if pairs else None

    def select(self, question: str, k: int) -> list[tuple[str, str]]:
        if k <= 0 or not self.bm25:
            return []
        scores = self.bm25.get_scores(_tok(question))
        idx = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [self.pairs[i] for i in idx]


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text).strip()
    m = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if m and m.start() > 0:
        text = text[m.start():]
    text = text.split(";")[0].strip()
    return re.sub(r"\s+", " ", text)


@dataclass
class CallStats:
    """Counts model calls and wall-clock seconds spent in the model (for latency/cost metrics)."""
    calls: int = 0
    seconds: float = 0.0
    log: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.calls, self.seconds, self.log = 0, 0.0, []


def _one_call(client: OpenAI, model: str, prompt: str, n: int, temperature: float,
              max_tokens: int, want_logprobs: bool):
    kwargs = dict(model=model, n=n, temperature=temperature, max_tokens=max_tokens,
                  messages=[{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}])
    if want_logprobs:
        try:
            return client.chat.completions.create(logprobs=True, **kwargs)
        except Exception:  # noqa: BLE001  backend may reject logprobs; retry without
            pass
    return client.chat.completions.create(**kwargs)


def generate(client: OpenAI, model: str, prompt: str, n: int = 1,
             temperature: float = 0.0, max_tokens: int = 300,
             stats: CallStats | None = None) -> list[tuple[str, float | None]]:
    """Return exactly n [(sql, avg_token_logprob or None)]. Logprobs feed tie-breaking rule 2.

    Some backends (e.g. Ollama's OpenAI endpoint) ignore `n` and return one choice. We then
    keep calling until we have n samples, so N-candidate voting really gets N candidates.
    """
    out: list[tuple[str, float | None]] = []
    ask_many = True
    while len(out) < n:
        t0 = time.perf_counter()
        k = n - len(out) if ask_many else 1
        try:
            resp = _one_call(client, model, prompt, k, temperature, max_tokens, True)
        except Exception:
            if k == 1:
                raise
            ask_many = False  # backend rejects n > 1: one sample per call from now on
            continue
        dt = time.perf_counter() - t0
        if stats is not None:
            stats.calls += 1
            stats.seconds += dt
            stats.log.append(dt)
        if not resp.choices:
            break
        for ch in resp.choices[: n - len(out)]:
            lp = None
            lps = getattr(ch, "logprobs", None)
            if lps and getattr(lps, "content", None):
                vals = [t.logprob for t in lps.content]
                lp = sum(vals) / max(len(vals), 1)
            out.append((extract_sql(ch.message.content or ""), lp))
    return out
