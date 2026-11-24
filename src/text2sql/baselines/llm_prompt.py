"""Tier-3 baseline: prompted open LLMs via any OpenAI-compatible endpoint (owner: Ankith).

Works with local Ollama (OLLAMA_BASE_URL) and ASU Voyager (VOYAGER_BASE_URL / KEY).
Prompt follows DAIL-SQL's code representation: CREATE TABLE schema + question.
"""
from __future__ import annotations

import os
import re

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
    raise ValueError(f"unknown backend {backend}")


def build_prompt(schema_text: str, question: str, evidence: str = "",
                 shots: list[tuple[str, str]] | None = None) -> str:
    parts = [f"/* Question: {q} */\n{s}\n" for q, s in (shots or [])]
    parts.append(schema_text)
    if evidence:
        parts.append(f"/* Hint: {evidence} */")
    parts.append(f"/* Question: {question} */")
    return "\n".join(parts)


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text).strip()
    idx = text.upper().find("SELECT")
    if idx > 0:
        text = text[idx:]
    text = text.split(";")[0].strip()
    return re.sub(r"\s+", " ", text)


def generate(client: OpenAI, model: str, prompt: str, n: int = 1,
             temperature: float = 0.0, max_tokens: int = 300) -> list[tuple[str, float | None]]:
    """Return [(sql, avg_token_logprob or None)]. Logprobs feed tie-breaking rule 2."""
    resp = client.chat.completions.create(
        model=model, n=n, temperature=temperature, max_tokens=max_tokens, logprobs=True,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    out = []
    for ch in resp.choices:
        lp = None
        lps = getattr(ch, "logprobs", None)
        if lps and lps.content:
            vals = [t.logprob for t in lps.content]
            lp = sum(vals) / max(len(vals), 1)
        out.append((extract_sql(ch.message.content or ""), lp))
    return out
