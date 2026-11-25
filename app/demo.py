"""Stage 4 demo (owner: Harshan). `make demo` -> pick a database, ask a question, see the SQL,
the result table, every candidate with its vote, and latency.

Works without Spider (uses data/samples/college) and without an LLM (backend "heuristic"),
so it can always be shown. With Spider downloaded, every dev database and its questions
(with gold SQL, so the demo shows whether the answer is correct) are available.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql.baselines import heuristic  # noqa: E402
from text2sql.baselines.llm_prompt import CallStats, build_prompt, generate, make_client  # noqa: E402
from text2sql.data.schema import Schema, Table  # noqa: E402
from text2sql.evaluation.execution import execute, result_key, same_result  # noqa: E402
from text2sql.pipeline.run_pipeline import answer_detailed  # noqa: E402
from text2sql.pruning.schema_pruning import prune  # noqa: E402

st.set_page_config(page_title="Text2SQL demo (G24-P13)", layout="wide")


# ---------- data sources ----------
@st.cache_data
def list_databases() -> dict[str, dict]:
    """{label: {"path": sqlite path, "questions": [{question, query}]}}"""
    dbs: dict[str, dict] = {}
    for qfile in sorted((ROOT / "data" / "samples").glob("*/questions.json")):
        rows = json.loads(qfile.read_text())
        for db_id in sorted({r["db_id"] for r in rows}):
            dbs[f"sample / {db_id}"] = {"path": str(qfile.parent / f"{db_id}.sqlite"),
                                        "questions": [r for r in rows if r["db_id"] == db_id]}
    sample_spider = ROOT / "data" / "samples" / "spider"
    spider = ROOT / "data" / "spider"
    for base, qname, tag in ((spider, "dev.json", "spider dev"),
                             (sample_spider, "dev_sample.json", "spider sample")):
        if not (base / qname).exists():
            continue
        rows = json.loads((base / qname).read_text())
        for db_id in sorted({r["db_id"] for r in rows}):
            path = base / "database" / db_id / f"{db_id}.sqlite"
            if path.exists():
                dbs[f"{tag} / {db_id}"] = {"path": str(path),
                                           "questions": [r for r in rows if r["db_id"] == db_id]}
        if tag == "spider dev":
            break  # full Spider present: skip the sample copy
    return dbs


@st.cache_data
def read_schema(db_path: str) -> Schema:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables, fks = [], []
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for name in names:
        info = con.execute(f'PRAGMA table_info("{name}")').fetchall()
        tables.append(Table(name, [i[1] for i in info], [(i[2] or "text").lower() for i in info],
                            [i[1] for i in info if i[5]]))
        for fk in con.execute(f'PRAGMA foreign_key_list("{name}")').fetchall():
            if fk[4]:
                fks.append((name, fk[3], fk[2], fk[4]))
    con.close()
    return Schema(Path(db_path).stem, tables, fks)


# ---------- sidebar ----------
st.sidebar.header("Settings")
dbs = list_databases()
choices = list(dbs) + ["custom path..."]
db_label = st.sidebar.selectbox("Database", choices)
if db_label == "custom path...":
    db_path = st.sidebar.text_input("SQLite path", str(ROOT / "data/samples/college/college.sqlite"))
    examples = []
else:
    db_path, examples = dbs[db_label]["path"], dbs[db_label]["questions"]

backend = st.sidebar.selectbox("Backend", ["ollama", "voyager", "vllm", "heuristic (no LLM)"],
                               help="heuristic = Tier-1 baseline, works offline")
use_llm = not backend.startswith("heuristic")
model = st.sidebar.text_input("Model", "qwen2.5-coder:7b", disabled=not use_llm)
n = st.sidebar.slider("Candidates N", 1, 10, 5, disabled=not use_llm)
r = st.sidebar.slider("Max repairs R", 0, 3, 2, disabled=not use_llm)
temp = st.sidebar.slider("Sampling temperature", 0.0, 1.2, 0.7, 0.1, disabled=not use_llm)
use_prune = st.sidebar.checkbox("Schema pruning", True, disabled=not use_llm)

# ---------- main ----------
st.title("Text2SQL demo")
st.caption("CSE 573 · Group 24 · Project 13 — schema pruning → Qwen2.5-Coder-7B → "
           "N candidates → execution voting → repair")

if not Path(db_path).exists():
    st.error(f"Database not found: {db_path}")
    st.stop()
schema = read_schema(db_path)

with st.expander(f"Schema of `{schema.db_id}` ({len(schema.tables)} tables)"):
    st.code(schema.to_prompt(), language="sql")

ex_labels = ["(type your own)"] + [e["question"] for e in examples]
picked = st.selectbox("Example question", ex_labels)
gold_sql = None
default_q = ""
if picked != "(type your own)":
    default_q = picked
    gold_sql = next(e["query"] for e in examples if e["question"] == picked)
q = st.text_input("Question", default_q, placeholder="e.g. How many students are there?")
if q != default_q:
    gold_sql = None

if st.button("Generate SQL", type="primary", disabled=not q):
    keep = prune(q, schema) if (use_llm and use_prune) else None
    prompt = build_prompt(schema.to_prompt(keep), q)
    t0 = time.perf_counter()
    if not use_llm:
        sql = heuristic.predict(q, schema)
        res_detail, stats = None, CallStats()
    else:
        try:
            client = make_client(backend)
        except KeyError as e:
            st.error(f"Missing environment variable {e} (copy .env.example to .env).")
            st.stop()
        stats = CallStats()
        try:
            res_detail = answer_detailed(
                prompt, db_path,
                lambda p, k, t: generate(client, model, p, n=k, temperature=t, stats=stats),
                n=n, r=r, temperature=temp)
        except Exception as e:  # noqa: BLE001
            st.error(f"Model call failed ({backend}): {e}\n\nIs the server running? "
                     "For Ollama: `ollama serve` and `ollama pull qwen2.5-coder:7b`.")
            st.stop()
        sql = res_detail.sql
    total = time.perf_counter() - t0
    result = execute(db_path, sql)

    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    c = st.columns(4)
    c[0].metric("Total latency", f"{total:.2f} s")
    c[1].metric("Model time", f"{stats.seconds:.2f} s")
    c[2].metric("Model calls", stats.calls)
    if res_detail:
        c[3].metric("Repairs used", res_detail.repairs_used)
        st.caption(f"Selection decided by **{res_detail.selection.decided_by}** · "
                   f"{res_detail.selection.n_valid} of {len(res_detail.candidates)} candidates executed")

    if gold_sql is not None:
        gold_res = execute(db_path, gold_sql)
        if same_result(result, gold_res, gold_sql):
            st.success("Correct: same result as the gold SQL")
        else:
            st.warning("Different result from the gold SQL")
        with st.expander("Gold SQL"):
            st.code(gold_sql, language="sql")

    tabs = st.tabs(["Result", "Candidates & vote", "Prompt"])
    with tabs[0]:
        if result.ok:
            st.dataframe(pd.DataFrame(result.rows), use_container_width=True)
            st.caption(f"{len(result.rows)} rows")
        else:
            st.error(result.error)
    with tabs[1]:
        if not res_detail:
            st.info("The heuristic baseline produces a single query (no sampling or voting).")
        else:
            groups = {}
            for cand in res_detail.candidates:
                if cand.result.ok:
                    groups.setdefault(result_key(cand.result.rows), []).append(cand)
            gid = {k: chr(65 + i) for i, k in enumerate(groups)}
            table = []
            for i, cand in enumerate(res_detail.candidates, 1):
                key = result_key(cand.result.rows) if cand.result.ok else None
                kind = ("greedy" if cand.is_greedy else
                        "repair" if i > n else "sample")
                table.append({
                    "#": i, "kind": kind,
                    "chosen": "✅" if res_detail.selection.chosen is cand else "",
                    "status": "ok" if cand.result.ok else f"error: {cand.result.error}",
                    "rows": len(cand.result.rows) if cand.result.ok else None,
                    "result group": gid.get(key, "-"),
                    "votes": len(groups[key]) if key in groups else 0,
                    "avg logprob": (round(cand.avg_logprob, 3)
                                    if cand.avg_logprob is not None else None),
                    "SQL": cand.sql,
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            st.caption("Candidates that return the same rows form one result group; the largest "
                       "group wins. Ties: non-empty → higher confidence → simpler SQL → greedy.")
    with tabs[2]:
        if keep is not None:
            st.caption(f"Pruned schema keeps {len(keep)} of {len(schema.tables)} tables.")
        st.code(prompt, language="sql")
