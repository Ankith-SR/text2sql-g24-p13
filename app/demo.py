"""Stage 4 demo (owner: Harshan). `make demo` -> question + database -> SQL + result table.

Minimal v0: runs the full pipeline against any SQLite file. Extend with a Spider db picker,
candidate/vote inspector, and a latency readout as time allows.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sqlite3  # noqa: E402

from text2sql.baselines.llm_prompt import build_prompt, generate, make_client  # noqa: E402
from text2sql.data.schema import Schema, Table  # noqa: E402
from text2sql.evaluation.execution import execute  # noqa: E402
from text2sql.pipeline.run_pipeline import answer  # noqa: E402

st.title("Text2SQL demo (G24 - P13)")
db_path = st.text_input("SQLite database path", "data/spider/database/concert_singer/concert_singer.sqlite")
backend = st.selectbox("Backend", ["ollama", "voyager"])
model = st.text_input("Model", "qwen2.5-coder:7b")
n = st.slider("Candidates N", 1, 10, 5)
r = st.slider("Max repairs R", 0, 3, 2)
q = st.text_input("Question")

if st.button("Run") and q and Path(db_path).exists():
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = []
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type=\"table\""):
        info = con.execute(f"PRAGMA table_info(\"{name}\")").fetchall()
        tables.append(Table(name, [i[1] for i in info], [i[2].lower() or "text" for i in info]))
    con.close()
    schema = Schema("demo", tables, [])
    client = make_client(backend)
    prompt = build_prompt(schema.to_prompt(), q)
    sql, sel, used = answer(prompt, db_path, lambda p, k, t: generate(client, model, p, n=k, temperature=t), n=n, r=r)
    st.code(sql, language="sql")
    st.caption(f"decided by: {sel.decided_by} | repairs used: {used} | valid candidates: {sel.n_valid}")
    res = execute(db_path, sql)
    if res.ok:
        st.dataframe(pd.DataFrame(res.rows))
    else:
        st.error(res.error)
