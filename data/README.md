# DATA

Full datasets are **not** tracked by git (too large; see `.gitignore`).

| Dataset | Size | Get it | Expected layout |
|---|---|---|---|
| Spider 1.0 | 10,181 questions, 200 DBs (7,000 train / 1,034 dev) | https://yale-lily.github.io/spider | `data/spider/{train_spider.json,dev.json,tables.json,database/<db>/<db>.sqlite}` |
| BIRD | 12,751 pairs, 95 DBs, 33.4 GB | https://bird-bench.github.io/ | `data/bird/{train,dev}/...` (see `scripts/setup_data.sh`) |

Run `bash scripts/setup_data.sh` to check the layout. If BIRD folders are named differently,
edit `load_bird` in `src/text2sql/data/loaders.py`.

## Committed samples (`data/samples/`)
- `college/` — a tiny 4-table database (20 students, 7 courses) with 8 questions + gold SQL,
  built by `scripts/make_toy_db.py`. Lets the demo and a smoke test run without Spider.
- `spider/` — after downloading Spider, `python scripts/make_samples.py` copies 20 dev
  questions and their 2 databases here (Spider is CC BY-SA 4.0; attribution in `LICENSE.txt`).

## Preprocessing (what the code does to every example)
1. **Schema serialization** (`data/schema.py`): `tables.json` → `CREATE TABLE` text with column
   types and foreign keys (DAIL-SQL code representation).
2. **Schema pruning** (`pruning/schema_pruning.py`): BM25 ranks every table/column against the
   question; keep the top 4 tables / 6 columns each, plus primary and foreign keys so joins stay possible.
3. **BIRD**: append the per-question evidence hint; optionally retrieve matching cell values (BM25).
4. **Few-shot** (optional): the k most similar train questions with their SQL (BM25 over questions).
5. **Difficulty** (Spider): easy / medium / hard / extra from the gold SQL (`evaluation/hardness.py`).
