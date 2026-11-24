# Text2SQL (CSE 573, Group 24 / Project 13)

Turn plain-English questions into executable SQL, evaluated on **Spider 1.0** and **BIRD**.
We compare three tiers of system under identical conditions, then build our own pipeline:

| Tier | System |
|---|---|
| 1 | Heuristic keyword / schema matching (`baselines/heuristic.py`) |
| 2 | LGESQL, classical neural (reproduced from its public code, see `docs/LGESQL.md`) |
| 3 | Prompted open LLMs: Qwen2.5-Coder-7B, a Llama-family model, plus a larger open model as upper reference |
| Ours | Schema pruning -> QLoRA-tuned Qwen2.5-Coder-7B -> N samples -> execution voting (with tie-breaks) -> repair (up to R) |

Docs: [proposal](docs/proposal.pdf), [presentation](docs/presentation.pptx), [task board](docs/TASKS.md), [git workflow](CONTRIBUTING.md).

## Quick start
```bash
git clone https://github.com/<org-or-user>/text2sql-g24-p13.git && cd text2sql-g24-p13
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add requirements-train.txt on a GPU box
cp .env.example .env                     # fill in VOYAGER_API_KEY if you use ASU Voyager
bash scripts/setup_data.sh               # prints where to get Spider/BIRD and checks layout
make test                                # 14 unit tests, no data or GPU needed
```

Smoke-test a system (needs data):
```bash
python scripts/run_eval.py --system heuristic --dataset spider --limit 100
python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml --limit 50
python scripts/run_eval.py --system pipeline --config configs/pipeline.yaml --limit 50
```
Each run writes `runs/<name>/{preds.json,metrics.json,run_record.json}`. Commit the two small JSON files, not `preds.json`.

Fine-tune (GPU): `python -m text2sql.training.qlora_sft --config configs/qlora.yaml --pilot 500` -- always pilot first and compare to the untuned base on the same prompts.

## Layout
```
src/text2sql/
  data/         loaders + schema serialization (Spider/BIRD)
  baselines/    heuristic.py, llm_prompt.py (Ollama / ASU Voyager)
  pruning/      BM25 schema pruning + BIRD value retrieval
  training/     QLoRA SFT (TRL + PEFT)
  pipeline/     selection.py (vote + tie-break), run_pipeline.py (sample, vote, repair)
  evaluation/   execution accuracy, validity, by-difficulty
  analysis/     error taxonomy for Stage 2
  utils/        run records
scripts/        run_eval.py, setup_data.sh, create_issues.sh
configs/        yaml per experiment
app/demo.py     Streamlit demo (Stage 4)
tests/          pytest
```

## Owners (from the proposal)
| Who | Area |
|---|---|
| Ankith | LLM prompting baselines, QLoRA runs, run records/configs |
| Divyesh | Spider/BIRD data prep, schema pruning, BIRD value retrieval |
| Harshan | Candidate sampling + voting, repair loop, web demo |
| Hui Yu | Heuristic + LGESQL baselines, eval scripts, error analysis |

## Status: what exists vs. what is a stub
- Working + tested: schema serialization, heuristic baseline, execution evaluation, voting with 4-step tie-break, repair loop (with a fake model in tests).
- Written but not yet run on real data/models: loaders (verify folder names after download), LLM baseline, pruning, QLoRA script, demo app.
- Not started: LGESQL reproduction, official EM / test-suite wrappers, VES for BIRD, ablation sweeps, trained schema ranker.
