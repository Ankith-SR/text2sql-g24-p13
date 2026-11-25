.PHONY: setup test lint demo toy-db samples heuristic baselines pipeline ablations summary

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	ruff check src tests scripts app

demo:
	streamlit run app/demo.py

toy-db:          ## tiny college DB for the demo (no Spider needed)
	python scripts/make_toy_db.py

samples:         ## after downloading Spider: 20-question sample into data/samples/spider
	python scripts/make_samples.py

heuristic:       ## Tier 1 on full Spider dev (~5 min)
	python scripts/run_eval.py --system heuristic --dataset spider

baselines:       ## Tier 3: every model in the config, 0-shot and 3-shot
	python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml
	python scripts/run_eval.py --system llm --config configs/baseline_llm.yaml --set shots=3

pipeline:        ## our pipeline on the first 200 dev questions
	python scripts/run_eval.py --system pipeline --config configs/pipeline.yaml --limit 200

ablations:
	python scripts/run_ablations.py --limit 200

summary:         ## copy metrics into evaluations/ and build the results table + figures
	python scripts/summarize_results.py
