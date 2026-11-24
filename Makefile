.PHONY: setup test lint demo

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	ruff check src tests

demo:
	streamlit run app/demo.py
