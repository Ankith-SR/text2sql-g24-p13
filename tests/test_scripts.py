import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_eval import parse_overrides  # noqa: E402


def test_parse_overrides_types():
    assert parse_overrides(["shots=3", "prune=false", "temperature=0.5", "model=llama3.1:8b"]) == {
        "shots": 3, "prune": False, "temperature": 0.5, "model": "llama3.1:8b"}


def test_toy_db_questions_execute():
    from text2sql.evaluation.execution import execute

    root = Path(__file__).resolve().parents[1] / "data" / "samples" / "college"
    for q in json.loads((root / "questions.json").read_text()):
        res = execute(str(root / "college.sqlite"), q["query"])
        assert res.ok and res.rows, q["question"]
