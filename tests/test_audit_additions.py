"""Tests for: BM25 value retrieval, VES, BIRD difficulty, n>1 fallback, scoring external preds."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from text2sql.baselines.llm_prompt import CallStats, generate
from text2sql.data.schema import Schema, Table
from text2sql.evaluation.execution import evaluate
from text2sql.pruning.schema_pruning import retrieve_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_bm25_value_retrieval(toy_db):
    schema = Schema("toy", [Table("student", ["id", "name", "dept_name"],
                                  ["number", "text", "text"], ["id"])], [])
    hints = retrieve_values("Which students are in math?", toy_db, schema)
    assert hints == ['student.dept_name = "math"']
    assert retrieve_values("nothing matches here", toy_db, schema) == []


def test_ves_and_difficulty(toy_db):
    ex = [SimpleNamespace(qid="1", db_path=toy_db, gold_sql="select count(*) from student"),
          SimpleNamespace(qid="2", db_path=toy_db, gold_sql="select name from student")]
    m = evaluate({"1": "select count(*) from student", "2": "select 1"}, ex,
                 difficulty={"1": "simple", "2": "challenging"}, ves=True)
    assert m["execution_accuracy"] == 0.5
    assert m["ex_by_difficulty"] == {"simple": 1.0, "challenging": 0.0}
    assert 0 < m["ves"] < 5  # one correct query, sqrt of a time ratio near 1, averaged over 2


class RejectsN:
    """Backend that errors when asked for more than one sample per call."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        if kw.get("n", 1) > 1:
            raise ValueError("n > 1 not supported")
        msg = SimpleNamespace(content="SELECT 1")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, logprobs=None)])


def test_generate_falls_back_when_n_rejected():
    stats = CallStats()
    out = generate(RejectsN(), "m", "p", n=3, temperature=0.7, stats=stats)
    assert len(out) == 3 and stats.calls == 3


def test_score_preds_reads_text_and_json(tmp_path):
    from score_preds import read_preds

    exs = [SimpleNamespace(qid="spider-dev-0"), SimpleNamespace(qid="spider-dev-1")]
    txt = tmp_path / "pred.txt"
    txt.write_text("SELECT 1\nSELECT 2\tdb\n")
    assert read_preds(txt, exs) == {"spider-dev-0": "SELECT 1", "spider-dev-1": "SELECT 2"}
    js = tmp_path / "pred.json"
    js.write_text(json.dumps({"spider-dev-0": "SELECT 3"}))
    assert read_preds(js, exs) == {"spider-dev-0": "SELECT 3"}


def test_bird_loader_keeps_difficulty(tmp_path):
    from text2sql.data.loaders import load_bird

    d = tmp_path / "dev"
    d.mkdir()
    (d / "dev.json").write_text(json.dumps([{"db_id": "x", "question": "q", "SQL": "select 1",
                                             "evidence": "e", "difficulty": "moderate"}]))
    ex = load_bird(tmp_path, "dev")[0]
    assert ex.difficulty == "moderate" and ex.evidence == "e"
