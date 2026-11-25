from types import SimpleNamespace

from text2sql.baselines.llm_prompt import CallStats, ShotSelector, extract_sql, generate
from text2sql.pipeline.run_pipeline import answer_detailed


class OneChoiceClient:
    """Mimics Ollama's OpenAI endpoint: ignores n, rejects logprobs."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        if kw.get("logprobs"):
            raise ValueError("logprobs not supported")
        self.calls += 1
        msg = SimpleNamespace(content=f"```sql\nSELECT {self.calls};\n```")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, logprobs=None)])


def test_generate_loops_until_n_and_counts_calls():
    client, stats = OneChoiceClient(), CallStats()
    out = generate(client, "m", "p", n=4, temperature=0.7, stats=stats)
    assert [s for s, _ in out] == ["SELECT 1", "SELECT 2", "SELECT 3", "SELECT 4"]
    assert stats.calls == 4 and all(lp is None for _, lp in out)


def test_extract_sql_strips_prose():
    assert extract_sql("Here you go:\nSELECT a FROM t; -- done") == "SELECT a FROM t"


def test_shot_selector_picks_similar():
    sel = ShotSelector([("how many singers", "S1"), ("list stadium names", "S2"),
                        ("average age of singers", "S3")])
    assert sel.select("How many singers are there?", 1) == [("how many singers", "S1")]
    assert sel.select("anything", 0) == []


def test_answer_detailed_reports_candidates_and_calls(toy_db):
    def gen(prompt, n, temp):
        return [("select count(*) from student", -0.1)] * n

    res = answer_detailed("p", toy_db, gen, n=3, r=2)
    assert res.sql == "select count(*) from student"
    assert len(res.candidates) == 3 and res.candidates[0].is_greedy
    assert res.n_generate_calls == 2 and res.repairs_used == 0
    assert res.selection.decided_by == "unique" and res.seconds >= 0
