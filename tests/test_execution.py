from types import SimpleNamespace

from text2sql.evaluation.execution import evaluate, execute


def test_execute_ok_and_error(toy_db):
    assert execute(toy_db, "select count(*) from student").rows == [(3,)]
    r = execute(toy_db, "select * from nope")
    assert not r.ok and "no such table" in r.error


def test_evaluate_ex_and_validity(toy_db):
    ex = [SimpleNamespace(qid="1", db_path=toy_db, gold_sql="select name from student where dept_name=\"cs\""),
          SimpleNamespace(qid="2", db_path=toy_db, gold_sql="select count(*) from student")]
    preds = {"1": "SELECT name FROM student WHERE dept_name = \"cs\" ORDER BY id DESC".replace(" ORDER BY id DESC", ""),
             "2": "select * from broken"}
    m = evaluate(preds, ex)
    assert m["n"] == 2 and m["execution_accuracy"] == 0.5 and m["validity_rate"] == 0.5


def test_execute_timeout_and_readonly(toy_db):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT count(*) FROM c")
    r = execute(toy_db, slow, timeout=1)
    assert not r.ok and r.error == "timeout"
    w = execute(toy_db, "DELETE FROM student")
    assert not w.ok and "readonly" in w.error.replace(" ", "")
    assert execute(toy_db, "select count(*) from student").rows == [(3,)]
