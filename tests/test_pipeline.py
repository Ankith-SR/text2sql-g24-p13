from text2sql.pipeline.run_pipeline import answer


def test_repair_recovers_from_failure(toy_db):
    calls = []

    def gen(prompt, n, temp):
        calls.append(prompt)
        if "Previous attempt" in prompt:
            return [("select name from student", -0.1)]
        return [("select name from nope", -0.3)] * n

    sql, sel, used = answer("p", toy_db, gen, n=3, r=2)
    assert used == 1 and sql == "select name from student"
