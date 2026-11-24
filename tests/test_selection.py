from text2sql.evaluation.execution import ExecResult
from text2sql.pipeline.selection import Candidate, select


def C(sql, rows, lp=None, greedy=False, ok=True):
    return Candidate(sql, ExecResult(ok, rows if ok else None, None if ok else "err"), lp, greedy)


def test_majority_wins():
    sel = select([C("a", [(1,)]), C("b", [(1,)]), C("c", [(2,)])])
    assert sel.decided_by == "unique" and sel.chosen.result.rows == [(1,)]


def test_all_failed():
    assert select([C("a", None, ok=False)]).decided_by == "all_failed"


def test_all_empty_flagged_for_repair():
    sel = select([C("a", []), C("b", [])])
    assert sel.decided_by == "all_empty"


def test_rule1_non_empty_beats_empty_on_tie():
    sel = select([C("a", []), C("b", [(1,)])])
    assert sel.decided_by == "non_empty" and sel.chosen.sql == "b"


def test_rule2_confidence():
    sel = select([C("a", [(1,)], lp=-0.5), C("b", [(2,)], lp=-0.1)])
    assert sel.decided_by == "confidence" and sel.chosen.sql == "b"


def test_rule3_simplicity():
    sel = select([C("select x from t join u", [(1,)], lp=-0.2), C("select x from t", [(2,)], lp=-0.2)])
    assert sel.decided_by == "simplicity" and sel.chosen.sql == "select x from t"


def test_rule4_greedy_fallback():
    sel = select([C("aaaa", [(1,)], lp=-0.2), C("bbbb", [(2,)], lp=-0.2, greedy=True)])
    assert sel.decided_by == "greedy" and sel.chosen.sql == "bbbb"


def test_order_insensitive_grouping():
    sel = select([C("a", [(1,), (2,)]), C("b", [(2,), (1,)]), C("c", [(3,)])])
    assert sel.decided_by == "unique" and len(sel.groups) == 2
