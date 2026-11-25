"""Spider difficulty levels (easy / medium / hard / extra) for "accuracy by difficulty".

Spider's dev.json has no difficulty field; the official evaluation.py derives it from the gold
SQL with fixed counting rules (Evaluator.eval_hardness). This is a re-implementation of those
rules on top of sqlglot, so we do not need the official parser:

  component1 = [WHERE] + [GROUP BY] + [ORDER BY] + [LIMIT] + (#tables in FROM - 1)
               + #OR in WHERE/HAVING + #LIKE in WHERE/HAVING
  component2 = #nested queries in WHERE/HAVING + #INTERSECT/UNION/EXCEPT
  others     = [#aggregates > 1] + [#select columns > 1] + [#where conditions > 1]
               + [#group-by columns > 1]

It matches the official labels on typical queries; cross-check a sample against the official
script (scripts/export_official.py) before reporting final numbers.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

AGG = (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min)


def _scope_nodes(node: exp.Expression | None, root: exp.Select, cls):
    """Nodes of type `cls` under `node` that belong to `root` (not to a nested SELECT)."""
    if node is None:
        return []
    out = []
    for n in node.find_all(cls):
        if n.find_ancestor(exp.Select) is root or n is root:
            out.append(n)
    return out


def _conditions(cond: exp.Expression | None) -> list[exp.Expression]:
    """Flatten an AND/OR tree into leaf predicates."""
    if cond is None:
        return []
    if isinstance(cond, (exp.And, exp.Or)):
        return _conditions(cond.left) + _conditions(cond.right)
    if isinstance(cond, exp.Paren):
        return _conditions(cond.this)
    return [cond]


def _nested_in(cond: exp.Expression | None) -> int:
    if cond is None:
        return 0
    return sum(1 for s in cond.find_all(exp.Select))


def _top_select(tree: exp.Expression) -> tuple[exp.Select | None, int]:
    """Left-most SELECT of a set operation, plus how many set operators there are."""
    n_set = 0
    node = tree
    while isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        n_set += 1
        node = node.this
    return (node if isinstance(node, exp.Select) else None), n_set


def components(sql: str) -> tuple[int, int, int]:
    tree = sqlglot.parse_one(sql, read="sqlite")
    sel, n_set = _top_select(tree)
    if sel is None:
        return 0, n_set, 0
    where = sel.args.get("where")
    having = sel.args.get("having")
    group = sel.args.get("group")
    order = sel.args.get("order")
    w_cond = where.this if where else None
    h_cond = having.this if having else None

    # tables in FROM (+ JOINs), not counting subqueries
    n_tables = 0
    frm = sel.args.get("from") or sel.args.get("from_")  # key renamed in newer sqlglot
    if frm is not None:
        n_tables += 1
    n_tables += len(sel.args.get("joins") or [])

    c1 = int(where is not None) + int(group is not None) + int(order is not None)
    c1 += int(sel.args.get("limit") is not None) + max(n_tables - 1, 0)
    for cond in (w_cond, h_cond):
        if cond is not None:
            c1 += sum(1 for o in cond.find_all(exp.Or) if o.find_ancestor(exp.Select) is sel)
            c1 += sum(1 for c in _conditions(cond) if isinstance(c, exp.Like))

    c2 = _nested_in(w_cond) + _nested_in(h_cond) + n_set

    aggs = 0
    for part in [*sel.expressions, w_cond, group, order, h_cond]:
        aggs += len([a for a in _scope_nodes(part, sel, AGG)])
    others = int(aggs > 1) + int(len(sel.expressions) > 1)
    others += int(len(_conditions(w_cond)) > 1)
    others += int(group is not None and len(group.expressions) > 1)
    return c1, c2, others


def hardness(sql: str) -> str:
    try:
        c1, c2, others = components(sql)
    except Exception:  # noqa: BLE001  unparsable gold SQL: count as extra (rare)
        return "extra"
    if c1 <= 1 and others == 0 and c2 == 0:
        return "easy"
    if (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0):
        return "medium"
    if ((others > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and others <= 2 and c2 == 0)
            or (c1 <= 1 and others == 0 and c2 <= 1)):
        return "hard"
    return "extra"
