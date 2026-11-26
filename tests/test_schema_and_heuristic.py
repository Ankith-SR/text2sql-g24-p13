import json

from text2sql.baselines import heuristic
from text2sql.data.schema import load_schemas
from text2sql.pruning.schema_pruning import prune

TABLES = [{
    "db_id": "uni",
    "table_names_original": ["student", "course"],
    "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "dept_name"], [1, "cid"], [1, "title"]],
    "column_types": ["text", "number", "text", "text", "number", "text"],
    "primary_keys": [1, 4],
    "foreign_keys": [],
}]


def _schema(tmp_path):
    p = tmp_path / "tables.json"
    p.write_text(json.dumps(TABLES))
    return load_schemas(p)["uni"]


def test_load_and_prompt(tmp_path):
    s = _schema(tmp_path)
    assert [t.name for t in s.tables] == ["student", "course"]
    assert "CREATE TABLE student" in s.to_prompt()
    assert s.tables[0].primary_keys == ["id"]


def test_heuristic_count(tmp_path):
    s = _schema(tmp_path)
    assert heuristic.predict("How many students are there?", s) == "SELECT COUNT(*) FROM student"


def test_prune_keeps_relevant_table_and_pk(tmp_path):
    s = _schema(tmp_path)
    keep = prune("list student names", s, top_tables=1)
    assert "student" in keep and "id" in keep["student"] and "course" not in keep


def test_prune_keeps_unmentioned_columns_of_narrow_tables(tmp_path):
    # "cs students" names a value, not the dept_name column; it must survive pruning.
    s = _schema(tmp_path)
    keep = prune("list cs students", s, top_tables=1)
    assert keep["student"] == {"id", "name", "dept_name"}


def test_prune_cuts_wide_tables(tmp_path):
    s = _schema(tmp_path)
    keep = prune("list student names", s, top_tables=1, max_cols=2)
    assert "name" in keep["student"] and "id" in keep["student"] and "dept_name" not in keep["student"]
