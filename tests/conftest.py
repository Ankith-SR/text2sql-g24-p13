import sqlite3

import pytest


@pytest.fixture
def toy_db(tmp_path):
    p = tmp_path / "toy.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
    CREATE TABLE student (id INTEGER PRIMARY KEY, name TEXT, dept_name TEXT);
    INSERT INTO student VALUES (1,\"a\",\"cs\"),(2,\"b\",\"cs\"),(3,\"c\",\"math\");
    """)
    con.commit()
    con.close()
    return str(p)
