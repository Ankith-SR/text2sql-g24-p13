import pytest

from text2sql.evaluation.hardness import hardness

# Labels as produced by the official Spider evaluation.py for these dev queries.
CASES = [
    ("SELECT count(*) FROM singer", "easy"),
    ("SELECT name, country, age FROM singer ORDER BY age DESC", "medium"),
    ("SELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'", "medium"),
    ("SELECT T2.name, count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id "
     "GROUP BY T1.stadium_id", "medium"),
    ("SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)", "hard"),
    ("SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30", "hard"),
    ("SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id = T2.singer_id "
     "JOIN concert AS T3 ON T1.concert_id = T3.concert_id WHERE T3.year = 2014", "hard"),
    ("SELECT name FROM t WHERE a LIKE '%x%' OR b = 1 GROUP BY c ORDER BY d LIMIT 1", "extra"),
]


@pytest.mark.parametrize("sql,level", CASES)
def test_hardness(sql, level):
    assert hardness(sql) == level
