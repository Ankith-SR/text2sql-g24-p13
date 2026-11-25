"""Build data/samples/college/college.sqlite: a tiny database so the demo and a smoke test work
before Spider is downloaded. Deterministic; safe to re-run.

  python scripts/make_toy_db.py
"""
import json
import random
import sqlite3
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "college"

DEPTS = [(1, "Computer Science", "Brickyard", 1200000), (2, "Mathematics", "Wexler", 650000),
         (3, "Physics", "PSF", 800000), (4, "History", "Coor", 300000)]
COURSES = [(101, "Databases", 1, 3), (102, "Machine Learning", 1, 3), (103, "Algorithms", 1, 4),
           (201, "Linear Algebra", 2, 3), (202, "Statistics", 2, 3), (301, "Quantum Mechanics", 3, 4),
           (401, "World History", 4, 3)]
FIRST = ["Ava", "Liam", "Mia", "Noah", "Emma", "Ethan", "Zoe", "Lucas", "Chloe", "Mason",
         "Ivy", "Leo", "Nora", "Owen", "Ruby", "Jack", "Lily", "Henry", "Ella", "Sam"]

QUESTIONS = [
    ("How many students are there?", "SELECT count(*) FROM student"),
    ("Which departments have more than 5 students?",
     "SELECT d.name FROM department d JOIN student s ON d.dept_id = s.dept_id GROUP BY d.dept_id HAVING count(*) > 5"),
    ("What is the average GPA of students in Computer Science?",
     "SELECT avg(s.gpa) FROM student s JOIN department d ON s.dept_id = d.dept_id WHERE d.name = 'Computer Science'"),
    ("List the names of students who are enrolled in Databases.",
     "SELECT s.name FROM student s JOIN enrollment e ON s.student_id = e.student_id JOIN course c ON e.course_id = c.course_id WHERE c.title = 'Databases'"),
    ("Which department has the largest budget?", "SELECT name FROM department ORDER BY budget DESC LIMIT 1"),
    ("Show each course title and how many students are enrolled in it.",
     "SELECT c.title, count(e.student_id) FROM course c LEFT JOIN enrollment e ON c.course_id = e.course_id GROUP BY c.course_id"),
    ("Which students have a GPA higher than the average GPA?",
     "SELECT name FROM student WHERE gpa > (SELECT avg(gpa) FROM student)"),
    ("Which courses have no students enrolled?",
     "SELECT title FROM course WHERE course_id NOT IN (SELECT course_id FROM enrollment)"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "college.sqlite"
    db.unlink(missing_ok=True)
    rng = random.Random(573)
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE department (dept_id INTEGER PRIMARY KEY, name TEXT, building TEXT, budget REAL);
    CREATE TABLE student (student_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER,
                          year INTEGER, gpa REAL, FOREIGN KEY (dept_id) REFERENCES department(dept_id));
    CREATE TABLE course (course_id INTEGER PRIMARY KEY, title TEXT, dept_id INTEGER, credits INTEGER,
                         FOREIGN KEY (dept_id) REFERENCES department(dept_id));
    CREATE TABLE enrollment (student_id INTEGER, course_id INTEGER, grade TEXT,
                             PRIMARY KEY (student_id, course_id),
                             FOREIGN KEY (student_id) REFERENCES student(student_id),
                             FOREIGN KEY (course_id) REFERENCES course(course_id));
    """)
    con.executemany("INSERT INTO department VALUES (?,?,?,?)", DEPTS)
    con.executemany("INSERT INTO course VALUES (?,?,?,?)", COURSES)
    students = []
    for i, name in enumerate(FIRST, start=1):
        dept = rng.choice([1, 1, 1, 2, 2, 3, 3, 4])
        students.append((i, name, dept, rng.randint(1, 4), round(rng.uniform(2.3, 4.0), 2)))
    con.executemany("INSERT INTO student VALUES (?,?,?,?,?)", students)
    enr = set()
    for sid, *_ in students:
        for cid in rng.sample([c[0] for c in COURSES if c[0] != 401], 2):
            enr.add((sid, cid, rng.choice("ABBC")))
    con.executemany("INSERT INTO enrollment VALUES (?,?,?)", sorted(enr))
    con.commit()
    con.close()
    (OUT / "questions.json").write_text(json.dumps(
        [{"db_id": "college", "question": q, "query": s} for q, s in QUESTIONS], indent=2))
    print(f"Wrote {db} and {OUT/'questions.json'}")


if __name__ == "__main__":
    main()
