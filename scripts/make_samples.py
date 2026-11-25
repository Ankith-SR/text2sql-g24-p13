"""After downloading Spider: copy a small, committable sample into data/samples/spider/.

  python scripts/make_samples.py            # 20 dev questions from 2 databases + their .sqlite files

Spider is released under CC BY-SA 4.0 (Yu et al., 2018); the sample keeps that attribution.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "spider"
DST = ROOT / "data" / "samples" / "spider"
DBS = ["concert_singer", "pets_1"]
PER_DB = 10


def main() -> None:
    dev = json.loads((SRC / "dev.json").read_text())
    tables = json.loads((SRC / "tables.json").read_text())
    sample = []
    for db in DBS:
        sample += [{"db_id": r["db_id"], "question": r["question"], "query": r["query"]}
                   for r in dev if r["db_id"] == db][:PER_DB]
        (DST / "database" / db).mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / "database" / db / f"{db}.sqlite", DST / "database" / db / f"{db}.sqlite")
    (DST / "dev_sample.json").write_text(json.dumps(sample, indent=2))
    (DST / "tables_sample.json").write_text(json.dumps([t for t in tables if t["db_id"] in DBS], indent=2))
    (DST / "LICENSE.txt").write_text(
        "Sample of Spider 1.0 (T. Yu et al., EMNLP 2018), https://yale-lily.github.io/spider\n"
        "Licensed under CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/\n")
    print(f"Wrote {len(sample)} questions and {len(DBS)} databases to {DST}")


if __name__ == "__main__":
    main()
