"""Spider / BIRD example loaders (owner: Divyesh).

Expected layout after scripts/setup_data.sh (see data/README.md):
  data/spider/{train_spider.json,dev.json,tables.json,database/<db>/<db>.sqlite}
  data/bird/{train/train.json,dev/dev.json,...}  (adjust db_dir if your download differs)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    qid: str
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    evidence: str = ""  # BIRD external-knowledge hint


def load_spider(root: str | Path, split: str = "dev") -> list[Example]:
    root = Path(root)
    fname = {"dev": "dev.json", "train": "train_spider.json"}[split]
    rows = json.loads((root / fname).read_text())
    return [
        Example(
            qid=f"spider-{split}-{i}",
            db_id=r["db_id"],
            question=r["question"],
            gold_sql=r["query"],
            db_path=str(root / "database" / r["db_id"] / f"{r['db_id']}.sqlite"),
        )
        for i, r in enumerate(rows)
    ]


def load_bird(root: str | Path, split: str = "dev") -> list[Example]:
    root = Path(root)
    rows = json.loads((root / split / f"{split}.json").read_text())
    db_dir = root / split / f"{split}_databases"
    return [
        Example(
            qid=f"bird-{split}-{i}",
            db_id=r["db_id"],
            question=r["question"],
            gold_sql=r["SQL"],
            db_path=str(db_dir / r["db_id"] / f"{r['db_id']}.sqlite"),
            evidence=r.get("evidence", ""),
        )
        for i, r in enumerate(rows)
    ]
