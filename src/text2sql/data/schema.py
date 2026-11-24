"""Load Spider/BIRD schemas and serialize them for prompts (owner: Divyesh)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Table:
    name: str
    columns: list[str]
    types: list[str]
    primary_keys: list[str] = field(default_factory=list)


@dataclass
class Schema:
    db_id: str
    tables: list[Table]
    foreign_keys: list[tuple[str, str, str, str]]  # (tbl, col, ref_tbl, ref_col)

    def to_prompt(self, keep: dict[str, set[str]] | None = None) -> str:
        """CREATE TABLE-style serialization (DAIL-SQL code representation).

        keep: optional {table: {columns}} from schema pruning. Tables/columns not in
        keep are dropped. FKs are kept only if both ends survive.
        """
        lines = []
        for t in self.tables:
            if keep is not None and t.name not in keep:
                continue
            cols = [
                (c, ty) for c, ty in zip(t.columns, t.types)
                if keep is None or c in keep[t.name]
            ]
            body = ",\n".join(f"  {c} {ty}" for c, ty in cols)
            lines.append(f"CREATE TABLE {t.name} (\n{body}\n);")
        for tbl, col, rt, rc in self.foreign_keys:
            if keep is None or (tbl in keep and col in keep[tbl] and rt in keep and rc in keep[rt]):
                lines.append(f"-- FOREIGN KEY {tbl}.{col} -> {rt}.{rc}")
        return "\n".join(lines)


def load_schemas(tables_json: str | Path) -> dict[str, Schema]:
    """Parse Spider/BIRD tables.json (same format) into {db_id: Schema}."""
    out: dict[str, Schema] = {}
    for db in json.loads(Path(tables_json).read_text()):
        tnames = db["table_names_original"]
        flat = [(ti, cname) for ti, cname in db["column_names_original"]]
        per_table: dict[int, list[tuple[str, str]]] = {i: [] for i in range(len(tnames))}
        for (ti, cname), ctype in zip(flat, db["column_types"]):
            if ti >= 0:
                per_table[ti].append((cname, ctype))
        pk_idx = {i for i in db.get("primary_keys", []) if isinstance(i, int)}
        pk_names = {(flat[i][0], flat[i][1]) for i in pk_idx}
        tables = [
            Table(
                name=tn,
                columns=[c for c, _ in per_table[i]],
                types=[t for _, t in per_table[i]],
                primary_keys=[c for c, _ in per_table[i] if (i, c) in pk_names],
            )
            for i, tn in enumerate(tnames)
        ]
        fks = []
        for a, b in db.get("foreign_keys", []):
            (ta, ca), (tb, cb) = flat[a], flat[b]
            fks.append((tnames[ta], ca, tnames[tb], cb))
        out[db["db_id"]] = Schema(db["db_id"], tables, fks)
    return out
