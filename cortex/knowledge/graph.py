"""SQLite-backed knowledge graph: versioned nodes and typed edges."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from cortex.interfaces import Provenance


class KnowledgeGraph:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                version TEXT NOT NULL,
                provenance TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                type TEXT NOT NULL,
                version TEXT NOT NULL,
                PRIMARY KEY (src, dst, type, version)
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, type)")
        self._conn.commit()

    def upsert_node(self, node_id: str, ntype: str, label: str,
                    provenance: Provenance) -> None:
        self._conn.execute(
            """INSERT INTO nodes (id, type, label, version, provenance)
               VALUES (?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 type=excluded.type, label=excluded.label,
                 version=excluded.version, provenance=excluded.provenance""",
            (node_id, ntype, label, provenance.version, json.dumps(provenance.__dict__)),
        )

    def upsert_edge(self, src: str, dst: str, etype: str, version: str) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO edges (src, dst, type, version)
               VALUES (?,?,?,?)""",
            (src, dst, etype, version),
        )

    def commit(self) -> None:
        self._conn.commit()

    def neighbors(self, node_id: str, etype: str | None = None,
                  direction: str = "out") -> list[dict[str, Any]]:
        if direction == "out":
            rows = self._conn.execute(
                "SELECT dst, type, version FROM edges WHERE src=?", (node_id,)
            ).fetchall()
        elif direction == "in":
            rows = self._conn.execute(
                "SELECT src, type, version FROM edges WHERE dst=?", (node_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT dst, type, version FROM edges WHERE src=? "
                "UNION ALL SELECT src, type, version FROM edges WHERE dst=?",
                (node_id, node_id),
            ).fetchall()
        return [{"node": r[0], "type": r[1], "version": r[2]} for r in rows
                if etype is None or r[1] == etype]

    def nodes_by_type(self, ntype: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id,label,version FROM nodes WHERE type=? LIMIT ?", (ntype, limit)
        ).fetchall()
        return [{"id": r[0], "label": r[1], "version": r[2]} for r in rows]

    def stats(self) -> dict[str, int]:
        nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        types = dict(self._conn.execute(
            "SELECT type, COUNT(*) FROM nodes GROUP BY type").fetchall())
        return {"nodes": nodes, "edges": edges,
                "node_types": {k: int(v) for k, v in types.items()}}

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
