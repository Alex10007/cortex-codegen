"""Persistent engineering memory: per-project event/fact ledger (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryEngine:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                kind TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(project, key)
            )"""
        )
        columns = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(facts)").fetchall()}
        if "verified" not in columns:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()

    def record(self, project: str, kind: str, payload: dict[str, Any],
               verified: bool) -> int:
        cur = self._conn.execute(
            "INSERT INTO events (project, kind, verified, payload, created_at) "
            "VALUES (?,?,?,?,?)",
            (project, kind, int(verified), json.dumps(payload), _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def put_fact(self, project: str, key: str, value: str,
                 verified: bool = False) -> None:
        """Store a project fact; facts are excluded from model context by default."""
        self._conn.execute(
            """INSERT INTO facts
               (project, key, value, verified, updated_at) VALUES (?,?,?,?,?)
               ON CONFLICT(project, key) DO UPDATE SET
                 value=excluded.value, verified=excluded.verified,
                 updated_at=excluded.updated_at""",
            (project, key, value, int(verified), _now()),
        )
        self._conn.commit()

    def get_fact(self, project: str, key: str,
                 verified_only: bool = True) -> str | None:
        if verified_only:
            row = self._conn.execute(
                "SELECT value FROM facts WHERE project=? AND key=? AND verified=1",
                (project, key))
        else:
            row = self._conn.execute(
                "SELECT value FROM facts WHERE project=? AND key=?",
                (project, key))
        row = row.fetchone()
        return row[0] if row else None

    def events(self, project: str, kind: str | None = None,
               limit: int = 100) -> list[dict[str, Any]]:
        if kind:
            rows = self._conn.execute(
                "SELECT kind, verified, payload, created_at FROM events "
                "WHERE project=? AND kind=? ORDER BY id DESC LIMIT ?",
                (project, kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT kind, verified, payload, created_at FROM events "
                "WHERE project=? ORDER BY id DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append({"kind": r[0], "verified": bool(r[1]),
                        "payload": json.loads(r[2]), "created_at": r[3]})
        return out

    def facts(self, project: str,
              verified_only: bool = True) -> dict[str, str]:
        if verified_only:
            rows = self._conn.execute(
                "SELECT key, value FROM facts WHERE project=? AND verified=1 "
                "ORDER BY key", (project,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value FROM facts WHERE project=? ORDER BY key",
                (project,)).fetchall()
        return {str(key): str(value) for key, value in rows}

    def verified_outcomes(self, project: str) -> list[dict[str, Any]]:
        return [e for e in self.events(project, limit=500)
                if e["verified"] and e["kind"] == "outcome"]

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
