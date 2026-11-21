"""Persistent hierarchical engineering memory backed by one SQLite ledger.

The existing events/facts store remains the canonical memory system. Additive
migrations introduce memory tiers and fact history; no parallel memory engine
or second retrieval store is created. Unverified conflicting assertions are
retained as contested history and cannot overwrite an active verified fact.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEMORY_TIERS = frozenset({"episodic", "semantic", "procedural", "decision"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return value


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
        self._migrate_schema()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS fact_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                verified INTEGER NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                disposition TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_history_project_key "
            "ON fact_history(project, key, id DESC)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_project_tier "
            "ON events(project, memory_tier, id DESC)")
        self._conn.commit()

    def _migrate_schema(self) -> None:
        migrations = {
            "events": {
                "memory_tier": "TEXT NOT NULL DEFAULT 'episodic'",
                "confidence": "REAL NOT NULL DEFAULT 1.0",
                "source": "TEXT NOT NULL DEFAULT 'unspecified'",
            },
            "facts": {
                "confidence": "REAL NOT NULL DEFAULT 1.0",
                "source": "TEXT NOT NULL DEFAULT 'unspecified'",
                "revision": "INTEGER NOT NULL DEFAULT 1",
            },
        }
        for table, columns in migrations.items():
            present = {row[1] for row in self._conn.execute(
                f"PRAGMA table_info({table})").fetchall()}
            for name, declaration in columns.items():
                if name not in present:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def record(self, project: str, kind: str, payload: dict[str, Any],
               verified: bool, tier: str = "episodic",
               confidence: float = 1.0, source: str = "unspecified") -> int:
        if tier not in MEMORY_TIERS:
            raise ValueError(f"tier must be one of {sorted(MEMORY_TIERS)}")
        confidence = _validate_confidence(confidence)
        cur = self._conn.execute(
            "INSERT INTO events "
            "(project, kind, verified, payload, created_at, memory_tier, confidence, source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (project, kind, int(verified), json.dumps(payload, sort_keys=True),
             _now(), tier, confidence, source),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def put_fact(self, project: str, key: str, value: str,
                 verified: bool = False, confidence: float = 1.0,
                 source: str = "unspecified") -> None:
        """Append a fact revision while preserving verified facts from untrusted overwrite."""
        confidence = _validate_confidence(confidence)
        now = _now()
        with self._conn:
            current = self._conn.execute(
                "SELECT value, verified, revision FROM facts WHERE project=? AND key=?",
                (project, key)).fetchone()
            relation = "assertion"
            disposition = "active"
            should_update = True
            revision = 1
            if current:
                old_value, old_verified, old_revision = current
                revision = int(old_revision) + 1
                if old_value != value:
                    relation = "contradiction" if old_verified else "revision"
                    if old_verified and not verified:
                        disposition = "contested"
                        should_update = False
                    else:
                        self._conn.execute(
                            "UPDATE fact_history SET disposition='superseded' "
                            "WHERE project=? AND key=? AND disposition='active'",
                            (project, key))
                else:
                    relation = "confirmation"
                    if old_verified and not verified:
                        disposition = "confirmation"
                        should_update = False
                    else:
                        self._conn.execute(
                            "UPDATE fact_history SET disposition='superseded' "
                            "WHERE project=? AND key=? AND disposition='active'",
                            (project, key))

            self._conn.execute(
                """INSERT INTO fact_history
                   (project,key,value,verified,confidence,source,relation,disposition,recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (project, key, value, int(verified), confidence, source,
                 relation, disposition, now),
            )
            if should_update:
                self._conn.execute(
                    """INSERT INTO facts
                       (project,key,value,verified,updated_at,confidence,source,revision)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(project,key) DO UPDATE SET
                         value=excluded.value, verified=excluded.verified,
                         updated_at=excluded.updated_at, confidence=excluded.confidence,
                         source=excluded.source, revision=excluded.revision""",
                    (project, key, value, int(verified), now, confidence, source, revision),
                )

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
               limit: int = 100, tier: str | None = None) -> list[dict[str, Any]]:
        if tier is not None and tier not in MEMORY_TIERS:
            raise ValueError(f"tier must be one of {sorted(MEMORY_TIERS)}")
        clauses = ["project=?"]
        params: list[Any] = [project]
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        if tier:
            clauses.append("memory_tier=?")
            params.append(tier)
        params.append(limit)
        rows = self._conn.execute(
            "SELECT kind, verified, payload, created_at, memory_tier, confidence, source "
            "FROM events WHERE " + " AND ".join(clauses)
            + " ORDER BY id DESC LIMIT ?", params).fetchall()
        return [{
            "kind": r[0], "verified": bool(r[1]), "payload": json.loads(r[2]),
            "created_at": r[3], "tier": r[4], "confidence": float(r[5]),
            "source": r[6],
        } for r in rows]

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

    def fact_history(self, project: str, key: str | None = None,
                     limit: int = 100) -> list[dict[str, Any]]:
        if key is None:
            rows = self._conn.execute(
                """SELECT key,value,verified,confidence,source,relation,
                          disposition,recorded_at
                   FROM fact_history WHERE project=? ORDER BY id DESC LIMIT ?""",
                (project, limit)).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT key,value,verified,confidence,source,relation,
                          disposition,recorded_at
                   FROM fact_history WHERE project=? AND key=?
                   ORDER BY id DESC LIMIT ?""",
                (project, key, limit)).fetchall()
        return [{
            "key": r[0], "value": r[1], "verified": bool(r[2]),
            "confidence": float(r[3]), "source": r[4], "relation": r[5],
            "disposition": r[6], "recorded_at": r[7],
        } for r in rows]

    def memory_summary(self, project: str) -> dict[str, Any]:
        tier_rows = self._conn.execute(
            "SELECT memory_tier, COUNT(*) FROM events WHERE project=? "
            "GROUP BY memory_tier", (project,)).fetchall()
        fact_rows = self._conn.execute(
            "SELECT COUNT(*), SUM(verified) FROM facts WHERE project=?",
            (project,)).fetchone()
        history_rows = self._conn.execute(
            "SELECT disposition, COUNT(*) FROM fact_history WHERE project=? "
            "GROUP BY disposition", (project,)).fetchall()
        return {
            "project": project,
            "events_by_tier": {str(k): int(v) for k, v in tier_rows},
            "current_facts": int(fact_rows[0] or 0),
            "verified_current_facts": int(fact_rows[1] or 0),
            "fact_history_by_disposition": {str(k): int(v) for k, v in history_rows},
        }

    def verified_outcomes(self, project: str) -> list[dict[str, Any]]:
        return [e for e in self.events(project, limit=500)
                if e["verified"] and e["kind"] == "outcome"]

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
