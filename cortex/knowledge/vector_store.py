"""Persistent vector store: SQLite records + hashed n-gram embeddings.

Embeddings are deterministic hashed character/token n-gram vectors (dim 256,
frozen hashing scheme) — a real, local, dependency-free retrieval primitive.
Cosine similarity is computed with numpy over the stored matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cortex.interfaces import KnowledgeItem, Provenance

EMBED_DIM = 256


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    tokens = _tokenize(text)
    grams: list[str] = list(tokens)
    grams += [a + "_" + b for a, b in zip(tokens, tokens[1:])]
    for w in tokens:
        padded = "^^" + w + "$$"
        grams += [padded[i:i + 3] for i in range(max(1, len(padded) - 2))]
    vec = np.zeros(dim, dtype=np.float32)
    for g in grams:
        slot = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big") % dim
        sign = 1.0 if (int(g.encode("utf-8")[0]) % 2 == 0) else -1.0
        vec[slot] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


class VectorStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                language TEXT NOT NULL,
                version TEXT NOT NULL,
                text TEXT NOT NULL,
                provenance TEXT NOT NULL,
                tags TEXT NOT NULL,
                embedding BLOB,
                verified INTEGER NOT NULL DEFAULT 0,
                verified_by TEXT
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kind ON knowledge_items(kind, version)"
        )
        self._conn.commit()
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []

    # -- write side ---------------------------------------------------------
    def add(self, item: KnowledgeItem, index: bool = True,
            defer_commit: bool = False) -> None:
        emb = embed_text(item.text) if index else None
        self._conn.execute(
            """INSERT INTO knowledge_items
               (id, kind, language, version, text, provenance, tags, embedding, verified)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 kind=excluded.kind, language=excluded.language,
                 version=excluded.version, text=excluded.text,
                 provenance=excluded.provenance, tags=excluded.tags,
                 embedding=excluded.embedding, verified=0, verified_by=NULL""",
            (
                item.item_id,
                item.kind,
                item.language,
                item.provenance.version,
                item.text,
                json.dumps(item.provenance.__dict__),
                json.dumps(item.tags),
                emb.tobytes() if emb is not None else None,
                0,  # trust is granted only by mark_verified after validation
            ),
        )
        if not defer_commit:
            self._conn.commit()
        self._invalidate()

    def flush(self) -> None:
        self._conn.commit()

    def mark_verified(self, item_id: str, verified_by: str) -> None:
        self._conn.execute(
            "UPDATE knowledge_items SET verified=1, verified_by=? WHERE id=?",
            (verified_by, item_id),
        )
        self._conn.commit()

    def get(self, item_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id,kind,language,version,text,provenance,tags,verified,verified_by "
            "FROM knowledge_items WHERE id=?",
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("id", "kind", "language", "version", "text", "provenance", "tags",
                "verified", "verified_by")
        rec = dict(zip(keys, row))
        rec["provenance"] = json.loads(rec["provenance"])
        rec["tags"] = json.loads(rec["tags"])
        return rec

    def iter_texts(self) -> list[tuple[str, str]]:
        return self._conn.execute("SELECT id,text FROM knowledge_items").fetchall()

    # -- read side ----------------------------------------------------------
    def search(self, query: str, k: int = 5,
               kind: str | None = None, language: str | None = None) -> list[dict[str, Any]]:
        self._ensure_matrix()
        if self._matrix is None or not self._ids:
            return []
        q = embed_text(query)
        sims = self._matrix @ q
        idx = np.argsort(-sims)[: k * 4]
        hits: list[dict[str, Any]] = []
        for i in idx:
            rec = self.get(self._ids[i])
            if rec and (kind is None or rec["kind"] == kind) and \
               (language is None or rec["language"] == language):
                hits.append({**rec, "score": float(sims[i])})
            if len(hits) >= k:
                break
        return hits

    def _ensure_matrix(self) -> None:
        if self._matrix is not None:
            return
        rows = self._conn.execute(
            "SELECT id, embedding FROM knowledge_items WHERE embedding IS NOT NULL"
        ).fetchall()
        self._ids = [r[0] for r in rows]
        if rows:
            self._matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        else:
            self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def _invalidate(self) -> None:
        self._matrix = None
        self._ids = []

    # -- stats --------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        cur = self._conn
        stored = cur.execute("SELECT COUNT(*) FROM knowledge_items WHERE TRIM(text) != ''").fetchone()[0]
        indexed = cur.execute("SELECT COUNT(*) FROM knowledge_items WHERE embedding IS NOT NULL").fetchone()[0]
        validated = cur.execute("SELECT COUNT(*) FROM knowledge_items WHERE verified=1 AND TRIM(text) != ''").fetchone()[0]
        return {"stored": stored, "indexed": indexed, "validated": validated}

    def close(self) -> None:
        self._conn.close()
