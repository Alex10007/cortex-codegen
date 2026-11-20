"""Process-shared API rate limiting backed by a local SQLite database.

SQLite's BEGIN IMMEDIATE serializes bucket updates across worker processes on
the same host. For multi-host deployments, use a shared external limiter.
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import time
from pathlib import Path


class SQLiteRateLimiter:
    def __init__(self, database: str | Path, *, busy_timeout_s: float = 5.0):
        self.database = Path(database).expanduser().absolute()
        self.busy_timeout_s = busy_timeout_s
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.database.is_symlink():
            raise ValueError("rate-limit database must not be a symlink")
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    bucket_key TEXT NOT NULL,
                    occurred_at REAL NOT NULL
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS rate_limit_events_bucket_time
                ON rate_limit_events(bucket_key, occurred_at)
            """)
        try:
            os.chmod(self.database, 0o600)
        except OSError:
            # Windows and restricted filesystems may not support POSIX modes.
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database), timeout=self.busy_timeout_s,
            isolation_level=None)
        connection.execute(
            f"PRAGMA busy_timeout = {int(self.busy_timeout_s * 1000)}")
        return connection

    @staticmethod
    def _digest(bucket: str) -> str:
        return hashlib.sha256(bucket.encode("utf-8")).hexdigest()

    def consume(self, bucket: str, limit: int, window_s: int = 60
                ) -> tuple[bool, int]:
        """Consume one request; return (allowed, retry_after_seconds)."""
        if limit < 1 or window_s < 1:
            raise ValueError("rate limit and window must be positive")
        now = time.time()
        cutoff = now - window_s
        key = self._digest(bucket)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Prune globally so stale buckets do not accumulate forever.
                connection.execute(
                    "DELETE FROM rate_limit_events WHERE occurred_at <= ?",
                    (cutoff,))
                row = connection.execute(
                    """SELECT COUNT(*), MIN(occurred_at)
                       FROM rate_limit_events
                       WHERE bucket_key = ? AND occurred_at > ?""",
                    (key, cutoff)).fetchone()
                count, oldest = int(row[0]), row[1]
                if count >= limit:
                    retry_after = max(1, math.ceil(float(oldest) + window_s - now))
                    connection.execute("ROLLBACK")
                    return False, retry_after
                connection.execute(
                    "INSERT INTO rate_limit_events(bucket_key, occurred_at) "
                    "VALUES (?, ?)", (key, now))
                connection.execute("COMMIT")
                return True, 0
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
