"""Audit logging and observability: append-only JSONL events."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    """Append-only audit trail for security-relevant operations."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = Path(path)

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        entry = {"ts": _now(), "event": event, "payload": payload or {}}
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]

    @classmethod
    def default(cls) -> "AuditLog":
        root = os.environ.get("CORTEX_STATE_DIR", ".cortex-state")
        return cls(str(Path(root) / "audit.jsonl"))


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("cortex")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
