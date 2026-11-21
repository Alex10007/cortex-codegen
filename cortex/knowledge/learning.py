"""Controlled learning: verified engineering outcomes may become knowledge.

Hard rule enforced here: code that has not actually been executed and passed
its tests can NEVER be stored as trusted training knowledge. Dedup is by
content hash; provenance is mandatory.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from cortex.interfaces import KnowledgeItem, Provenance
from cortex.knowledge.vector_store import VectorStore


class UnverifiedOutcomeError(ValueError):
    pass


class LearningEngine:
    def __init__(self, store: VectorStore):
        self._store = store

    @staticmethod
    def _pid(text: str) -> str:
        return "verified-pattern:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def ingest_verified_outcome(self, code: str, execution_ok: bool,
                                tests_passed: bool, provenance: Provenance,
                                tags: dict[str, Any] | None = None) -> KnowledgeItem:
        if not (execution_ok and tests_passed):
            raise UnverifiedOutcomeError(
                "rejected: code was not executed and test-verified; it cannot enter "
                "trusted knowledge (cortex.knowledge.learning.UnverifiedOutcomeError)"
            )
        item = KnowledgeItem(
            item_id=self._pid(code),
            kind="pattern",
            language=provenance.version and "python",
            text=code,
            provenance=provenance,
            tags={**(tags or {}), "verified": True},
        )
        self._store.add(item)
        self._store.mark_verified(item.item_id, "execution+tests")
        return item

    def ingest_failed_attempt(self, code: str, error: str,
                              provenance: Provenance) -> KnowledgeItem:
        """Failed attempts are stored explicitly as negative evidence, provenance
        tracked, and never marked verified."""
        item = KnowledgeItem(
            item_id="failure:" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:24],
            kind="failure",
            language="python",
            text=code + "\n# observed error: " + error[:2000],
            provenance=provenance,
            tags={"verified": False, "failure_mode": error[:200]},
        )
        self._store.add(item, index=False)
        return item

    def is_known_pattern(self, code: str) -> bool:
        return self._store.get(self._pid(code)) is not None
