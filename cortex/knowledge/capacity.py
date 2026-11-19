"""Capacity measurement: stored / indexed / retrievable / validated.

Retrievability is MEASURED, not asserted: a probe set is drawn from indexed
items themselves; for each probe we build a query from its distinctive tokens
and check whether it is reachable in the top-k window. The reported number is
a real measurement on the live database.
"""
from __future__ import annotations

import random
from typing import Any

from cortex.knowledge.vector_store import VectorStore, _tokenize


def _probe_query(text: str, min_len: int = 5) -> str | None:
    words = [w for w in _tokenize(text) if len(w) >= min_len]
    if len(words) < 4:
        return None
    # distinctive but realistic query: 4-6 content tokens
    return " ".join(words[:6])


def measure_capacity(store: VectorStore, probe_n: int = 200, k: int = 5,
                     seed: int = 1234) -> dict[str, Any]:
    counts = store.counts()
    stored = counts["stored"]
    indexed = counts["indexed"]
    validated = counts["validated"]

    rows = store.iter_texts()
    indexed_ids = {i for i, _ in rows}
    probe_pool = [(i, t) for i, t in rows if len(t) > 80]
    rng = random.Random(seed)
    sample = rng.sample(probe_pool, min(probe_n, len(probe_pool))) if probe_pool else []

    hits = 0
    tries = 0
    for item_id, text in sample:
        q = _probe_query(text)
        if not q:
            continue
        tries += 1
        for hit in store.search(q, k=k):
            if hit["id"] == item_id:
                hits += 1
                break
    recall = hits / tries if tries else 0.0
    retrievable = int(indexed * recall)

    return {
        "stored": stored,
        "indexed": indexed,
        "retrievable": retrievable,
        "retrieval_recall_at_%d" % k: round(recall, 4),
        "probe_attempts": tries,
        "probe_hits": hits,
        "validated": validated,
    }
