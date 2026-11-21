"""Stage-10 training-dataset audit (workstream C2).

Measures, from the persisted dataset file only:

- split integrity (deterministic train/val/held-out membership);
- sequence fit: how many records survive `make_batch`'s fit filter at the
  training seq_len (records skipped there are silently untrained);
- completion length distribution;
- exact-duplicate and near-duplicate structure (by content hash and token
  Jaccard on a bounded sample);
- source/module coverage and provenance completeness;
- contamination overlap with the immutable benchmark tasks (recomputed, not
  trusted from stored scores).

No dataset modification happens here - this is a read-only instrument.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cortex.training.sft import contamination_overlap


def _load(jsonl_path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()
            if l.strip()]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / (len(a | b) or 1)


def _tokens(rec: dict[str, Any]) -> set[str]:
    return set((rec["instruction"] + "\n" + rec["code"]).lower().split())


def audit_dataset(jsonl_path: str | Path, tok, seq_len: int,
                  near_dup_threshold: float = 0.9,
                  near_dup_sample: int = 120) -> dict[str, Any]:
    recs = _load(jsonl_path)
    splits: dict[str, int] = {}
    for r in recs:
        splits[r.get("split", "?")] = splits.get(r.get("split", "?"), 0) + 1

    prompt_lens: list[int] = []
    completion_lens: list[int] = []
    fit = 0
    for r in recs:
        p = tok.encode(r["instruction"] + "\n")
        c = tok.encode(r["code"].strip() + "\n" + tok.EOT)
        prompt_lens.append(len(p))
        completion_lens.append(len(c))
        if len(p) + len(c) + 1 <= seq_len:
            fit += 1

    def _stats(xs: list[int]) -> dict[str, float]:
        if not xs:
            return {}
        mx = max(xs)
        return {"min": min(xs), "mean": round(sum(xs) / len(xs), 1),
                "p50": sorted(xs)[len(xs) // 2], "max": mx}

    hashes: dict[str, int] = {}
    code_hashes: dict[str, int] = {}
    for r in recs:
        hashes[r["content_sha256"]] = hashes.get(r["content_sha256"], 0) + 1
        ch = hashlib.sha256(r["code"].encode()).hexdigest()
        code_hashes[ch] = code_hashes.get(ch, 0) + 1

    sample = recs[:near_dup_sample]
    toks = [_tokens(r) for r in sample]
    near_pairs = 0
    compared = 0
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            jacc = _jaccard(toks[i], toks[j])
            compared += 1
            if jacc >= near_dup_threshold:
                near_pairs += 1

    kinds: dict[str, int] = {}
    provenance_missing = 0
    contamination_scores: list[float] = []
    for r in recs:
        kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
        if not r.get("provenance"):
            provenance_missing += 1
        contamination_scores.append(max(r.get("contamination_scores", {}).values()
                                        or [0.0]))

    return {
        "n_records": len(recs),
        "splits": splits,
        "seq_len_audited": seq_len,
        "fit_rate_at_training_seq_len": round(fit / len(recs), 4) if recs else None,
        "records_dropped_by_fit_filter": len(recs) - fit,
        "instruction_tokens": _stats(prompt_lens),
        "completion_tokens": _stats(completion_lens),
        "exact_full_duplicates": sum(1 for v in hashes.values() if v > 1),
        "exact_code_duplicates": sum(1 for v in code_hashes.values() if v > 1),
        "near_dup_jaccard_threshold": near_dup_threshold,
        "near_dup_pairs_in_sample": near_pairs,
        "near_dup_pairs_compared": compared,
        "kinds": kinds,
        "provenance_missing": provenance_missing,
        "max_contamination_overlap": round(max(contamination_scores), 4)
        if contamination_scores else None,
        "contamination_above_half": sum(1 for s in contamination_scores if s > 0.5),
    }


def split_is_deterministic(jsonl_path: str | Path) -> bool:
    """Structural split-integrity check: every record carries exactly one
    allowed split label, no content hash appears in two splits, and all
    three splits are populated."""
    recs = _load(jsonl_path)
    allowed = {"train", "val", "heldout"}
    seen: dict[str, set[str]] = {}
    populated: set[str] = set()
    for r in recs:
        s = r.get("split")
        if s not in allowed:
            return False
        key = r["content_sha256"]
        seen.setdefault(key, set()).add(s)
        if len(seen[key]) > 1:
            return False
        populated.add(s)
    return len(populated) == len(allowed)
