"""Long-sequence handling for completion-only SFT (stage 6).

Resolves the measured stage-5 exclusion (records longer than the training
context contributed zero gradient). An over-length record is split into
overlapping windows over the token stream: window k covers stream positions
[start_k, min(start_k+seq_len, L)) with stride seq_len-1, so each window's
input row (ids[start:end-1]) and target row (labels[start+1:end]) continue
the previous window's exactly - concatenated rows reconstruct ids[:-1] and
labels[1:] with every supervised label covered once. Supervised targets
stay exactly the completion tokens (plus EOT); instruction tokens are
masked (-100) in every window, so completion-only semantics are preserved.
A record is untrainable only when the instruction ALONE cannot fit one
window - counted and reported, never silently truncated.

For records that fit one window, the produced row is byte-identical to the
legacy `make_batch` path.
"""
from __future__ import annotations

import json
from typing import Any


def record_ids(rec: dict[str, Any], tok) -> tuple[list[int], list[int]]:
    prompt = tok.encode(rec["instruction"] + "\n")
    completion = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
    return prompt, completion


def chunk_record(rec: dict[str, Any], tok,
                 seq_len: int) -> list[tuple[list[int], list[int]]] | None:
    """Split one record into raw (ids, labels) windows covering the stream
    exactly. Returns None when the instruction alone cannot fit one window
    (counted by the caller, never truncated)."""
    prompt, completion = record_ids(rec, tok)
    if len(prompt) + 1 > seq_len:
        return None
    ids = prompt + completion
    labels = [-100] * len(prompt) + completion[:]
    L = len(ids)
    if L <= seq_len:
        return [(ids, labels)]
    stride = seq_len - 1
    windows: list[tuple[list[int], list[int]]] = []
    start = 0
    while start < L - 1:
        end = min(start + seq_len, L)
        windows.append((ids[start:end], labels[start:end]))
        if end == L:
            break
        start += stride
    return windows


def make_batch_chunked(recs: list[dict[str, Any]], tok, seq_len: int):
    """Batch of shifted rows from possibly over-long records. A record the
    legacy silently dropped here contributes one or more rows with real
    supervised targets."""
    import torch

    pad_id = tok.pad_id
    width = seq_len - 1
    batch_ids: list[list[int]] = []
    batch_labels: list[list[int]] = []
    for rec in recs:
        windows = chunk_record(rec, tok, seq_len)
        if windows is None:
            continue
        for win_ids, win_labels in windows:
            if len(win_ids) < seq_len:
                # final/clamped window: pad the raw window to the full
                # stream width first, then shift - byte-identical to the
                # legacy path for fitting records
                ids_pad = win_ids + [pad_id] * (seq_len - len(win_ids))
                lab_pad = win_labels + [-100] * (seq_len - len(win_labels))
                row_ids, row_labels = ids_pad[:-1], lab_pad[1:]
            else:
                row_ids, row_labels = win_ids[:-1], win_labels[1:]
            pad = width - len(row_ids)
            batch_ids.append(row_ids + [pad_id] * pad)
            batch_labels.append(row_labels + [-100] * pad)
    if not batch_ids:
        return None, None
    return (torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_labels, dtype=torch.long))


def completion_coverage(recs: list[dict[str, Any]], tok,
                        seq_len: int) -> dict[str, Any]:
    """Measured guarantee: every completion token (incl. EOT) appears as an
    unmasked target exactly once for every record, via full stream
    reconstruction from the windows."""
    total_targets = 0
    for rec in recs:
        windows = chunk_record(rec, tok, seq_len)
        if windows is None:
            continue
        _, completion = record_ids(rec, tok)
        cat_targets: list[int] = []
        for win_ids, win_labels in windows:
            cat_targets += win_labels[1:]
        hit = sum(1 for v in cat_targets if v != -100)
        if hit != len(completion):
            return {"complete": False, "record": rec["name"],
                    "expected": len(completion), "seen": hit}
        total_targets += len(completion)
    return {"complete": True, "total_completion_targets": total_targets}


def dataset_length_stats(jsonl_path: str, tok, seq_len: int,
                         splits: tuple[str, ...] = ("train", "val",
                                                    "heldout")) -> dict[str, Any]:
    """Measured sequence-length compatibility of a dataset at a given
    context length: what the legacy silently-dropping path trained on, what
    chunking recovers, and what is truly untrainable (the instruction alone
    longer than one window)."""
    out: dict[str, Any] = {"seq_len": seq_len, "splits": {}}
    for split in splits:
        n = fit = chunked = untrainable = 0
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("split") != split:
                    continue
                n += 1
                prompt, completion = record_ids(rec, tok)
                if len(prompt) + 1 > seq_len:
                    untrainable += 1
                elif len(prompt) + len(completion) + 1 <= seq_len:
                    fit += 1
                else:
                    chunked += 1
        out["splits"][split] = {
            "records": n,
            "fit_unmodified": fit,
            "recovered_by_chunking": chunked,
            "legacy_trainable": fit,
            "trainable_with_chunking": fit + chunked,
            "instruction_too_long_untrainable": untrainable,
            "legacy_drop_rate": round(1 - fit / n, 4) if n else 0.0,
            "remaining_exclusion_rate": (round(1 - (fit + chunked) / n, 4)
                                         if n else 0.0),
        }
    return out
