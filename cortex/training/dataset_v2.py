"""Dataset v2: train/val/held-out with a fingerprint manifest.

The held-out split is immutable in practice: its fingerprint is recorded in
the manifest at build time and every later load verifies it. Existing v1
gates (dedup, license, contamination) are reused, not duplicated.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from cortex.training.sft import build_dataset


def _canonical_fingerprint(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted((r["name"], r["code"], r["instruction"], r["split"])
               for r in records), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dataset_v2(pairs: Iterable[dict[str, Any]], out_jsonl: str,
                     manifest_path: str | None = None,
                     val_fraction: float = 0.1,
                     held_out_fraction: float = 0.2,
                     seed: int = 1337) -> dict[str, Any]:
    """Two-stage split dedupe first via build_dataset (train/heldout), then
    carve validation out of the training side. Deterministic given seed."""
    stage1 = build_dataset(pairs, out_jsonl, held_out_fraction=held_out_fraction,
                           seed=seed)
    records = [json.loads(l) for l in
               Path(out_jsonl).read_text(encoding="utf-8").splitlines()]
    train = [r for r in records if r["split"] == "train"]
    held = [r for r in records if r["split"] == "heldout"]

    rng = random.Random(seed + 1)
    order = list(range(len(train)))
    rng.shuffle(order)
    n_val = max(1, int(len(train) * val_fraction)) if train else 0
    val_idx = set(order[:n_val])
    val = [train[i] for i in order[:n_val]]
    final_train = [train[i] for i in order[n_val:]]
    for r in val:
        r["split"] = "val"

    final = final_train + val + held
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r) + "\n")

    manifest = {
        "path": str(out_jsonl),
        "counts": {"train": len(final_train), "val": len(val),
                   "heldout": len(held)},
        "stats": stage1["stats"],
        "seeds": {"split": seed, "val_carve": seed + 1},
        "fingerprints": {
            "full": _canonical_fingerprint(final),
            "train": _canonical_fingerprint(final_train),
            "val": _canonical_fingerprint(val),
            "heldout": _canonical_fingerprint(held),
        },
    }
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    return manifest


def load_split_v2(jsonl_path: str, split: str,
                  manifest_path: str | None = None) -> list[dict[str, Any]]:
    recs = [json.loads(l) for l in
            Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if json.loads(l).get("split") == split]
    if manifest_path and split == "heldout":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        current = _canonical_fingerprint(recs)
        recorded = manifest["fingerprints"]["heldout"]
        if current != recorded:
            raise RuntimeError(
                "held-out split has changed since training "
                f"({current} != {recorded}); results are not comparable"
            )
    return recs
