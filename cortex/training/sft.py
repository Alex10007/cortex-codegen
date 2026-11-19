"""Fine-tuning from verified outcomes.

- Dataset builder: dedup by content hash, license gate, contamination filter
  against evaluation tasks, deterministic train/held-out split.
- SFT loop: completion-only loss masking, warm-started from an existing
  checkpoint, held-out loss per epoch, checkpoint rotation, JSONL metrics.

Verified outcome = the code actually executed and its doctests/tests passed
(see cortex.learning.doctest_harvester and cortex.knowledge.learning).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from cortex.evaluation.tasks import TASKS
from cortex.interfaces import ModelStatus
from cortex.model.configs import CONFIGS, MODEL_STATUS
from cortex.model.transformer import CortexTransformer

ALLOWED_LICENSES = {"Python-2.0 / PSF", "project-local", "apache-2.0", "MIT"}


def _norm_tokens(text: str) -> set[str]:
    cur: list[str] = []
    out: set[str] = set()
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.add("".join(cur))
                cur = []
    if cur:
        out.add("".join(cur))
    return out


def contamination_overlap(record: dict[str, Any]) -> dict[str, float]:
    doc = _norm_tokens(record["instruction"] + "\n" + record["code"])
    scores: dict[str, float] = {}
    for task in TASKS:
        ref = _norm_tokens(task.instruction + "\n" + task.test_code + "\n"
                           + task.reference_solution)
        inter = len(doc & ref)
        union = len(doc | ref) or 1
        scores[task.task_id] = inter / union
    return scores


def build_dataset(pairs: Iterable[dict[str, Any]], out_jsonl: str,
                  contamination_threshold: float = 0.5,
                  held_out_fraction: float = 0.2, seed: int = 1337) -> dict[str, Any]:
    seen_hashes: set[str] = set()
    kept: list[dict[str, Any]] = []
    stats = {"input": 0, "deduped": 0, "license_rejected": 0,
             "contamination_dropped": 0}

    for pair in pairs:
        stats["input"] += 1
        content_hash = hashlib.sha256(pair["code"].encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            stats["deduped"] += 1
            continue
        seen_hashes.add(content_hash)
        license_tag = (pair.get("provenance") or {}).get("license", "unknown")
        if license_tag not in ALLOWED_LICENSES:
            stats["license_rejected"] += 1
            continue
        scores = contamination_overlap(pair)
        if scores and max(scores.values()) > contamination_threshold:
            stats["contamination_dropped"] += 1
            continue
        kept.append({**pair, "content_sha256": content_hash,
                     "contamination_scores": scores})

    rng = random.Random(seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    n_test = max(1, int(len(kept) * held_out_fraction)) if kept else 0
    test_idx = set(order[:n_test])

    train_records = [kept[i] for i in order[n_test:]]
    test_records = [kept[i] for i in order[:n_test]]

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for role, recs in (("train", train_records), ("heldout", test_records)):
            for rec in recs:
                fh.write(json.dumps({**rec, "split": role}) + "\n")

    return {
        "path": str(out),
        "train": len(train_records),
        "heldout": len(test_records),
        "stats": stats,
        "seed": seed,
    }


def load_split(jsonl_path: str, split: str) -> list[dict[str, Any]]:
    recs = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == split:
                recs.append(rec)
    return recs


def make_batch(recs: list[dict[str, Any]], tok, seq_len: int):
    import torch

    pad_id = tok.pad_id
    batch_ids: list[list[int]] = []
    batch_labels: list[list[int]] = []
    for rec in recs:
        prompt = tok.encode(rec["instruction"] + "\n")
        completion = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
        if len(prompt) + len(completion) + 1 > seq_len:
            continue
        ids = prompt + completion
        labels = [-100] * len(prompt) + completion[:]
        ids = ids + [pad_id] * (seq_len - len(ids))
        labels = labels + [-100] * (seq_len - len(labels))
        batch_ids.append(ids[:-1])
        batch_labels.append(labels[1:])
    if not batch_ids:
        return None, None
    return (torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_labels, dtype=torch.long))


def evaluate_heldout(model: CortexTransformer, recs: list[dict[str, Any]], tok,
                     seq_len: int, batch_size: int = 8) -> float | None:
    import torch

    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    with torch.no_grad():
        for i in range(0, len(recs), batch_size):
            x, y = make_batch(recs[i:i + batch_size], tok, seq_len)
            if x is None:
                continue
            logits, _ = model(x)
            mask = y != -100
            if int(mask.sum()) == 0:
                continue
            loss = torch.nn.functional.cross_entropy(
                logits[mask], y[mask], reduction="sum")
            losses.append(float(loss))
            weights.append(int(mask.sum()))
    model.train()
    if not losses:
        return None
    return sum(losses) / sum(weights)


def fine_tune(config_name: str, tokenizer_path: str, dataset_jsonl: str,
              init_checkpoint: str | None, out_dir: str,
              epochs: int = 2, batch_size: int = 8, lr: float = 1e-4,
              seed: int = 7, keep_checkpoints: int = 2,
              log_every: int = 10) -> dict[str, Any]:
    import torch

    from cortex.training.engine import set_seed
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    set_seed(seed)
    cfg = CONFIGS[config_name]
    if cfg.status is ModelStatus.DECLARED:
        raise PermissionError(
            f"config {config_name} is declared-not-trainable on this host; "
            "only cortex-demo may be fine-tuned here"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(tokenizer_path)
    model = CortexTransformer(cfg)
    if init_checkpoint:
        ck = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    train_recs = load_split(dataset_jsonl, "train")
    test_recs = load_split(dataset_jsonl, "heldout")

    metrics_path = out / "metrics.jsonl"
    model.train()
    step = 0
    losses: list[float] = []
    heldout_losses: list[float | None] = []
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_recs)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idxs = order[i:i + batch_size]
            x, y = make_batch([train_recs[j] for j in idxs], tok, cfg.max_seq_len)
            if x is None:
                continue
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % log_every == 0:
                with open(metrics_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(
                        {"step": step, "epoch": epoch,
                         "train_loss": losses[-1]}) + "\n")
        hl = evaluate_heldout(model, test_recs, tok, cfg.max_seq_len)
        heldout_losses.append(hl)
        payload = {"model": config_name, "step": step, "epoch": epoch,
                   "kind": "sft-from-verified-outcomes",
                   "model_state": model.state_dict(),
                   "optimizer_state": opt.state_dict()}
        torch.save(payload, out / f"sft-epoch-{epoch:03d}.pt")
        _rotate(out, keep_checkpoints)

    final = out / "sft-final.pt"
    torch.save({"model": config_name, "step": step,
                "kind": "sft-from-verified-outcomes",
                "model_state": model.state_dict()}, final)
    MODEL_STATUS[config_name].setdefault("sft", []).append({
        "checkpoint": str(final),
        "epochs": epochs,
        "examples": len(train_recs),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "heldout_loss_first_epoch": heldout_losses[0],
        "heldout_loss_last_epoch": heldout_losses[-1],
        "dataset": dataset_jsonl,
    })
    return {
        "config": config_name,
        "epochs": epochs,
        "train_examples": len(train_recs),
        "heldout_examples": len(test_recs),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "heldout_loss_first_epoch": heldout_losses[0],
        "heldout_loss_last_epoch": heldout_losses[-1],
        "param_count": model.actual_param_count(),
        "checkpoint": str(final),
        "metrics": str(metrics_path),
    }


def _rotate(out: Path, keep: int) -> None:
    ckpts = sorted(out.glob("sft-epoch-*.pt"))
    for p in ckpts[:-keep]:
        p.unlink()
