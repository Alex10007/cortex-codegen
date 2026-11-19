"""Training engine: token packing, training loop, checkpoint rotation,
fault-recovery resume, seeded reproducibility, JSONL metrics.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cortex.interfaces import ModelConfig, ModelStatus
from cortex.model.configs import CONFIGS, MODEL_STATUS
from cortex.model.transformer import CortexTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pack_corpus(tokenizer_path: str, roots: list[str], out_bin: str,
                max_bytes: int = 16_000_000) -> tuple[str, int, int]:
    """Tokenize the corpus once into a flat int32 token stream (.npy).

    Returns (path, token_count, file_count).
    """
    from cortex.tokenizer.cortex_tokenizer import CortexTokenizer, _iter_py_files

    tok = load_tokenizer(tokenizer_path)
    files = _iter_py_files(roots, max_bytes=max_bytes, max_file_bytes=300_000)
    Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    n_files = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ids = tok.encode(tok.EOT + "\n" + text.strip() + "\n")
        chunks.append(np.asarray(ids, dtype=np.int32))
        n_files += 1
    stream = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    np.save(out_bin, stream)
    n_tokens = int(stream.shape[0])
    meta_path = str(out_bin).rsplit(".npy", 1)[0] + ".meta.json"
    Path(meta_path).write_text(json.dumps({"tokens": n_tokens, "files": n_files}),
                               encoding="utf-8")
    return out_bin, n_tokens, n_files


def load_tokenizer(tokenizer_path: str):
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer as _load

    return _load(tokenizer_path)


def get_batch(tokens: np.memmap, batch: int, seq_len: int, rng: np.random.Generator):
    ix = rng.integers(0, len(tokens) - seq_len - 1, size=batch)
    src = np.stack([tokens[i:i + seq_len + 1].astype(np.int64) for i in ix])
    x = torch.tensor(src[:, :-1], dtype=torch.long)
    y = torch.tensor(src[:, 1:], dtype=torch.long)
    return x, y



def train(config_name: str, tokenizer_path: str, data_bin: str, out_dir: str,
          steps: int = 800, batch_size: int = 16, lr: float = 6e-4,
          warmup_steps: int = 40, weight_decay: float = 0.1,
          seed: int = 1337, keep_checkpoints: int = 2,
          log_every: int = 25, resume: bool = True) -> dict:
    """Run (or resume) a real training run; returns the run summary.

    Checkpoint rotation keeps the last `keep_checkpoints` checkpoints.
    Metrics go to <out_dir>/metrics.jsonl. Final state updates MODEL_STATUS.
    """
    set_seed(seed)
    cfg: ModelConfig = CONFIGS[config_name]
    if cfg.status is ModelStatus.DECLARED:
        raise PermissionError(
            f"config {config_name} is declared-not-trainable on this host; "
            "use cortex-demo or a host that meets the compute profile"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = CortexTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    step = 0

    if resume:
        existing = sorted(out.glob("checkpoint-*.pt"))
        if existing:
            ck = torch.load(existing[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["optimizer_state"])
            step = ck["step"]

    tokens = np.load(data_bin, mmap_mode="r") if str(data_bin).endswith(".npy") \
        else np.memmap(data_bin, dtype=np.int32, mode="r")
    total = len(tokens)
    rng = np.random.default_rng(seed)
    metrics_path = out / "metrics.jsonl"
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    model.train()
    losses: list[float] = []
    start = time.time()
    while step < steps:
        step += 1
        lr_now = lr if step <= warmup_steps else lr * 0.5 * (
            1.0 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(1, steps - warmup_steps)))
        )
        for group in opt.param_groups:
            group["lr"] = lr_now
        x, y = get_batch(tokens, batch_size, cfg.max_seq_len, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "loss": losses[-1], "lr": lr_now,
                   "tokens_seen": step * batch_size * cfg.max_seq_len,
                   "elapsed_s": round(time.time() - start, 2)}
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        if step % 100 == 0 or step == steps:
            payload = {
                "model": config_name,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
            }
            torch.save(payload, out / f"checkpoint-{step:06d}.pt")
            _rotate_checkpoints(out, keep_checkpoints)

    final_payload = {
        "model": config_name,
        "step": step,
        "model_state": model.state_dict(),
    }
    torch.save(final_payload, out / "final.pt")

    # promote status: only allowed for configs whose status is not DECLARED
    MODEL_STATUS[config_name] = {
        "status": "trained",
        "checkpoint": str(out / "final.pt"),
        "eval": None,
        "run": {
            "steps": step,
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "tokens_seen": step * batch_size * cfg.max_seq_len,
            "seed": seed,
        },
    }
    return {
        "config": config_name,
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "param_count": model.actual_param_count(),
        "tokens_available": int(total),
        "checkpoint": str(out / "final.pt"),
        "metrics": str(metrics_path),
        "wall_time_s": round(time.time() - start, 2),
    }


def _rotate_checkpoints(out: Path, keep: int) -> None:
    ckpts = sorted(out.glob("checkpoint-*.pt"))
    for p in ckpts[:-keep]:
        p.unlink()
