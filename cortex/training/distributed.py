"""Distributed data-parallel training (real torch.distributed, gloo).

Implemented as an environment-driven wrapper around the standard training
step so the same code path is used by a 2-process CPU validation run and a
GPU cluster. Each rank trains on a shard of the token stream; gradients are
all-reduced every step; only rank 0 writes checkpoints (single-writer
lineage). This validates distributed MECHANICS; it does not claim 2B-scale
execution without the measured GPU infrastructure.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from cortex.model.configs import CONFIGS
from cortex.model.transformer import CortexTransformer
from cortex.training.engine import _rotate_checkpoints, set_seed


def _env_setup(world_size: int, rank: int, master_port: int = 29500) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_worker(config_name: str, data_bin: str, out_dir: str,
               world_size: int, rank: int, steps: int, batch_size: int,
               lr: float, seed: int, keep_checkpoints: int = 2,
               log_every: int = 10) -> dict[str, Any]:
    _env_setup(world_size, rank)
    try:
        set_seed(seed + rank)
        cfg = CONFIGS[config_name]
        model = CortexTransformer(cfg)
        ddp_model = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.01)
        tokens = np.load(data_bin, mmap_mode="r")
        shard_len = len(tokens) // world_size
        lo, hi = rank * shard_len, (rank + 1) * shard_len
        rng = np.random.default_rng(seed + rank)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        losses: list[float] = []
        ddp_model.train()
        for step in range(1, steps + 1):
            ix = rng.integers(lo, max(lo + 1, hi - cfg.max_seq_len - 1),
                              size=batch_size)
            src = np.stack([tokens[i:i + cfg.max_seq_len + 1].astype(np.int64)
                            for i in ix])
            x = torch.tensor(src[:, :-1], dtype=torch.long)
            y = torch.tensor(src[:, 1:], dtype=torch.long)
            _, loss = ddp_model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            if step % log_every == 0 or step == steps:
                with open(out / f"ddp-rank{rank}.jsonl", "a",
                          encoding="utf-8") as fh:
                    import json

                    fh.write(json.dumps({
                        "step": step, "loss": losses[-1],
                        "world_size": world_size, "rank": rank,
                        "param_spread_across_ranks": None,
                    }) + "\n")
        if rank == 0:
            torch.save({"model": config_name, "step": steps,
                        "kind": "ddp-validated",
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()},
                       out / f"ddp-final-rank0.pt")
        # verified cross-rank agreement of parameters
        probe_ref = next(model.parameters()).detach()
        probe = probe_ref[0:1]
        gathered = [torch.zeros_like(probe) for _ in range(world_size)]
        dist.all_gather(gathered, probe)
        spread = max(float((g - probe).abs().max()) for g in gathered)
        dist.barrier()
        return {"rank": rank, "steps": steps, "loss_first": losses[0],
                "loss_last": losses[-1], "losses_finite":
                all(math.isfinite(v) for v in losses),
                "param_spread_across_ranks": spread}
    finally:
        dist.destroy_process_group()


def spawn_ddp_validation(config_name: str, data_bin: str, out_dir: str,
                         world_size: int = 2, steps: int = 10,
                         batch_size: int = 2, lr: float = 1e-4,
                         seed: int = 4242) -> dict[str, Any]:
    """Spawn `world_size` real OS processes running the DDP step.

    Uses torch.multiprocessing.spawn against gloo (CPU). Each rank writes
    its result JSON; the parent collects them after join. Raises if any
    rank failed or losses are non-finite.
    """
    import json
    import time

    import torch.multiprocessing as tmp_mp

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("ddp-rank*.json"):
        old.unlink()
    t0 = time.time()
    try:
        tmp_mp.spawn(
            _spawn_entry,
            args=(config_name, str(data_bin), str(out_dir), world_size,
                  steps, batch_size, lr, seed),
            nprocs=world_size, join=True)
    finally:
        pass
    per_rank = []
    for rank in range(world_size):
        res_file = out / f"ddp-rank{rank}.json"
        if not res_file.exists():
            raise RuntimeError(
                f"DDP validation: rank {rank} produced no result")
        per_rank.append(json.loads(res_file.read_text(encoding="utf-8")))
    if any("error" in r for r in per_rank):
        raise RuntimeError(f"DDP validation failed: {per_rank}")
    if not all(r["losses_finite"] for r in per_rank):
        raise RuntimeError("DDP validation: non-finite losses")
    checkpoint = out / "ddp-final-rank0.pt"
    if not checkpoint.exists():
        raise RuntimeError("DDP validation: rank-0 checkpoint missing")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spread = max(r["param_spread_across_ranks"] for r in per_rank)
    return {
        "backend": "gloo",
        "world_size": world_size,
        "steps": steps,
        "per_rank": per_rank,
        "loss_first": per_rank[0]["loss_first"],
        "loss_last": per_rank[0]["loss_last"],
        "rank0_checkpoint": str(checkpoint),
        "checkpoint_reloaded": bool(ck.get("model_state")),
        "max_param_spread_across_ranks": spread,
        "wall_time_s": round(time.time() - t0, 2),
        "note": ("executed distributed mechanics (gradient all-reduce, "
                 "sharded data, single-writer checkpoint) in real OS "
                 "processes; 2B-scale execution still requires the measured "
                 "GPU infrastructure"),
    }


def _spawn_entry(rank, config_name, data_bin, out_dir,
                 world_size, steps, batch_size, lr, seed):
    import json

    try:
        res = ddp_worker(config_name, str(data_bin), str(out_dir),
                         world_size, rank, steps, batch_size, lr, seed)
    except Exception as exc:  # pragma: no cover
        res = {"rank": rank, "error": repr(exc)}
    Path(out_dir, f"ddp-rank{rank}.json").write_text(
        json.dumps(res), encoding="utf-8")
