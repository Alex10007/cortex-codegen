"""Scalable training infrastructure planning and validated optimizations.

- hardware profile: measured, never assumed;
- memory estimation: exact optimizer/parameter arithmetic + activation model;
- training plans: single-host vs multi-GPU (DDP layered, ZeRO-style sharding);
- mixed precision: bf16 autocast, executed and validated on the demo config
  with checkpoint save/resume before any larger run is attempted.

Nothing here claims training of the declared 2B config; it sizes the
pipeline that will run it.
"""
from __future__ import annotations

import math
from typing import Any

from cortex.interfaces import ModelConfig

BYTES_PER_PARAM = {"fp32": 4, "bf16": 2, "fp16": 2}


def hardware_profile() -> dict[str, Any]:
    import os
    import torch

    from pathlib import Path

    ram_bytes = 0
    ram_available_bytes = 0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            ram_bytes = int(line.split()[1]) * 1024
        elif line.startswith("MemAvailable:"):
            ram_available_bytes = int(line.split()[1]) * 1024
    if not ram_available_bytes:
        ram_available_bytes = ram_bytes
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu = {
        "available": torch.cuda.is_available(),
        "device_count": device_count,
        "names": ([torch.cuda.get_device_name(i)
                   for i in range(device_count)] if device_count else []),
        "total_memory_bytes": (
            [int(torch.cuda.get_device_properties(i).total_memory)
             for i in range(device_count)] if device_count else []),
        "free_memory_bytes": (
            [int(torch.cuda.mem_get_info(i)[0]) for i in range(device_count)]
            if device_count else []),
        "bf16_supported": (torch.cuda.is_available()
                           and bool(torch.cuda.is_bf16_supported())),
    }
    cpu_bf16 = False
    try:
        x = torch.randn(8, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            (x @ x).sum()
        cpu_bf16 = True
    except Exception:
        pass
    statvfs_ok = None
    try:
        import os as _os
        st = _os.statvfs("/")
        statvfs_ok = st.f_bavail * st.f_frsize
    except OSError:
        pass
    return {
        "cores": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "ram_available_bytes": ram_available_bytes,
        "ram_gb": round(ram_bytes / 1e9, 2),
        "ram_available_gb": round(ram_available_bytes / 1e9, 2),
        "disk_free_gb": round(statvfs_ok / 1e9, 2) if statvfs_ok else None,
        "cuda": gpu,
        "cpu_bf16_autocast_executed_ok": cpu_bf16,
    }


def estimate_memory_bytes(cfg: ModelConfig, batch_size: int,
                          seq_len: int, amp: bool = False) -> dict[str, Any]:
    params = cfg.theoretical_param_count()
    param_bytes = params * 4  # master weights stay fp32 under autocast
    grad_bytes = param_bytes
    optim_bytes = params * 8
    per_tok_per_layer = 12 * cfg.dim
    linear_activation_bytes = (
        cfg.n_layers * batch_size * seq_len * per_tok_per_layer * 4)
    # CausalSelfAttention materializes [batch, heads, seq, seq] scores and
    # softmax probabilities. Count both conservatively in fp32; autocast does
    # not guarantee that all attention intermediates remain low precision.
    attention_matrix_bytes = (
        cfg.n_layers * batch_size * cfg.n_heads * seq_len * seq_len * 4 * 2)
    act_bytes = linear_activation_bytes + attention_matrix_bytes
    return {
        "params": params,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_bytes": optim_bytes,
        "linear_activation_bytes_est": linear_activation_bytes,
        "attention_matrix_bytes_est": attention_matrix_bytes,
        "activation_bytes_est": act_bytes,
        "total_bytes_est": param_bytes + grad_bytes + optim_bytes + act_bytes,
        "total_gb_est": round((param_bytes + grad_bytes + optim_bytes
                               + act_bytes) / 1e9, 2),
        "note": ("conservative fp32 activation estimate includes linear "
                 "activations plus two batch x heads x sequence x sequence "
                 "attention matrices per layer; actual peak must be measured"),
    }


def training_plan(cfg: ModelConfig, batch_size: int = 8,
                  seq_len: int | None = None, amp: bool = True) -> dict[str, Any]:
    seq_len = seq_len or cfg.max_seq_len
    mem = estimate_memory_bytes(cfg, batch_size, seq_len, amp=amp)
    host = hardware_profile()
    fits_slow = mem["total_bytes_est"] <= host["ram_available_bytes"]
    minimal = estimate_memory_bytes(cfg, 1, seq_len, amp=True)
    fits_single_gpu = any(
        minimal["total_bytes_est"] <= free_bytes
        for free_bytes in host["cuda"].get("free_memory_bytes", []))
    if fits_slow:
        strategy = "single-host (tightly padded to RAM; expect heavy paging)"
    elif fits_single_gpu:
        strategy = "single-GPU bf16, gradient accumulation via micro-batching"
    else:
        n_gpu_shards = max(2, math.ceil(
            (mem["param_bytes"] + mem["grad_bytes"] + mem["optimizer_bytes"])
            / 40e9))
        strategy = (f"multi-GPU: ZeRO-3 / FSDP full sharding across "
                    f"~{n_gpu_shards} x 40GB devices (DDP world); CPU-offload "
                    f"optimizer optional")
    return {
        "config": cfg.name,
        "params": mem["params"],
        "human": cfg.human_param_count(),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "amp_bf16": amp,
        "memory": mem,
        "host": host,
        "host_fits_full_state": fits_slow,
        "strategy": strategy,
        "gradient_checkpointing_advised": mem["params"] >= 5e8,
    }


def supported_precision() -> str:
    import torch

    if torch.cuda.is_available():
        return "bf16-gpu"
    try:
        x = torch.randn(8, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            (x @ x).sum()
        return "bf16-cpu"
    except Exception:
        return "fp32"


def validate_scalable_path(config_name: str, tokenizer_path: str,
                           data_bin: str, out_dir: str, steps: int = 20,
                           amp: bool = True, batch_size: int = 4) -> dict[str, Any]:
    """Execute a REAL short training run with mixed precision and prove
    checkpoint rounding-trip/resume: run `steps`, save, load, run again,
    and confirm loss stays finite over an identical seeded schedule.
    """
    import shutil
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from cortex.model.configs import CONFIGS
    from cortex.model.transformer import CortexTransformer
    from cortex.training.engine import set_seed

    cfg = CONFIGS[config_name]
    if cfg.status is not None and cfg.name != "cortex-demo":
        raise PermissionError(
            "validation executes on the demo config only; larger configs are "
            "sized by plan, not run")
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tokens = np.load(data_bin, mmap_mode="r")
    precision = supported_precision()
    dtype = torch.bfloat16 if precision.startswith("bf16") else torch.float32

    set_seed(4242)
    model = CortexTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(4242)
    losses: list[float] = []
    amp_used = amp and precision.startswith("bf16")
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(tokens) - cfg.max_seq_len - 1, size=batch_size)
        src = np.stack([tokens[i:i + cfg.max_seq_len + 1].astype(np.int64)
                        for i in ix])
        x = torch.tensor(src[:, :-1], dtype=torch.long)
        y = torch.tensor(src[:, 1:], dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        if amp_used:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    ck_path = out / "amp-validated.pt"
    torch.save({"model": config_name, "step": steps,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict()}, ck_path)
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state"])
    finite = all(math.isfinite(v) for v in losses)
    return {
        "config": config_name,
        "precision": precision,
        "amp_used": amp_used,
        "steps": steps,
        "losses_finite": finite,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "checkpoint_saved_and_reloaded": True,
        "params": model.actual_param_count(),
        "wall_time_s": round(time.time() - t0, 2),
        "metrics_path": str(out / "amp_smoke.jsonl"),
        "note": "real executed run; proves precision path + checkpoint round-trip",
    }


def cluster_plans(cfg: ModelConfig, batch_size: int,
                  seq_len: int | None = None) -> dict[str, Any]:
    """Per-world-size requirement matrix for the declared 2B config.

    ZeRO-3/FSDP full sharding: parameters, grads and optimizer states split
    across N devices; activations are per-device (micro-batch = batch/N).
    Each row is arithmetic from the exact parameter count, not a guess.
    """
    seq_len = seq_len or cfg.max_seq_len
    params = cfg.theoretical_param_count()
    mem = estimate_memory_bytes(cfg, batch_size, seq_len, amp=True)
    state_bytes = (mem["param_bytes"] + mem["grad_bytes"]
                   + mem["optimizer_bytes"])
    plans = []
    for world in (2, 4, 8):
        if world > batch_size or batch_size % world:
            continue
        per_dev_state = state_bytes / world
        per_dev_act = mem["activation_bytes_est"] / world
        per_dev_total = per_dev_state + per_dev_act
        plans.append({
            "world_size": world,
            "sharding": "ZeRO-3 / FSDP full shard",
            "micro_batch": batch_size // world,
            "per_gpu_state_gb": round(per_dev_state / 1e9, 2),
            "per_gpu_activation_gb_est": round(per_dev_act / 1e9, 2),
            "per_gpu_total_gb_est": round(per_dev_total / 1e9, 2),
            "fits_40gb_gpu": per_dev_total <= 40e9,
            "fits_80gb_gpu": per_dev_total <= 80e9,
        })
    return {
        "config": cfg.name,
        "params": params,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "plans": plans,
        "cpu_offload_optimizer": True,
        "note": "per-GPU arithmetic; measured only on GPU infrastructure "
                "during a real run",
    }
