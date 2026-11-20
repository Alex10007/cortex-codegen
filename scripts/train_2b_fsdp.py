#!/usr/bin/env python
"""Actual 2B Cortex Codegen training launcher (FSDP full shard).

Executes REAL distributed training of a declared Cortex config when run on
the GPU infrastructure it requires (8 x 40 GB or 4 x 80 GB GPUs, see
`cortex/training/scaling.py::cluster_plans`). On hardware that cannot host
the run it refuses with a measured failure instead of simulating training.

Usage:
  python scripts/train_2b_fsdp.py --config cortex-2b --steps 100000 \
      --dataset artifacts/sft/dataset_v3.jsonl
  python scripts/train_2b_fsdp.py --config cortex-demo --steps 6  \
      (mechanics validation on this host)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))


def build_token_bin(dataset_jsonl: str, tokenizer_path: str,
                    out_bin: str, split: str = "train") -> tuple[str, int]:
    """Pack the dataset split into the flat int32 token stream used by the
    distributed workers (train = full LM stream; completion-only SFT data is
    consumed through the SFT path, this bin carries pretraining-style
    corpora such as reference code)."""
    import numpy as np

    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    tok = load_tokenizer(tokenizer_path)
    chunks = []
    with open(dataset_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") not in (split, None):
                continue
            text = (rec.get("code") or "").strip()
            if not text:
                continue
            chunks.append(np.asarray(
                tok.encode(tok.EOT + "\n" + text + "\n"), dtype=np.int32))
    stream = np.concatenate(chunks) if chunks else np.zeros(0, np.int32)
    Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_bin, stream)
    return out_bin, int(stream.shape[0])


def required_gpus(config_name: str, batch_size: int,
                  gpu_memory_gb: float = 40.0) -> tuple[int, float]:
    """Smallest validated world that fits the per-device memory budget."""
    from cortex.model.configs import get_config
    from cortex.training.scaling import cluster_plans

    cfg = get_config(config_name)
    plans = cluster_plans(cfg, batch_size)
    for row in sorted(plans["plans"], key=lambda p: p["world_size"]):
        if row["per_gpu_total_gb_est"] <= gpu_memory_gb:
            return row["world_size"], row["per_gpu_total_gb_est"]
    raise RuntimeError(
        f"no cluster size in the validated matrix fits {config_name} "
        f"(batch {batch_size}, per-device budget {gpu_memory_gb:.2f} GB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cortex-2b")
    ap.add_argument("--dataset", default=str(ART / "sft" / "dataset_v3.jsonl"))
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--world", type=int, default=0,
                    help="override rank world size (defaults to the fitted "
                         "matrix requirement)")
    args = ap.parse_args()

    from cortex.model.configs import CONFIGS
    from cortex.training.scaling import cluster_plans, hardware_profile

    profile = hardware_profile()
    cfg = CONFIGS[args.config]
    cuda = profile["cuda"]
    total_devices = int(cuda.get("device_count", 0))
    total_memory = cuda.get("total_memory_bytes", [])
    free_memory = cuda.get("free_memory_bytes", [])
    # Select the plan against the smallest physical device, not the largest.
    device_budget_gb = (
        min(total_memory) / 1e9 if total_devices and total_memory else 40.0)
    try:
        wanted_world, _ = required_gpus(
            args.config, args.batch_size, gpu_memory_gb=device_budget_gb)
    except RuntimeError as exc:
        print(json.dumps({
            "status": "REFUSED-HARDWARE-GATE",
            "reason": str(exc),
            "config": args.config,
            "params": cfg.theoretical_param_count(),
            "device_budget_gb": round(device_budget_gb, 2),
            "measured_cuda_available": cuda["available"],
            "measured_device_count": total_devices,
        }, indent=2))
        print("The current attention implementation does not fit the "
              "validated cluster matrix. No training was started.")
        return 2
    plans = cluster_plans(cfg, args.batch_size)["plans"]
    plan_by_world = {int(row["world_size"]): row for row in plans}
    world = args.world or wanted_world
    refusal = None
    if world not in plan_by_world:
        refusal = f"world size {world} is not in the validated matrix"
    elif world < wanted_world:
        refusal = (f"world override {world} is below the required "
                   f"minimum {wanted_world}")
    selected = plan_by_world.get(world)
    per_gpu_gb = (selected["per_gpu_total_gb_est"] if selected else None)
    device_report = []
    device_free_bytes = []
    if cuda["available"]:
        import torch
        for i in range(min(world, total_devices)):
            free, total = torch.cuda.mem_get_info(i)
            device_free_bytes.append(int(free))
            device_report.append({
                "device": i,
                "free_gb": round(free / 1e9, 2),
                "total_gb": round(total / 1e9, 2),
            })
        if total_devices < world:
            refusal = (f"requested world {world} requires {world} GPUs, "
                       f"but only {total_devices} are visible")
        elif per_gpu_gb is not None and any(
                free < per_gpu_gb * 1e9 for free in device_free_bytes):
            refusal = (
                f"at least one GPU has less free memory than the "
                f"{per_gpu_gb:.2f} GB per-device estimate")
    elif args.config != "cortex-demo":
        refusal = "CUDA is unavailable; full-scale training is blocked"

    needs = {"config": args.config,
             "params": cfg.theoretical_param_count(),
             "required_world": wanted_world,
             "selected_world": world,
             "required_per_gpu_gb": per_gpu_gb,
             "measured_cuda_available": cuda["available"],
             "measured_device_count": total_devices,
             "device_memory": device_report,
             "device_budget_gb": round(device_budget_gb, 2)}

    if refusal:
        print(json.dumps({"status": "REFUSED-HARDWARE-GATE",
                          "reason": refusal, **needs}, indent=2))
        print("Nothing was trained and nothing was simulated. "
              "Use --config cortex-demo only for mechanics validation.")
        return 2

    data_bin, n_tokens = build_token_bin(
        args.dataset, str(ART / "tokenizer" / "tokenizer.json"),
        str(ART / "data" / f"tokens-{args.config}.npy"))
    out_dir = args.out_dir or str(ART / "runs" / f"fsdp-{args.config}")
    print(json.dumps({"status": "LAUNCHING", **needs,
                      "token_bin": data_bin, "tokens": n_tokens,
                      "world": world, "out_dir": out_dir}, indent=2))

    import torch.multiprocessing as tmp_mp

    from cortex.training.fsdp import fsdp_worker

    tmp_mp.spawn(
        _entry,
        args=(args.config, data_bin, out_dir, world, args.steps,
              args.batch_size, args.lr, args.seed, args.checkpoint_every),
        nprocs=world, join=True)
    print(json.dumps({"status": "COMPLETED", "checkpoint":
                      str(Path(out_dir) / "fsdp-final-rank0.pt")}, indent=2))
    return 0


def _entry(rank, config_name, data_bin, out_dir, world, steps, batch_size,
           lr, seed, checkpoint_every):
    from cortex.training.fsdp import fsdp_worker

    fsdp_worker(config_name, data_bin, out_dir, world, rank, steps,
                batch_size, lr, seed, checkpoint_every=checkpoint_every)


if __name__ == "__main__":
    sys.exit(main())
