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


def required_gpus(config_name: str, batch_size: int) -> tuple[int, float]:
    """Smallest world that fits the config per the cluster matrix, and its
    per-GPU total GB estimate."""
    from cortex.model.configs import get_config
    from cortex.training.scaling import cluster_plans

    cfg = get_config(config_name)
    plans = cluster_plans(cfg, batch_size)
    for row in sorted(plans["plans"], key=lambda p: p["world_size"]):
        if row["fits_40gb_gpu"]:
            return row["world_size"], row["per_gpu_total_gb_est"]
    raise RuntimeError(
        f"no cluster size in the validated matrix fits {config_name} "
        f"(batch {batch_size})")


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
    wanted_world, per_gpu_gb = required_gpus(args.config, args.batch_size)
    world = args.world or wanted_world

    gpu_gib = None
    if profile["cuda"]["available"]:
        total_mem = 0.0
        for i in range(profile["cuda"]["device_count"]):
            free, total = __import__("torch").cuda.mem_get_info(i)
            gpu_gib = max(gpu_gib or 0.0, total / 1024**3)
    needs = {"config": args.config,
             "params": cfg.theoretical_param_count(),
             "required_world": wanted_world,
             "required_per_gpu_gb": per_gpu_gb,
             "measured_cuda_available": profile["cuda"]["available"],
             "measured_device_count": profile["cuda"].get("device_count", 0),
             "measured_max_gpu_gib": gpu_gib}

    if not profile["cuda"]["available"] and args.config != "cortex-demo":
        print(json.dumps({"status": "REFUSED-HARDWARE-GATE", **needs},
                         indent=2))
        print(
            "2B training requires the validated GPU matrix "
            f"(world {wanted_world} x >= {per_gpu_gb} GB). This host measured "
            f"cuda={profile['cuda']['available']}, devices="
            f"{profile['cuda'].get('device_count', 0)}. Nothing was trained "
            "and nothing was simulated. Run with --config cortex-demo to "
            "validate the launch mechanics.")
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
