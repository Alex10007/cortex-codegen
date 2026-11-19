"""Two-rank distributed virtual-parameter-store sharding probe (real gloo).

Each rank owns a disjoint bank range of a 500B-layout partition space,
pages slots in, applies real optimizer updates, commits its shard, and all
ranks exchange digests over gloo to produce a verified union manifest with
zero ownership overlap. Run:

    python scripts/virtual_shard_probe.py --root <dir> --banks 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import cortex.virtual.store as vstore  # noqa: E402
from cortex.model.virtual_slot import VirtualAdamW  # noqa: E402
from cortex.virtual.store import BANK_ROWS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--banks", type=int, default=40)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    dist.init_process_group(backend="gloo", init_method="env://")
    wrank = dist.get_rank()
    world = dist.get_world_size()

    vstore.N_BANKS = args.banks
    store = vstore.VirtualParameterStore(
        os.path.join(args.root, f"shard-{wrank}"))
    store.manifest = store.allocate()

    n_virtual = 8
    opt = VirtualAdamW(lr=1e-3)
    n_banks = args.banks
    per = n_banks // world
    owned = list(range(wrank * per, (wrank + 1) * per))

    opt_rec = []
    for i, bank in enumerate(owned):
        row = bank % BANK_ROWS
        st = store.slot_state(bank, row, args.dim, args.rank,
                              seed=4242 + wrank)
        x = torch.randn(args.dim)
        y = st.forward(x.unsqueeze(0), torch.ones(()))
        y.sum().backward()
        rec = opt.step(st)
        if rec:
            opt_rec.append((bank, row, rec["step"]))

    commits = store.commit(len(opt_rec), f"tx-shard-rank{wrank}")
    integrity = store.verify_integrity()

    payload = {
        "rank": wrank,
        "owned_banks": owned,
        "committed": [(c["bank"], c["row"], c["digest"][:16])
                      for c in commits],
        "coverage_rows_per_bank": store.manifest["coverage_rows_per_bank"],
        "integrity_ok": integrity["integrity_ok"],
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)

    union_banks: set[int] = set()
    overlap: set[int] = set()
    for p in gathered:
        for b in p["owned_banks"]:
            if b in union_banks:
                overlap.add(b)
            union_banks.add(b)
    report = {
        "kind": "cortex-virtual-shard-probe",
        "world_size": world,
        "banks_per_rank": per,
        "banks_covered": len(union_banks),
        "banks_expected": args.banks,
        "ownership_overlap": sorted(overlap),
        "digests_exchanged": sum(len(p["committed"]) for p in gathered),
        "all_integrity_ok": all(p["integrity_ok"] for p in gathered),
        "per_rank": gathered,
        "owned_by": "Fibonera",
    }
    report["sharding_ok"] = (
        not overlap and len(union_banks) == args.banks
        and report["all_integrity_ok"])
    if wrank == 0:
        with open(os.path.join(args.root, "shard-probe-report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps({k: report[k] for k in (
            "world_size", "banks_covered", "ownership_overlap",
            "digests_exchanged", "all_integrity_ok", "sharding_ok")}))
    dist.destroy_process_group()
    return 0 if report["sharding_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
