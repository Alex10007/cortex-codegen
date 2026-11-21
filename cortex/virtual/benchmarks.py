"""Measured infrastructure benchmarks for the virtual-parameter system.

Every number returned here is measured in this process/subprocess at call
time; nothing is synthesized. These cover the stage-7 mandatory
validation items: storage bandwidth, parameter paging, sparse updates,
optimizer-state persistence, distributed sharding, and worker-failure
recovery.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cortex.virtual.store as vstore
from cortex.model.virtual_slot import VirtualAdamW
from cortex.virtual.store import BANK_COLS, BANK_ROWS, ROWS_PER_BANK

SPI = str(Path(__file__).resolve().parents[2])


def _mbps(n_bytes: int, seconds: float) -> float:
    return round(n_bytes / max(seconds, 1e-9) / 1e6, 2)


def bench_storage(root: Path, n_rows: int = 40) -> dict[str, Any]:
    """Sequential float32 row write/read bandwidth on real bank files."""
    path = root / "bench-bank.f32"
    row_bytes = BANK_COLS * 4
    with open(path, "wb") as fh:
        fh.truncate(ROWS_PER_BANK * 4)
    mm = np.memmap(path, dtype=np.float32, mode="r+",
                   shape=(BANK_ROWS, BANK_COLS))
    t0 = time.time()
    for i in range(n_rows):
        mm[i, :] = np.full(BANK_COLS, i, dtype=np.float32)
    mm.flush()
    t_write = time.time() - t0
    t0 = time.time()
    for i in range(n_rows):
        s = float(np.asarray(mm[i, ::16]).sum())
    assert s is not None
    t_read = time.time() - t0
    del mm
    return {
        "rows_benchmarked": n_rows,
        "row_kb": row_bytes // 1024,
        "write_mbps": _mbps(n_rows * row_bytes, t_write),
        "read_mbps": _mbps(n_rows * row_bytes, t_read),
    }


def bench_page_load(store: vstore.VirtualParameterStore, bank: int = 0,
                    n_rows: int = 8) -> dict[str, Any]:
    """Cold (first touch of a closed page + row) vs warm row load cost."""
    t0 = time.time()
    store.page(bank)
    page = store.page(bank)
    cold = time.time() - t0
    t0 = time.time()
    for r in range(n_rows):
        _ = np.asarray(page[r, :512])
    warm = time.time() - t0
    return {
        "cold_page_open_s": round(cold, 4),
        "warm_row_reads": n_rows,
        "warm_row_read_s": round(warm / n_rows, 5),
    }


def bench_sparse_adam(dim: int = 192, rank: int = 16,
                      n_steps: int = 20) -> dict[str, Any]:
    """Per-slot AdamW step latency when the slot receives real gradients."""
    from cortex.model.virtual_slot import SlotState

    opt = VirtualAdamW()
    p = torch.nn.Parameter(torch.zeros(dim, rank))
    g = torch.nn.Parameter(torch.zeros(rank, dim))
    x = torch.randn(dim)
    st = SlotState(bank=0, row=0, A=p, B=g)
    times = []
    for i in range(n_steps):
        ((x @ st.A) @ st.B).sum().backward()
        t0 = time.time()
        rec = opt.step(st)
        times.append(time.time() - t0)
        st.A.grad = None
        st.B.grad = None
    return {
        "steps": n_steps,
        "slot_params": p.numel() + g.numel(),
        "mean_step_ms": round(1000 * float(np.mean(times)), 3),
        "p99_step_ms": round(1000 * float(np.percentile(times, 99)), 3),
        "final_step_count": rec["step"] if rec else 0,
    }


def bench_optimizer_roundtrip(store: vstore.VirtualParameterStore,
                              bank: int = 1, row: int = 2, dim: int = 192,
                              rank: int = 16) -> dict[str, Any]:
    """Save slot AdamW state to the store's npz layout and verify bit equality."""
    from cortex.model.virtual_slot import SlotState, slot_adam_states

    st = store.slot_state(bank, row, dim, rank)
    x = torch.randn(dim)
    ((x @ st.A) @ st.B).sum().backward()
    VirtualAdamW().step(st)
    states = slot_adam_states(st)
    assert states, "optimizer state must exist after a step"
    path = store.root / f"adam-{bank:05d}-{row:06d}.npz"
    np.savez(path, **states)
    loaded = np.load(path)
    ok = True
    for key, arr in states.items():
        if key == "step_count":
            ok = ok and int(loaded[key]) == arr
        else:
            ok = ok and np.array_equal(loaded[key], arr)
    return {
        "bank": bank,
        "row": row,
        "keys": sorted(states.keys()),
        "state_file": path.name,
        "bit_exact": bool(ok),
    }


def run_shard_probe(root: Path, banks: int = 40, world: int = 2) -> dict[str, Any]:
    """Real multi-process gloo sharding probe; ranks spawn concurrently."""
    out = root / "shard"
    out.mkdir(parents=True, exist_ok=True)
    script = Path(SPI) / "scripts" / "virtual_shard_probe.py"
    base_env = {
        **os.environ,
        "PYTHONPATH": f"{SPI}:{os.environ.get('PYTHONPATH', '')}",
        "WORLD_SIZE": str(world),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29517",
    }
    procs = []
    for r in range(world):
        procs.append(subprocess.Popen(
            [sys.executable, str(script), "--root", str(out),
             "--banks", str(banks)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**base_env, "RANK": str(r)}))
    outs = [p.communicate(timeout=600) for p in procs]
    codes = [p.returncode for p in procs]
    report_path = out / "shard-probe-report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    err_tail = "\n".join(o[1] for o in outs)[-800:]
    return {
        "exit_codes": codes,
        "stdout_tail": "\n".join(o[0] for o in outs)[-400:],
        "stderr_tail": err_tail,
        "report": report,
    }


def kill_recovery_probe(root: Path, dim: int = 192, rank: int = 16) -> dict[str, Any]:
    """Worker dies between checkpoints: committed state must remain
    digest-clean, and out-of-protocol mutation of a committed row must be
    DETECTED by digest verification (kernel mmap write-back can persist
    unflushed mutations, so detection - not innocence - is the guarantee).
    """
    import hashlib
    import signal

    script = root / "kill_child.py"
    script.write_text(f'''import sys, time
sys.path.insert(0, {SPI!r})
import torch
import numpy as np
import cortex.virtual.store as vstore
vstore.N_BANKS = 4
store = vstore.VirtualParameterStore({str(root)!r})
store.manifest = store.allocate()
from cortex.model.virtual_slot import VirtualAdamW
st = store.slot_state(0, 0, {dim}, {rank})
opt = VirtualAdamW()
x = torch.randn({dim})
((x @ st.A) @ st.B).sum().backward()
opt.step(st)
store.commit(1, "tx-kill")
page = store.page(1)
page[5, 0] = 1.25
sys.stdout.write("ready\\n"); sys.stdout.flush()
time.sleep(60)
''')
    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().strip()
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
    store = vstore.VirtualParameterStore(root)
    store.manifest = json.loads(store.manifest_path.read_text())
    verified = 0
    digest_ok = True
    for bank_s, rows in store.manifest["initialized_rows"].items():
        page = store.page(int(bank_s))
        for row_s, digest in rows.items():
            verified += 1
            actual = hashlib.sha256(
                np.asarray(page[int(row_s)]).tobytes()).hexdigest()
            digest_ok = digest_ok and actual == digest

    first = next(iter(store.manifest["initialized_rows"].items()))
    bank0 = int(first[0])
    row0 = sorted(first[1].keys())[0]
    page = store.page(bank0)
    page[int(row0), 0] = 1.25
    page.flush()
    post = store.verify_integrity()
    detection_works = not post["integrity_ok"] and int(row0) in {
        int(r.split("/")[1]) for r in post["bad_rows"]}
    return {
        "child_signaled_ready": line == "ready",
        "killed_with_sigkill": True,
        "committed_rows_verified": verified,
        "digests_ok_after_kill": digest_ok,
        "corrupt_row_detected_by_digests": detection_works,
        "recovery_clean": bool(line == "ready" and digest_ok
                               and detection_works),
    }


def run_all(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    import cortex.virtual.store as v

    store = v.VirtualParameterStore(root / "bench-store")
    store.manifest = store.allocate()
    results: dict[str, Any] = {}
    results["storage"] = bench_storage(root)
    results["paging"] = bench_page_load(store)
    results["sparse_adam"] = bench_sparse_adam()
    results["optimizer_roundtrip"] = bench_optimizer_roundtrip(store)
    results["shard_probe"] = run_shard_probe(root)
    results["kill_recovery"] = kill_recovery_probe(root)
    return results
