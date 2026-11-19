"""Scalar-level training-coverage measurement for the virtual system
(stage 8 extension of the strict stage-7 accounting).

Measurements are derived from real state: the store manifest (committed
rows on disk), live slot objects (activation/gradient state), the persisted
AdamW state files (an npz exists iff VirtualAdamW ever stepped that slot,
i.e. real gradients reached it), the coordinator participation log (per-step
update distributions), and sampling of persisted values. Nothing is
inferred from allocation counts alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import cortex.virtual.store as vstore
from cortex.virtual.store import BANK_COLS, BANK_ROWS, N_BANKS


def persisted_optimizer_scan(
        store: vstore.VirtualParameterStore) -> dict[str, Any]:
    """Ground truth from disk: an adam-*.npz exists iff VirtualAdamW ever
    stepped that slot (step only increments after a real gradient pass)."""
    slots_updated = 0
    params_per_slot = None
    steps: list[int] = []
    for f in sorted(store.root.glob("adam-*.npz")):
        with np.load(f) as z:
            sc = int(z["step_count"])
            if sc <= 0:
                continue
            slots_updated += 1
            if params_per_slot is None:
                params_per_slot = int(z["A_m"].size + z["B_m"].size)
            steps.append(sc)
    if steps:
        arr = np.asarray(steps, dtype=np.float64)
        per_slot = {"n": len(steps), "mean": round(float(arr.mean()), 2),
                    "p50": round(float(np.percentile(arr, 50)), 2),
                    "max": int(arr.max()), "total": int(arr.sum())}
    else:
        per_slot = {"n": 0}
    return {"slots_updated": slots_updated,
            "params_per_slot": params_per_slot,
            "optimizer_steps_per_slot": per_slot}


def measure(store: vstore.VirtualParameterStore,
            trainer=None,
            participation: list[dict] | None = None) -> dict[str, Any]:
    manifest = store.manifest
    committed = manifest["initialized_rows"]
    rows_per_bank = {str(b): len(rows) for b, rows in committed.items()}
    rows_total = sum(rows_per_bank.values())
    participation = participation or []

    # live in-process slot state (this session only)
    live_activated_params = 0
    live_grad_params = 0
    scalars_with_grad = 0
    for st in store._slots.values():
        n = st.A.numel() + st.B.numel()
        live_activated_params += n
        g_nonzero = 0
        if st.A.grad is not None:
            g_nonzero += int((st.A.grad.abs() > 0).sum())
        if st.B.grad is not None:
            g_nonzero += int((st.B.grad.abs() > 0).sum())
        scalars_with_grad += g_nonzero
        if g_nonzero:
            live_grad_params += n

    # gradient recipients derivable from the persisted participation log
    part_slots: set = set()
    for e in participation:
        for br in e.get("virtual_slots_updated", []):
            part_slots.add(tuple(br))
    part_params = None
    for e in participation:
        n_upd = len(e.get("virtual_slots_updated", []))
        if e.get("virtual_grad_params") and n_upd:
            part_params = int(e["virtual_grad_params"]) // n_upd
            break

    persisted = persisted_optimizer_scan(store)
    pps = persisted["params_per_slot"] or part_params
    if pps is None:
        for st in store._slots.values():
            pps = st.A.numel() + st.B.numel()
            break

    persisted_updated_scalars = persisted["slots_updated"] * (pps or 0)
    part_updated_scalars = len(part_slots) * (pps or 0)
    received_scalars = max(live_grad_params, part_updated_scalars,
                           persisted_updated_scalars)

    # activated: live slots, participation slots, or any slot with a
    # persisted optimizer state (activation precedes any gradient/update)
    activated_scalars = max(
        live_activated_params,
        (len(part_slots) + len([k for k in store._slots
                                if tuple(k) in part_slots])) * (pps or 0),
        persisted_updated_scalars)

    # persisted value sampling among committed rows
    rows_sampled = 0
    nonzero_rows = 0
    nonzero_scalars = 0
    for bank_s in list(committed)[:64]:
        page = store.page(int(bank_s))
        for row_s in committed[bank_s]:
            vec = np.asarray(page[int(row_s)])
            rows_sampled += 1
            row_nonzero = int(np.count_nonzero(vec))
            nonzero_scalars += row_nonzero
            nonzero_rows += int(row_nonzero > 0)

    dist = update_distribution(participation)
    scalars = {
        "allocated": vstore.exact_virtual_count(),
        "initialized": rows_total * BANK_COLS,
        "activated": activated_scalars,
        "received_gradients": received_scalars,
        "received_gradients_scalars": scalars_with_grad,
        "optimizer_updated": persisted_updated_scalars,
        "demonstrably_trained": persisted_updated_scalars,
    }
    note = ("received_gradients is the union of persisted optimizer state "
            "(npz step_count>0), participation-log recipients and live "
            "session gradients; optimizer_updated/demonstrably_trained come "
            "strictly from persisted optimizer state on disk")
    return {
        "identity": "cortex-virtual-scalar-coverage",
        "scalars": scalars,
        "persisted_optimizer_scan": persisted,
        "note": note,
        "scalar_level": {
            "rows_committed": rows_total,
            "rows_sampled": rows_sampled,
            "sampled_rows_nonzero": nonzero_rows,
            "persisted_nonzero_scalars_sampled": nonzero_scalars,
            "fraction_of_sampled_rows_with_nonzero_values":
                round(nonzero_rows / rows_sampled, 4) if rows_sampled else 0.0,
        },
        "partition_level": {
            "partitions_with_committed_row": len(rows_per_bank),
            "partitions_total": N_BANKS,
            "rows_per_bank_max": max(rows_per_bank.values())
            if rows_per_bank else 0,
            "coverage_fraction_of_partitions": round(
                len(rows_per_bank) / N_BANKS, 4),
        },
        "update_distribution": dist,
        "invariants_hold": (
            scalars["allocated"] >= scalars["initialized"]
            >= scalars["received_gradients"]
            >= scalars["optimizer_updated"]
            >= scalars["demonstrably_trained"]
            and scalars["demonstrably_trained"] < scalars["allocated"]),
    }


def update_distribution(participation: list[dict]) -> dict[str, Any]:
    """Distribution of scalar update magnitudes across participation log."""
    maxes = [e["update_scalar_max_abs_delta"] for e in participation
             if "update_scalar_max_abs_delta" in e]
    means = [e["update_scalar_mean_abs_delta"] for e in participation
             if "update_scalar_mean_abs_delta" in e]
    moved = [e["update_scalars_moved"] for e in participation
             if "update_scalars_moved" in e]
    totals = [e["update_scalars_total"] for e in participation
              if "update_scalars_total" in e]
    if not maxes:
        return {"steps_with_data": 0}
    bucket = lambda xs: {  # noqa: E731
        f"p{p}": round(float(np.percentile(xs, p)), 8) for p in (5, 50, 95)}
    pct = {f"p{p}": round(float(np.percentile(maxes, p)), 8)
           for p in (5, 50, 95)}
    return {
        "steps_with_data": len(maxes),
        "max_abs_delta": {**pct, "max": round(max(maxes), 8)},
        "mean_abs_delta": bucket(means),
        "scalars_moved_per_step": bucket(moved),
        "scalars_total_per_step": bucket(totals),
        "hist_max_abs_delta": histogram(maxes),
    }


def _np():
    return np


def histogram(values: list[float], bins: int = 10) -> dict[str, int]:
    arr = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    return {
        f"[{edges[i]:.2e},{edges[i + 1]:.2e})": int(counts[i])
        for i in range(len(counts)) if counts[i] > 0
    }


def summarize_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())
