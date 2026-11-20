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
from cortex.virtual.store import N_BANKS


def persisted_optimizer_scan(
        store: vstore.VirtualParameterStore) -> dict[str, Any]:
    """Ground truth from disk: an adam-*.npz exists iff VirtualAdamW ever
    stepped that slot (step only increments after a real gradient pass)."""
    slots_updated = 0
    params_per_slot = None
    optimizer_updated_scalars = 0
    demonstrably_trained_scalars = 0
    demonstrably_trained_unknown_slots = 0
    orphan_optimizer_slots = 0
    committed = {
        (int(bank), int(row))
        for bank, rows in store.manifest.get("initialized_rows", {}).items()
        for row in rows
    }
    steps: list[int] = []
    for f in sorted(store.root.glob("adam-*.npz")):
        with np.load(f) as z:
            stem = f.stem.removeprefix("adam-")
            bank_s, row_s = stem.split("-", 1)
            if (int(bank_s), int(row_s)) not in committed:
                orphan_optimizer_slots += 1
                continue
            sc = int(z["step_count"])
            if sc <= 0:
                continue
            slots_updated += 1
            slot_scalars = int(z["A_m"].size + z["B_m"].size)
            optimizer_updated_scalars += slot_scalars
            if ("changed_evidence_known" in z
                    and "A_changed_mask" in z
                    and "B_changed_mask" in z):
                demonstrably_trained_scalars += int(
                    np.count_nonzero(z["A_changed_mask"])
                    + np.count_nonzero(z["B_changed_mask"]))
                if int(z["changed_evidence_known"]) != 1:
                    demonstrably_trained_unknown_slots += 1
            else:
                demonstrably_trained_unknown_slots += 1
            if params_per_slot is None:
                params_per_slot = slot_scalars
            elif params_per_slot != slot_scalars:
                params_per_slot = -1  # heterogeneous slot shapes
            steps.append(sc)
    if steps:
        arr = np.asarray(steps, dtype=np.float64)
        per_slot = {"n": len(steps), "mean": round(float(arr.mean()), 2),
                    "p50": round(float(np.percentile(arr, 50)), 2),
                    "max": int(arr.max()), "total": int(arr.sum())}
    else:
        per_slot = {"n": 0}
    return {"n_allocated": vstore.exact_virtual_count(),
            "slots_updated": slots_updated,
            "orphan_optimizer_slots": orphan_optimizer_slots,
            "params_per_slot": params_per_slot,
            "optimizer_updated_scalars": optimizer_updated_scalars,
            "optimizer_steps_per_slot": per_slot}


def measure(store: vstore.VirtualParameterStore,
            trainer=None,
            participation: list[dict] | None = None) -> dict[str, Any]:
    manifest = store.manifest
    committed = manifest["initialized_rows"]
    rows_per_bank = {str(b): len(rows) for b, rows in committed.items()}
    rows_total = sum(rows_per_bank.values())
    committed_key_set = {
        (int(bank), int(row))
        for bank, rows in committed.items() for row in rows
    }
    slot_shapes = {
        bank: {row: shape for row, shape in rows.items()
               if (int(bank), int(row)) in committed_key_set}
        for bank, rows in manifest.get("initialized_slot_shapes", {}).items()
    }
    # Legacy committed rows can be sized exactly when a persisted optimizer
    # state exists. Do not treat an entire 50,000-value bank row as a slot.
    for path in store.root.glob("adam-*.npz"):
        stem = path.stem.removeprefix("adam-")
        bank_s, row_s = stem.split("-", 1)
        if (int(bank_s), int(row_s)) not in committed_key_set:
            continue
        with np.load(path) as state:
            if int(state["step_count"]) <= 0:
                continue
            slot_shapes.setdefault(bank_s, {})[row_s] = {
                "dim": int(state["A_m"].shape[0]),
                "rank": int(state["A_m"].shape[1]),
                "scalars": int(state["A_m"].size + state["B_m"].size),
            }
    # A live SlotState is an exact shape witness for its committed row even
    # when an older manifest predates initialized_slot_shapes.
    for (bank, row), slot in store._slots.items():
        if (bank, row) in committed_key_set:
            slot_shapes.setdefault(str(bank), {})[str(row)] = {
                "dim": int(slot.A.shape[0]),
                "rank": int(slot.A.shape[1]),
                "scalars": int(slot.A.numel() + slot.B.numel()),
            }
    initialized_scalars = sum(
        int(shape.get("scalars", 2 * int(shape["dim"]) * int(shape["rank"])))
        for rows in slot_shapes.values() for shape in rows.values())
    known_shape_rows = sum(len(rows) for rows in slot_shapes.values())
    unknown_shape_rows = max(0, rows_total - known_shape_rows)
    participation = participation or []

    # Build exact per-slot scalar shapes and identity sets. Never estimate a
    # union with max(counts): overlapping evidence must be deduplicated by
    # (bank,row), and legacy rows without shape evidence remain unknown.
    shape_by_key = {}
    for bank_s, rows in slot_shapes.items():
        for row_s, shape in rows.items():
            shape_by_key[(int(bank_s), int(row_s))] = int(shape["scalars"])

    committed_keys = {
        (int(bank_s), int(row_s))
        for bank_s, rows in committed.items() for row_s in rows
    }
    live_keys = set(store._slots)
    live_grad_keys = set()
    live_shape = {}
    scalars_with_grad = 0
    for key, st in store._slots.items():
        live_shape[key] = st.A.numel() + st.B.numel()
        g_count = 0
        if st.A.grad is not None:
            g_count += int((st.A.grad.abs() > 0).sum())
        if st.B.grad is not None:
            g_count += int((st.B.grad.abs() > 0).sum())
        scalars_with_grad += g_count
        if g_count:
            live_grad_keys.add(key)

    part_slots = set()
    for entry in participation:
        for bank_row in entry.get("virtual_slots_updated", []):
            part_slots.add((int(bank_row[0]), int(bank_row[1])))

    persisted_slots = set()
    for path in store.root.glob("adam-*.npz"):
        bank_s, row_s = path.stem.removeprefix("adam-").split("-", 1)
        key = (int(bank_s), int(row_s))
        if key not in committed_keys:
            continue
        with np.load(path) as state:
            if int(state["step_count"]) > 0:
                persisted_slots.add(key)
                shape_by_key[key] = int(
                    state["A_m"].size + state["B_m"].size)

    activated_keys = committed_keys | live_keys | part_slots | persisted_slots
    received_keys = live_grad_keys | part_slots | persisted_slots
    for key, n in live_shape.items():
        shape_by_key.setdefault(key, n)

    def exact_scalar_total(keys):
        known = sum(shape_by_key[key] for key in keys if key in shape_by_key)
        unknown = sum(1 for key in keys if key not in shape_by_key)
        return known, unknown

    activated_scalars, activated_unknown_slots = exact_scalar_total(
        activated_keys)
    received_scalars, received_unknown_slots = exact_scalar_total(
        received_keys)
    persisted = persisted_optimizer_scan(store)
    persisted_updated_scalars = int(
        persisted.get("optimizer_updated_scalars", 0))

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
    demonstrably_trained_scalars = int(
        persisted.get("demonstrably_trained_scalars", 0))
    demonstrably_trained_unknown_slots = int(
        persisted.get("demonstrably_trained_unknown_slots", 0))
    scalars = {
        "allocated": vstore.exact_virtual_count(),
        "initialized": initialized_scalars,
        "initialized_unknown_rows": unknown_shape_rows,
        "initialized_exact": unknown_shape_rows == 0,
        "activated": activated_scalars,
        "activated_unknown_slots": activated_unknown_slots,
        "received_gradients": received_scalars,
        "received_gradients_unknown_slots": received_unknown_slots,
        "received_gradients_scalars": scalars_with_grad,
        "optimizer_updated": persisted_updated_scalars,
        "demonstrably_trained": demonstrably_trained_scalars,
        "demonstrably_trained_unknown_slots":
            demonstrably_trained_unknown_slots,
        "demonstrably_trained_exact":
            demonstrably_trained_unknown_slots == 0,
    }
    note = ("activated and received-gradient counts are exact unions by "
            "(bank,row) identity where slot shape evidence exists; unknown "
            "legacy slots are reported separately. optimizer_updated and "
            "demonstrably_trained come from persisted optimizer state.")
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
            and scalars["demonstrably_trained"] < scalars["allocated"]
            and activated_unknown_slots == 0
            and received_unknown_slots == 0
            and unknown_shape_rows == 0
            and demonstrably_trained_unknown_slots == 0
            and persisted.get("orphan_optimizer_slots", 0) == 0),
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
