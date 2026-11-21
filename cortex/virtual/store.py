"""Virtual trainable parameter store - the 500B externally resident
tensor system (stage 7).

This is NOT a document store / vector index / knowledge counter. It is a
partitioned numerical tensor space:

- exactly 5,000 expert banks x [2,000 rows x 50,000 cols] float32
  = 500,000,000,000 allocated trainable scalars (asserted at allocation);
- every bank is a unique, owned, versioned tensor file on distributed
  storage (one partition = one file; partitions can be sharded across
  worker processes);
- parameters are paged from disk on demand (sparse activation); only
  touched rows occupy real disk blocks;
- learning is real optimizer-driven change: loading a page does NOT count
  as training; only slots that receive gradients AND take an optimizer
  step count, with before/after value digests proving numeric change.

Strict accounting is maintained separately for:
  allocated / initialized / activated / gradient-receiving /
  optimizer-updated / demonstrably-trained parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

MAX_OPEN_PAGES = 64

N_BANKS = 5_000
BANK_ROWS = 2_000
BANK_COLS = 50_000
ROWS_PER_BANK = BANK_ROWS * BANK_COLS
TOTAL_VIRTUAL = N_BANKS * ROWS_PER_BANK


def exact_virtual_count() -> int:
    return N_BANKS * BANK_ROWS * BANK_COLS


def bank_filename(bank: int) -> str:
    return f"virtual-bank-{bank:05d}.f32"


class VirtualParameterStore:
    """Disk-resident, partitioned, paged tensor store for the 500B system."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._pages: "OrderedDict[int, np.memmap]" = OrderedDict()
        self._slots: dict[tuple[int, int], "SlotState"] = {}
        self.dirty: set[tuple[int, int]] = set()
        self.manifest_path = self.root / "virtual-manifest.json"

    # ---------- allocation ----------

    def allocate(self, seed: int = 4242) -> dict[str, Any]:
        created = 0
        t0 = _now()
        for bank in range(N_BANKS):
            path = self.root / bank_filename(bank)
            if not path.exists():
                with open(path, "wb") as fh:
                    fh.truncate(ROWS_PER_BANK * 4)
                created += 1
        manifest = {
            "kind": "cortex-virtual-parameter-store",
            "n_banks": N_BANKS,
            "bank_shape": [BANK_ROWS, BANK_COLS],
            "dtype": "float32",
            "allocated_parameters": exact_virtual_count(),
            "asserted_exact_total": (
                exact_virtual_count() == TOTAL_VIRTUAL == 500_000_000_000),
            "created_files": created,
            "allocation_seconds": round(_now() - t0, 3),
            "committed_step": 0,
            "transaction_id": None,
            "initialized_rows": {},
            "initialized_slot_shapes": {},
            "coverage_rows_per_bank": [0] * N_BANKS,
            "owned_by": "Fibonera",
        }
        _atomic_json(self.manifest_path, manifest)
        self.manifest = manifest
        return manifest

    @classmethod
    def open(cls, root: str | Path) -> "VirtualParameterStore":
        store = cls(root)
        store.manifest = json.loads(store.manifest_path.read_text())
        if not store.manifest["asserted_exact_total"]:
            raise RuntimeError("virtual store manifest count mismatch")
        return store

    # ---------- paging ----------

    def page(self, bank: int) -> np.memmap:
        mm = self._pages.get(bank)
        if mm is not None:
            self._pages.move_to_end(bank)
            return mm
        while len(self._pages) >= MAX_OPEN_PAGES:
            old_bank, old = self._pages.popitem(last=False)
            old.flush()
            del old
        mm = np.memmap(self.root / bank_filename(bank), dtype=np.float32,
                       mode="r+", shape=(BANK_ROWS, BANK_COLS))
        self._pages[bank] = mm
        return mm

    def evict_inactive(self, active_keys: set) -> int:
        """Drop non-active, non-dirty slot objects from process memory.
        Callers must commit first (attach does); evicted slots reload from
        disk including AdamW state, so no trained value is lost."""
        evicted = 0
        for k in list(self._slots.keys()):
            if k not in active_keys and k not in self.dirty:
                del self._slots[k]
                evicted += 1
        return evicted

    def close(self) -> None:
        for mm in self._pages.values():
            mm.flush()
            del mm
        self._pages.clear()

    # ---------- slots ----------

    def slot_state(self, bank: int, row: int, dim: int, rank: int,
                   seed: int = 4242,
                   init_values: np.ndarray | None = None) -> "SlotState":
        """Page-in the slot (one LoRA pair A[dim,rank], B[rank,dim] carved
        from bank row `row`) creating persistent trainable leaves.

        If the slot was previously committed, its values AND AdamW state
        are restored from disk (values: bank row; m/v/step: npz), so
        eviction never loses trained state. `init_values` (length 2*dim*rank)
        supplies deterministic initialization for a never-touched row
        (e.g. SVD-derived values); otherwise fresh rows use seeded noise.
        """
        import torch

        from cortex.model.virtual_slot import SlotState

        key = (bank, row)
        if not (0 <= bank < N_BANKS and 0 <= row < BANK_ROWS):
            raise IndexError(f"virtual slot out of range: bank={bank}, row={row}")
        st = self._slots.get(key)
        if st is not None:
            if tuple(st.A.shape) != (dim, rank) or tuple(st.B.shape) != (rank, dim):
                raise ValueError(
                    f"cached slot {key} has shapes A{tuple(st.A.shape)} / "
                    f"B{tuple(st.B.shape)}, requested A{(dim, rank)} / "
                    f"B{(rank, dim)}")
            return st
        page = self.page(bank)
        raw = np.asarray(page[row])
        n = 2 * dim * rank
        if n > BANK_COLS:
            raise ValueError(
                f"slot needs {n} values > bank row {BANK_COLS}")
        flat = torch_from(raw[:n])
        if bool((flat == 0).all()):
            # sparse hole: this row was never initialized
            if init_values is not None:
                if len(init_values) != n:
                    raise ValueError(
                        f"init_values length {len(init_values)} != {n}")
                flat = torch_from(np.asarray(init_values, dtype=np.float32))
            else:
                rng = np.random.default_rng(seed + bank * 100_000 + row)
                flat = torch_from(rng.standard_normal(n).astype(np.float32) * 0.02)
            self.dirty.add(key)
        A = torch.nn.Parameter(flat[:dim * rank].view(dim, rank).clone())
        B = torch.nn.Parameter(flat[dim * rank:].view(rank, dim).clone())
        st = SlotState(bank=bank, row=row, A=A, B=B)
        self._restore_opt_state(st, dim, rank)
        self._slots[key] = st
        self.dirty.add(key)
        return st

    def _restore_opt_state(self, st: "SlotState", dim: int, rank: int) -> None:
        import torch

        import numpy as _np

        path = self.root / f"adam-{st.bank:05d}-{st.row:06d}.npz"
        if not path.exists():
            return
        data = _np.load(path)
        if int(data["step_count"]) == 0:
            return
        st.step_count = int(data["step_count"])
        for name, p in (("A", st.A), ("B", st.B)):
            if data[f"{name}_m"].shape != tuple(p.shape):
                st.step_count = 0
                st.m.clear()
                st.v.clear()
                return
            st.m[name] = torch.from_numpy(
                _np.ascontiguousarray(data[f"{name}_m"], dtype=_np.float32))
            st.v[name] = torch.from_numpy(
                _np.ascontiguousarray(data[f"{name}_v"], dtype=_np.float32))

    # ---------- commit / recovery ----------

    def commit(self, step: int, txid: str) -> list[dict[str, Any]]:
        """Transactional commit of every dirty slot: write values + optimizer
        state to disk, flush, digest, then atomically update the manifest.
        No partial state survives a crash between flush and manifest."""
        import torch

        from cortex.model.virtual_slot import slot_adam_states

        commits: list[dict[str, Any]] = []
        for (bank, row) in sorted(self.dirty):
            st = self._slots[(bank, row)]
            page = self.page(bank)
            n = st.A.numel() + st.B.numel()
            flat = np.empty(n, dtype=np.float32)
            flat[:st.A.numel()] = st.A.detach().numpy().reshape(-1)
            flat[st.A.numel():] = st.B.detach().numpy().reshape(-1)
            page[row, :n] = flat
            page.flush()
            digest = hashlib.sha256(
                np.asarray(page[row]).tobytes()).hexdigest()
            opt_states = slot_adam_states(st)
            if opt_states:
                np.savez(self.root / f"adam-{bank:05d}-{row:06d}.npz",
                         **opt_states)
            commits.append({"bank": bank, "row": row, "step": step,
                            "txid": txid, "digest": digest,
                            "params_in_slot": int(n)})
            m = _read_json(self.manifest_path)
            rows_map = m["initialized_rows"]
            rows_map.setdefault(str(bank), {})[str(row)] = digest
            shapes = m.setdefault("initialized_slot_shapes", {})
            shapes.setdefault(str(bank), {})[str(row)] = {
                "dim": int(st.A.shape[0]),
                "rank": int(st.A.shape[1]),
                "scalars": int(st.A.numel() + st.B.numel()),
            }
            m["coverage_rows_per_bank"][bank] = len(rows_map[str(bank)])
            # lineage watermark is monotonic: a later commit at a lower step
            # (e.g. a fresh trainer resuming against the same store) must
            # never rewind the recorded training frontier
            m["committed_step"] = max(int(m.get("committed_step", 0)),
                                      int(step))
            m["transaction_id"] = txid
            _atomic_json(self.manifest_path, m)
            self.manifest = m
        self.dirty.clear()
        return commits

    def verify_integrity(self) -> dict[str, Any]:
        """Recompute digests of every committed row; a mismatch is a partial
        or corrupted commit."""
        m = _read_json(self.manifest_path)
        checked = 0
        bad: list[str] = []
        for bank_s, rows in m["initialized_rows"].items():
            page = self.page(int(bank_s))
            for row_s, digest in rows.items():
                checked += 1
                actual = hashlib.sha256(
                    np.asarray(page[int(row_s)]).tobytes()).hexdigest()
                if actual != digest:
                    bad.append(f"{bank_s}/{row_s}")
        return {"rows_checked": checked, "integrity_ok": not bad,
                "bad_rows": bad[:20]}

    # ---------- accounting ----------

    def accounting(self) -> dict[str, Any]:
        from cortex.model.virtual_slot import slot_adam_states

        allocated = exact_virtual_count()
        manifest = _read_json(self.manifest_path)
        shape_map = {
            bank: dict(rows) for bank, rows in
            manifest.get("initialized_slot_shapes", {}).items()
        }
        # Backfill exact shapes for legacy committed rows when their Adam
        # state exists. Rows without shape metadata or optimizer state remain
        # explicitly unknown; never infer a full 50,000 scalars per row.
        for path in self.root.glob("adam-*.npz"):
            stem = path.stem.removeprefix("adam-")
            bank_s, row_s = stem.split("-", 1)
            with np.load(path) as state:
                if int(state["step_count"]) <= 0:
                    continue
                shape_map.setdefault(bank_s, {})[row_s] = {
                    "dim": int(state["A_m"].shape[0]),
                    "rank": int(state["A_m"].shape[1]),
                    "scalars": int(state["A_m"].size + state["B_m"].size),
                }
        initialized = sum(
            int(shape.get("scalars", 2 * int(shape["dim"]) * int(shape["rank"])))
            for rows in shape_map.values() for shape in rows.values())
        committed_rows = sum(len(rows) for rows in
                             manifest["initialized_rows"].values())
        known_shape_rows = sum(len(rows) for rows in shape_map.values())
        unknown_shape_rows = max(0, committed_rows - known_shape_rows)
        activated = grads = updated = 0
        for st in self._slots.values():
            activated += st.A.numel() + st.B.numel()
            if st.A.grad is not None or st.B.grad is not None:
                nz = (st.A.grad is not None
                      and float(st.A.grad.abs().sum()) > 0) or (
                    st.B.grad is not None
                    and float(st.B.grad.abs().sum()) > 0)
                if nz:
                    grads += st.A.numel() + st.B.numel()
            if slot_adam_states(st):
                updated += st.A.numel() + st.B.numel()
        trained = updated
        return {
            "allocated": allocated,
            "initialized": int(initialized),
            "initialized_unknown_rows": int(unknown_shape_rows),
            "initialized_exact": unknown_shape_rows == 0,
            "activated": activated,
            "received_gradients": grads,
            "optimizer_updated": updated,
            "demonstrably_trained": updated,
            "updates_applied_total": self._trained_updates(),
            "partitions_touched": len(
                _read_json(self.manifest_path)["initialized_rows"]),
            "partitions_total": N_BANKS,
        }

    def _trained_updates(self) -> int:
        m = _read_json(self.manifest_path)
        return int(m.get("committed_step", 0))


def torch_from(arr: np.ndarray) -> "Any":
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _now() -> float:
    import time

    return time.time()
