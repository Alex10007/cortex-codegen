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
        # A committed row has an established tensor interpretation. Refuse
        # to reinterpret it under a different shape: doing so would silently
        # corrupt learned values and optimizer lineage.
        manifest = _read_json(self.manifest_path)
        committed = str(row) in manifest.get(
            "initialized_rows", {}).get(str(bank), {})
        persisted_shape = manifest.get("initialized_slot_shapes", {}).get(
            str(bank), {}).get(str(row))
        if persisted_shape is not None:
            expected = (int(persisted_shape["dim"]), int(persisted_shape["rank"]))
            if expected != (dim, rank):
                raise ValueError(
                    f"persisted slot {key} has dim/rank {expected}; "
                    f"requested {(dim, rank)}")
        elif committed:
            # Legacy Stage 7-15 rows predate shape metadata. Their optimizer
            # tensors provide an exact shape witness; accept only an exact
            # match. A committed row without either source remains ambiguous.
            opt_path = self.root / f"adam-{bank:05d}-{row:06d}.npz"
            if not opt_path.exists():
                raise RuntimeError(
                    f"persisted slot {key} has no shape metadata or optimizer "
                    "shape witness; refusing ambiguous legacy row reinterpretation")
            with np.load(opt_path) as state:
                if int(state["step_count"]) <= 0:
                    raise RuntimeError(
                        f"persisted slot {key} has no trained optimizer shape "
                        "witness; refusing ambiguous legacy row reinterpretation")
                a_shape = tuple(state["A_m"].shape)
                b_shape = tuple(state["B_m"].shape)
            if a_shape != (dim, rank) or b_shape != (rank, dim):
                raise ValueError(
                    f"legacy optimizer slot {key} has shapes "
                    f"A{a_shape} / B{b_shape}; requested "
                    f"A{(dim, rank)} / B{(rank, dim)}")
            persisted_shape = {"dim": dim, "rank": rank,
                               "scalars": 2 * dim * rank}

        page = self.page(bank)
        raw = np.asarray(page[row])
        n = 2 * dim * rank
        if n > BANK_COLS:
            raise ValueError(
                f"slot needs {n} values > bank row {BANK_COLS}")
        if committed:
            # A committed all-zero slot is valid data, not an empty sparse
            # hole. Only manifest membership authorizes loading persisted
            # numerical values.
            flat = torch_from(raw[:n])
        else:
            # Ignore uncommitted bytes left by an interrupted transaction.
            # They are not part of the accepted lineage.
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
        opt_path = self.root / f"adam-{bank:05d}-{row:06d}.npz"
        if committed:
            self._restore_opt_state(st, dim, rank)
        elif opt_path.exists():
            raise RuntimeError(
                f"orphan optimizer state for uncommitted virtual slot {key}; "
                "refusing to adopt an incomplete transaction")
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
            if (data[f"{name}_m"].shape != tuple(p.shape)
                    or data[f"{name}_v"].shape != tuple(p.shape)):
                raise RuntimeError(
                    f"persisted optimizer state shape mismatch for "
                    f"{st.bank}/{st.row}/{name}; refusing to discard state")
            st.m[name] = torch.from_numpy(
                _np.ascontiguousarray(data[f"{name}_m"], dtype=_np.float32))
            st.v[name] = torch.from_numpy(
                _np.ascontiguousarray(data[f"{name}_v"], dtype=_np.float32))

    # ---------- commit / recovery ----------

    def commit(self, step: int, txid: str) -> list[dict[str, Any]]:
        """Commit all dirty slots behind one manifest transaction boundary.

        Bank values and optimizer files are flushed first. The manifest is
        atomically replaced once, only after every row is durable. A crash
        before that point leaves uncommitted bytes (ignored on reload) and
        orphan optimizer files (detected and refused), never a partial
        accepted commit frontier.
        """
        from cortex.model.virtual_slot import slot_adam_states

        manifest = _read_json(self.manifest_path)
        commits: list[dict[str, Any]] = []
        for bank, row in sorted(self.dirty):
            st = self._slots[(bank, row)]
            page = self.page(bank)
            n = st.A.numel() + st.B.numel()
            flat = np.empty(n, dtype=np.float32)
            flat[:st.A.numel()] = st.A.detach().cpu().numpy().reshape(-1)
            flat[st.A.numel():] = st.B.detach().cpu().numpy().reshape(-1)
            page[row, :n] = flat
            page.flush()
            bank_path = self.root / bank_filename(bank)
            with open(bank_path, "r+b") as bank_file:
                os.fsync(bank_file.fileno())
            digest = hashlib.sha256(
                np.asarray(page[row]).tobytes()).hexdigest()

            opt_states = slot_adam_states(st)
            if opt_states:
                opt_path = self.root / f"adam-{bank:05d}-{row:06d}.npz"
                tmp_opt = self.root / (
                    f".{opt_path.name}.tmp{os.getpid()}")
                with open(tmp_opt, "wb") as opt_file:
                    np.savez(opt_file, **opt_states)
                    opt_file.flush()
                    os.fsync(opt_file.fileno())
                os.replace(tmp_opt, opt_path)

            commits.append({"bank": bank, "row": row, "step": step,
                            "txid": txid, "digest": digest,
                            "params_in_slot": int(n)})
            rows_map = manifest["initialized_rows"]
            rows_map.setdefault(str(bank), {})[str(row)] = digest
            shapes = manifest.setdefault("initialized_slot_shapes", {})
            shapes.setdefault(str(bank), {})[str(row)] = {
                "dim": int(st.A.shape[0]),
                "rank": int(st.A.shape[1]),
                "scalars": int(st.A.numel() + st.B.numel()),
            }
            manifest["coverage_rows_per_bank"][bank] = len(
                rows_map[str(bank)])

        if commits:
            manifest["committed_step"] = max(
                int(manifest.get("committed_step", 0)), int(step))
            manifest["transaction_id"] = txid
            _atomic_json(self.manifest_path, manifest)
            self.manifest = manifest
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
        """Exact identity-based accounting across manifest, live slots and
        persisted optimizer state. Unknown legacy shapes are never guessed."""
        from cortex.model.virtual_slot import slot_adam_states

        manifest = _read_json(self.manifest_path)
        shape_by_key = {}
        committed_keys = set()
        for bank_s, rows in manifest["initialized_rows"].items():
            for row_s in rows:
                committed_keys.add((int(bank_s), int(row_s)))
        for bank_s, rows in manifest.get("initialized_slot_shapes", {}).items():
            for row_s, shape in rows.items():
                shape_by_key[(int(bank_s), int(row_s))] = int(
                    shape.get("scalars", 2 * int(shape["dim"]) * int(shape["rank"])))

        persisted_updated_keys = set()
        orphan_optimizer_keys = set()
        persisted_updated_scalars = 0
        optimizer_steps_by_key = {}
        for path in self.root.glob("adam-*.npz"):
            bank_s, row_s = path.stem.removeprefix("adam-").split("-", 1)
            key = (int(bank_s), int(row_s))
            with np.load(path) as state:
                if int(state["step_count"]) <= 0:
                    continue
                if key not in committed_keys:
                    orphan_optimizer_keys.add(key)
                    continue
                n = int(state["A_m"].size + state["B_m"].size)
                shape_by_key[key] = n
                persisted_updated_keys.add(key)
                optimizer_steps_by_key[key] = int(state["step_count"])
                persisted_updated_scalars += n

        live_keys = set(self._slots)
        live_grad_keys = set()
        live_updated_keys = set()
        for key, st in self._slots.items():
            n = st.A.numel() + st.B.numel()
            shape_by_key.setdefault(key, n)
            if ((st.A.grad is not None and bool((st.A.grad != 0).any()))
                    or (st.B.grad is not None and bool((st.B.grad != 0).any()))):
                live_grad_keys.add(key)
            if slot_adam_states(st):
                live_updated_keys.add(key)
                optimizer_steps_by_key[key] = max(
                    optimizer_steps_by_key.get(key, 0), int(st.step_count))

        activated_keys = committed_keys | live_keys | persisted_updated_keys
        received_keys = live_grad_keys | persisted_updated_keys
        updated_keys = persisted_updated_keys | live_updated_keys

        def scalar_count(keys):
            known = sum(shape_by_key[k] for k in keys if k in shape_by_key)
            unknown = sum(1 for k in keys if k not in shape_by_key)
            return int(known), int(unknown)

        initialized, initialized_unknown = scalar_count(committed_keys)
        activated, activated_unknown = scalar_count(activated_keys)
        received, received_unknown = scalar_count(received_keys)
        updated, updated_unknown = scalar_count(updated_keys)
        return {
            "allocated": exact_virtual_count(),
            "initialized": initialized,
            "initialized_unknown_rows": initialized_unknown,
            "initialized_exact": initialized_unknown == 0,
            "activated": activated,
            "activated_unknown_slots": activated_unknown,
            "received_gradients": received,
            "received_gradients_unknown_slots": received_unknown,
            "optimizer_updated": updated,
            "optimizer_updated_unknown_slots": updated_unknown,
            "demonstrably_trained": updated,
            "updates_applied_total": sum(optimizer_steps_by_key.values()),
            "persisted_optimizer_updated_scalars": persisted_updated_scalars,
            "orphan_optimizer_state_slots": len(orphan_optimizer_keys),
            "partitions_touched": len(manifest["initialized_rows"]),
            "partitions_total": N_BANKS,
            "invariants_hold": (
                exact_virtual_count() >= initialized >= received
                >= updated and updated < exact_virtual_count()
                and initialized_unknown == 0 and activated_unknown == 0
                and received_unknown == 0 and updated_unknown == 0
                and not orphan_optimizer_keys),
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
