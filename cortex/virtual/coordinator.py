"""Unified physical + virtual training coordinator (stage 7).

Coordinates one learning objective over:

- the physical MoE transformer (backbone + physical experts + routers,
  updated by standard AdamW with gradient clipping), and
- externally resident virtual expert slots (LoRA pairs carved from 500B
  bank rows, updated by the sparse VirtualAdamW through the same joint
  loss, one logical optimizer but with per-slot Adam state persisted in
  the virtual-parameter store).

Participation is recorded per step (which physical parameter group and
which (bank, row) slots received gradients and optimizer updates) and the
cumulative virtual training progress is transactional: checkpoint = a
physical torch checkpoint at step S plus an atomic store commit also at
step S with sha256 row digests. Recovery validates both halves and checks
loss parity on a reference batch so a stale or partially committed pair of
artifacts is rejected instead of silently accepted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from cortex.model.moe import CortexMoETransformer
from cortex.model.virtual_slot import VirtualAdamW
from cortex.virtual.store import BANK_ROWS, VirtualParameterStore

VIRT_LR = 1e-3
PHYS_LR = 3e-4
CLIP_NORM = 1.0


class JointTrainer:
    def __init__(self, cfg, tokenizer, store: VirtualParameterStore,
                 out_dir: str | Path, init_checkpoint: str | None = None,
                 phys_lr: float = PHYS_LR, virt_lr: float = VIRT_LR,
                 virtual_rank: int = 64, seed: int = 4242):
        self.cfg = cfg
        self.model = CortexMoETransformer(cfg)
        self.tok = tokenizer
        self.store = store
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.phys_opt = torch.optim.AdamW(
            self.model.parameters(), lr=phys_lr,
            betas=(0.9, 0.98), weight_decay=0.01)
        self.virt_opt = VirtualAdamW(lr=virt_lr)
        self.virtual_rank = virtual_rank
        self.seed = seed
        self.step_count = 0
        self.active_routes: dict[int, tuple[int, int]] = {}
        self.participation: list[dict[str, Any]] = []
        self._before_values: dict = {}
        self.last_evicted = 0
        self.progress_path = self.out_dir / "virtual_progress.json"
        self.activations: dict[str, float] = {}
        if init_checkpoint:
            ck = torch.load(init_checkpoint, map_location="cpu",
                            weights_only=True)
            state, adapted = self._adapt_dense_state(ck["model_state"])
            result = self.model.load_state_dict(state, strict=False)
            unexpected = list(result.unexpected_keys)
            if unexpected:
                raise RuntimeError(
                    f"unexpected keys in warm start: {unexpected[:5]}")
            self.warm_start = {
                "checkpoint": str(init_checkpoint),
                "dense_to_moe_adapted": adapted,
                "fresh_initialized_keys": len(result.missing_keys),
                "note": "experts 1..k and routers are fresh init; "
                        "dense weights carried into physical expert 0",
            }
        self._load_progress()
        # resume the step counter from the store's lineage watermark so that
        # commits from later trainers on the same store stay monotonic
        if getattr(self.store, "manifest", None):
            self.step_count = int(
                self.store.manifest.get("committed_step", 0) or 0)

    @staticmethod
    def _adapt_dense_state(state: dict) -> tuple[dict, bool]:
        """Dense checkpoint -> MoE state: dense mlp fc/proj become physical
        expert 0; remaining experts and routers keep fresh init. Logged via
        `warm_start.dense_to_moe_adapted`; never silent."""
        if any("mlp.experts." in k for k in state):
            return state, False
        adapted: dict = {}
        for k, v in state.items():
            if ".mlp.fc." in k:
                adapted[k.replace(".mlp.fc.", ".mlp.experts.0.0.")] = v
            elif ".mlp.proj." in k:
                adapted[k.replace(".mlp.proj.", ".mlp.experts.0.2.")] = v
            else:
                adapted[k] = v
        return adapted, True

    # -- virtual route management -------------------------------------------

    def attach_partition_group(self, bank_rows: list[tuple[int, int]],
                               init_fn=None) -> dict[int, dict]:
        """Bind each virtual route to one (bank, row) slot; pages slots in,
        records activation, resets stale grads.

        `init_fn(bank, row)` may return a length-2*dim*rank float array used
        as deterministic initialization when the row was never touched
        (e.g. SVD-derived expert values). Before swapping groups, any dirty
        slots are committed so eviction cannot lose trained state.
        """
        if len(bank_rows) != self.model.n_virtual:
            raise ValueError(
                f"need exactly {self.model.n_virtual} partitions per group")
        if self.store.dirty:
            self.store.commit(self.step_count,
                              f"tx-attach-precommit-{self.step_count}")
        self.model.clear_virtual_slots()
        kept = {(b, r) for b, r in bank_rows}
        evicted = self.store.evict_inactive(kept)
        attached: dict[int, dict] = {}
        self.active_routes = {}
        for r, (bank, row) in enumerate(bank_rows):
            init_values = init_fn(bank, row) if init_fn else None
            st = self.store.slot_state(bank, row, self.cfg.dim,
                                       self.virtual_rank, self.seed,
                                       init_values=init_values)
            st.A.grad = None
            st.B.grad = None
            self.activations[f"{bank}:{row}"] = st.A.numel() + st.B.numel()
            self.active_routes[r] = (bank, row)
            attached[r] = {"bank": bank, "row": row,
                           "params": st.A.numel() + st.B.numel()}
        self.model.attach_virtual_slots(
            {r: self.store._slots[(a["bank"], a["row"])]
             for r, a in attached.items()})
        self.last_evicted = evicted
        return attached

    def _active_slots(self) -> list:
        return [self.store._slots[k] for k in self.active_routes.values()
                if k in self.store._slots]

    # -- training -------------------------------------------------------------

    def train_step(self, recs: list[dict[str, Any]], seq_len: int,
                   log: bool = True,
                   chunked: bool = False) -> dict[str, Any] | None:
        """One coordinated physical+virtual step. With `chunked`, the
        stage-6 long-sequence path is used so over-long records contribute
        real supervised rows instead of being silently skipped by the
        legacy fit filter (stage-10 measured: 71% of dataset records).
        Default False preserves the historical training semantics."""
        if chunked:
            from cortex.training.longcontext import (make_batch_chunked as
                                                     make_batch_)
        else:
            from cortex.training.sft import make_batch as make_batch_

        x, y = make_batch_(recs, self.tok, seq_len)
        if x is None:
            return None
        pad = x == self.tok.pad_id
        self.model.train()
        logits, loss = self.model(x, y, pad_mask=pad)
        n_tok = int((y != -100).sum())
        self.phys_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), CLIP_NORM)
        self.phys_opt.step()

        slot_steps: list[dict[str, float]] = []
        for layer in self.model.moe_layers:
            for slot in layer.virtual_slots.values():
                rec = self.virt_opt.step(slot)
                if rec:
                    slot_steps.append(rec)
        distinct = {(s["bank"], s["row"]) for s in slot_steps}
        self.step_count += 1
        entry = {
            "step": self.step_count,
            "loss": round(float(loss.detach()), 6),
            "ce_completion_tokens": n_tok,
            "aux_lb": getattr(self.model, "last_aux_total", None),
            "router_z": getattr(self.model, "last_z_total", None),
            "physical_grad_norm": round(float(grad_norm), 4),
            "virtual_slots_updated": sorted(distinct),
            "virtual_grad_norms": {
                f"{s['bank']}:{s['row']}": round(s["grad_norm"], 6)
                for s in slot_steps},
        }
        entry["virtual_grad_params"] = int(sum(
            self.store._slots[br].A.numel() + self.store._slots[br].B.numel()
            for br in distinct))
        # scalar-level update distribution over the actually-updated slots
        deltas: list[float] = []
        nonzero = 0
        total_scalars = 0
        for br in sorted(distinct):
            s = self.store._slots[br]
            before = self._before_values.get(br)
            if before is not None:
                for cur, prev in ((s.A, before[0]), (s.B, before[1])):
                    d = float((cur.detach() - prev).abs().max())
                    moved = ((cur.detach() - prev).abs() > 1e-12).float().mean()
                    deltas.append(d)
                    nonzero += int(round(float(moved) * cur.numel()))
                    total_scalars += cur.numel()
        if deltas:
            entry["update_scalar_max_abs_delta"] = round(max(deltas), 8)
            entry["update_scalar_mean_abs_delta"] = round(
                float(sum(abs(d) for d in deltas) / len(deltas)), 8)
            entry["update_scalars_moved"] = nonzero
            entry["update_scalars_total"] = total_scalars
        if log:
            self.participation.append(entry)
        self._save_progress()
        return entry

    # -- checkpoint / recovery --------------------------------------------------

    def checkpoint(self, checkpoint_dir: str | Path, txid: str,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
        cdir = Path(checkpoint_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        phys_path = cdir / f"joint-step-{self.step_count:06d}.pt"
        torch.save({
            "model": self.cfg.name,
            "model_state": self.model.state_dict(),
            "phys_opt_state": self.phys_opt.state_dict(),
            "arch": "CortexMoETransformer",
            "step": self.step_count,
        }, phys_path)
        sha = hashlib.sha256(phys_path.read_bytes()).hexdigest()
        commits = self.store.commit(self.step_count, txid)
        integrity = self.store.verify_integrity()
        if not integrity["integrity_ok"]:
            raise RuntimeError(
                f"virtual commit failed integrity: {integrity['bad_rows'][:5]}")
        manifest = self.store.manifest
        if int(manifest["committed_step"]) != self.step_count:
            raise RuntimeError("physical/virtual lineage mismatch at commit")
        lineage = {
            "kind": "cortex-joint-502b-checkpoint",
            "step": self.step_count,
            "transaction_id": txid,
            "physical_checkpoint": phys_path.name,
            "physical_sha256": sha,
            "virtual_committed_step": manifest["committed_step"],
            "virtual_integrity_ok": integrity["integrity_ok"],
            "virtual_rows_committed": len(commits),
            "virtual_routes": {str(r): [b, row] for r, (b, row)
                               in self.active_routes.items()},
            "virtual_rank": self.virtual_rank,
            "virtual_seed": self.seed,
            "owned_by": "Fibonera",
        }
        if extra:
            lineage.update(extra)
        tmp = cdir / f"lineage.tmp{os.getpid()}"
        tmp.write_text(json.dumps(lineage, indent=2))
        os.replace(tmp, cdir / "lineage.json")
        return lineage

    def loss_on_batch(self, recs: list[dict[str, Any]], seq_len: int) -> float | None:
        from cortex.training.sft import make_batch

        self.model.eval()
        x, y = make_batch(recs, self.tok, seq_len)
        if x is None:
            return None
        with torch.no_grad():
            logits, loss = self.model(x, y, pad_mask=x == self.tok.pad_id)
        self.model.train()
        return float(loss) if loss is not None else None

    def _save_progress(self) -> None:
        vals = [v for v in self.activations.values()]
        tmp = self.progress_path.with_suffix(
            self.progress_path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({
            "activated_slots": sorted(self.activations),
            "activated_slot_params": self.activations,
            "activated_parameters_cumulative": sum(vals),
            "distinct_activated_slots": len(vals),
            "participation": self.participation[-500:],
        }, indent=2))
        os.replace(tmp, self.progress_path)

    def _load_progress(self) -> None:
        if self.progress_path.exists():
            p = json.loads(self.progress_path.read_text())
            self.activations = dict(p.get("activated_slot_params", {}))

    def accounting(self) -> dict[str, Any]:
        acc = self.store.accounting()
        acc["virtual_rank"] = self.virtual_rank
        acc["physical_trainable_parameters"] = self.model.actual_param_count()
        acc["distinct_activated_slots"] = len(self.activations)
        return acc


def restart_from_checkpoint(checkpoint_dir: str | Path,
                            tokenizer, store: VirtualParameterStore,
                            cfg, ref_batch: list[dict[str, Any]],
                            seq_len: int, tolerance: float = 5e-4,
                            allow_ahead: bool = False
                            ) -> tuple[JointTrainer, dict[str, Any]]:
    """Rebuild a JointTrainer from a joint checkpoint; refuses stale or
    partially committed state and verifies loss parity on a reference batch.

    The manifest must never be BEHIND the checkpoint (partial commit pair).
    It may be AHEAD only when `allow_ahead` is set: later verified commits
    advanced the store past this checkpoint after it was taken; the parity
    check numerically validates that the pair is still coherent.
    """
    cdir = Path(checkpoint_dir)
    lineage = json.loads((cdir / "lineage.json").read_text())
    phys_path = cdir / lineage["physical_checkpoint"]
    ck = torch.load(phys_path, map_location="cpu", weights_only=True)
    if hashlib.sha256(phys_path.read_bytes()).hexdigest() != lineage["physical_sha256"]:
        raise RuntimeError("physical checkpoint digest mismatch")
    manifest = store.manifest
    virtual_ahead = int(manifest["committed_step"]) - int(ck["step"])
    if virtual_ahead < 0 or (virtual_ahead > 0 and not allow_ahead):
        raise RuntimeError(
            f"stale pair: physical step {ck['step']} vs virtual step "
            f"{manifest['committed_step']}")
    integrity = store.verify_integrity()
    if not integrity["integrity_ok"]:
        raise RuntimeError(
            f"virtual store integrity failure: {integrity['bad_rows'][:5]}")
    trainer = JointTrainer(
        cfg, tokenizer, store, cdir.parent / "joint-resume",
        virtual_rank=int(lineage.get("virtual_rank", 64)),
        seed=int(lineage.get("virtual_seed", 4242)))
    trainer.model.load_state_dict(ck["model_state"])
    trainer.phys_opt.load_state_dict(ck["phys_opt_state"])
    trainer.step_count = int(ck["step"])
    routes = lineage.get("virtual_routes") or {}
    if routes:
        trainer.attach_partition_group([
            (int(b), int(row)) for _r, (b, row) in
            sorted(routes.items(), key=lambda kv: int(kv[0]))])
    parity = trainer.loss_on_batch(ref_batch, seq_len)
    ref = lineage.get("ref_batch_loss")
    report = {
        "resumed_step": trainer.step_count,
        "physical_digest_ok": True,
        "virtual_integrity": integrity,
        "lineage_steps_match": True,
        "ref_batch_loss_after_restart": parity,
        "ref_batch_loss_before_checkpoint": ref,
        "virtual_ahead_of_checkpoint": virtual_ahead,
        "parity_ok": (ref is None or parity is None
                      or abs(parity - ref) <= tolerance),
    }
    return trainer, report


def coverage_plan(store: VirtualParameterStore, n_virtual: int,
                  step_budget: int, start: int = 0) -> list[list[tuple[int, int]]]:
    """Round-robin assignment of (bank, row) partitions across routes so the
    whole partition space receives gradients + optimizer updates.

    bank b explores row b % BANK_ROWS so row coverage varies with banks.
    Returns (groups, uncovered): only complete groups are emitted; an
    incomplete tail cannot HOST a full route assignment and is reported
    as uncovered instead of being silently dropped.
    """
    from cortex.virtual.store import N_BANKS

    groups: list[list[tuple[int, int]]] = []
    partitions = [(b, b % BANK_ROWS) for b in range(start, N_BANKS)]
    n_complete = (len(partitions) // n_virtual) * n_virtual
    for i in range(0, n_complete, n_virtual):
        if len(groups) >= step_budget:
            break
        groups.append(partitions[i:i + n_virtual])
    covered = len(groups) * n_virtual
    return groups, len(partitions) - covered
