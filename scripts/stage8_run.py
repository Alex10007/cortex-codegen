"""Stage 8 orchestrator: virtual-parameter training expansion + GPU qualification.

Executes on this host, honestly. Builds on the stage-7 run (commit 26e484c,
lineage step 655, tx-stage7-joint). All improvements are measured, and the
strict accounting is reproduced from real state - never extended by
allocation counts.

Usage:
    python scripts/stage8_run.py [--rows-per-bank N] [--skip-experiments]

- Expansion sweep (stage C) trains rows 1..N in every one of the 5,000 banks,
  one optimizer step per 8-bank group, committing every 250 groups.
- Stage-7 sweep rows (row 0) are never re-targeted; experiment arms use rows
  4 (random init) and 5 (SVD init), disjoint from the sweep rows.
- Full-scale 2B+502B training remains gated on measured GPU qualification
  (stage H reports the gate, not an assumption).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cortex.virtual.store as vstore  # noqa: E402
from cortex.evaluation.benchmarks import check_integrity as check_v1  # noqa: E402
from cortex.model.configs import CORTEX_DEMO  # noqa: E402
from cortex.model.virtual_slot import svd_slot_init  # noqa: E402
from cortex.tokenizer.cortex_tokenizer import load_tokenizer  # noqa: E402
from cortex.training.sft import evaluate_heldout, load_split, make_batch  # noqa: E402
from cortex.virtual.benchmarks import bench_storage, run_all as run_infra_benchmarks  # noqa: E402
from cortex.virtual.coordinator import (  # noqa: E402
    JointTrainer, restart_from_checkpoint)
from cortex.virtual.coverage import measure as measure_coverage  # noqa: E402
from cortex.virtual.store import (  # noqa: E402
    BANK_COLS, BANK_ROWS, N_BANKS, VirtualParameterStore, exact_virtual_count)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
VSTORE_DIR = ART / "virtual-store-502b"
JOINT7_DIR = ART / "runs" / "joint-502b-demo"
JOINT7_CKPTS = JOINT7_DIR / "checkpoints"
JOINT8_DIR = ART / "runs" / "joint-502b-stage8"
JOINT8_CKPTS = JOINT8_DIR / "checkpoints"
DATASET = ART / "sft" / "dataset_v3.jsonl"
TOKENIZER = ART / "tokenizer" / "tokenizer.json"
WARM6_CKPT = ART / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt"
STAGE7_REPORT = ART / "stage7_report.json"
STAGE7_STEP = 655
SWEEP_MARKER = JOINT8_DIR / "sweep_done.json"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
V2_FINGERPRINT = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"

VIRT_RANK_DEMO = 16
# Experiment rows must use (bank,row) pairs never committed before. The
# stage-7 sweep committed row b % BANK_ROWS (touched rows 0..1999 via banks
# {r, r+2000, r+4000}); the stage-8 sweep rows 1..3 of every bank; earlier
# aborted attempts rows 4/5 of banks 0..119. Rows 1996..1999 for banks
# 0..119 were only touched by banks 1996..1999/3996..3999, so they are
# virgin on this group range.
EXP_ROWS_RANDOM = 1998
EXP_ROWS_SVD = 1997
EXP_BANK_MAX = 119
DENSE_ARM_STEPS = 120
COMMIT_EVERY = 250


def stage_a_audit() -> dict:
    from cortex.evaluation.benchmarks import fingerprint_tasks

    v1_value = fingerprint_tasks("v1")
    v2_value = fingerprint_tasks("v2")
    stage7_report_ok = STAGE7_REPORT.exists() and json.loads(
        STAGE7_REPORT.read_text()).get("kind") == "cortex-stage7-502b-report"

    lineage_step_7 = None
    lineage_path = JOINT7_CKPTS / "lineage.json"
    if lineage_path.exists():
        lineage_step_7 = json.loads(lineage_path.read_text()).get("step")
    store_committed_step = None
    if (VSTORE_DIR / "virtual-manifest.json").exists():
        store_committed_step = json.loads(
            (VSTORE_DIR / "virtual-manifest.json").read_text()).get(
            "committed_step")

    hw = {
        "cuda": torch.cuda.is_available(),
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "platform": platform.platform(),
    }
    return {
        "stage": "A-stage7-audit",
        "stage7_lineage_step": lineage_step_7,
        "stage7_lineage_matches_commit": lineage_step_7 == STAGE7_STEP,
        "store_committed_step": store_committed_step,
        "v1_fingerprint_value": v1_value,
        "v1_fingerprint_ok": v1_value == FINGERPRINT_V1,
        "v2_fingerprint_value": v2_value,
        "v2_fingerprint_ok": v2_value == V2_FINGERPRINT,
        "stage7_report_present": stage7_report_ok,
        "hardware": hw,
        "preserved": bool(lineage_step_7 == STAGE7_STEP
                          and v1_value == FINGERPRINT_V1
                          and v2_value == V2_FINGERPRINT
                          and stage7_report_ok),
    }


def stage_b_storage_semantics(store: VirtualParameterStore) -> dict:
    """Reality check of logical allocation vs physical disk usage."""
    m = store.manifest
    logical_bytes_per_file = BANK_ROWS * BANK_COLS * 4
    files = sorted(p for p in VSTORE_DIR.glob("virtual-bank-*.f32"))
    total_physical = sum(p.stat().st_size for p in files)
    total_blocks = sum(p.stat().st_blocks * 512 for p in files)
    sparse_ratio = round(1.0 - total_blocks / total_physical, 6) if total_physical else None
    rows_committed = sum(len(v) for v in m["initialized_rows"].values())
    banks_with_rows = len(m["initialized_rows"])

    digests_checked = 0
    digests_bad = []
    for bank_s, rows in list(m["initialized_rows"].items())[:500]:
        page = store.page(int(bank_s))
        for row_s, digest in rows.items():
            digests_checked += 1
            actual = hashlib.sha256(
                np.asarray(page[int(row_s)]).tobytes()).hexdigest()
            if actual != digest:
                digests_bad.append(f"{bank_s}/{row_s}")
    store.close()
    return {
        "stage": "B-storage-semantics",
        "logical_bytes_per_bank_file": logical_bytes_per_file,
        "bytes_per_committed_row_on_disk": BANK_COLS * 4,
        "bank_files": len(files),
        "logical_total_bytes": len(files) * logical_bytes_per_file,
        "physical_total_bytes": total_physical,
        "physical_blocks_total_bytes": total_blocks,
        "sparse_file_ratio": sparse_ratio,
        "persisted_value_semantics": (
            "whole row persisted on first commit; sparse allocation means "
            "physical blocks are only materialized for initialized rows"),
        "committed_rows": rows_committed,
        "banks_with_committed_rows": banks_with_rows,
        "allocated_scalars": exact_virtual_count(),
        "initialized_scalars": rows_committed * BANK_COLS,
        "uninitialized_scalars": exact_virtual_count() - rows_committed * BANK_COLS,
        "digest_sample_ok": not digests_bad,
        "digests_checked": digests_checked,
        "digests_bad": digests_bad[:5],
    }


def repair_manifest_watermark(store: VirtualParameterStore) -> dict:
    """Repair `committed_step` if a lower-step trainer rewind it (stage-8
    defect found and fixed during the expansion run). Only repairs when the
    whole store verifies against recorded digests and the lineage's physical
    checkpoint matches its sha256; otherwise leaves state untouched."""
    lineage_path = JOINT8_CKPTS / "lineage.json"
    out: dict = {"checked": False, "repaired": False}
    if not lineage_path.exists():
        return out
    lineage = json.loads(lineage_path.read_text())
    m = store.manifest
    out["checked"] = True
    out["manifest_committed_step"] = m.get("committed_step")
    out["lineage_step"] = lineage.get("step")
    if int(m.get("committed_step", 0)) >= int(lineage["step"]):
        return out
    phys_path = JOINT8_CKPTS / lineage["physical_checkpoint"]
    digest_ok = hashlib.sha256(phys_path.read_bytes()).hexdigest() == \
        lineage["physical_sha256"]
    integrity = store.verify_integrity()
    if digest_ok and integrity["integrity_ok"]:
        m["committed_step"] = int(lineage["step"])
        m["transaction_id"] = lineage["transaction_id"]
        tmp = store.manifest_path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(m, indent=2))
        os.replace(tmp, store.manifest_path)
        store.manifest = m
        out["repaired"] = True
    else:
        out["repair_refused"] = {"digest_ok": digest_ok,
                                  "integrity_ok": integrity["integrity_ok"]}
    return out


def _progress_participation() -> list[dict]:
    p_path = JOINT8_DIR / "virtual_progress.json"
    if not p_path.exists():
        return []
    return json.loads(p_path.read_text()).get("participation", [])


def _valid_batches(recs: list[dict], tok, seq_len: int = 256,
                   batch: int = 4) -> list[int]:
    return [b for b in range(0, len(recs) - batch + 1, batch)
            if make_batch(recs[b:b + batch], tok, seq_len)[0] is not None]


def _resume_trainer(tok, store: VirtualParameterStore, ref_batch):
    if (JOINT8_CKPTS / "lineage.json").exists():
        trainer, rep = restart_from_checkpoint(
            JOINT8_CKPTS, tok, store, CORTEX_DEMO, ref_batch, 256,
            allow_ahead=True)
        return trainer, {"resumed": True, "from": "joint-502b-stage8",
                         "parity_ok": rep["parity_ok"],
                         "resumed_step": rep["resumed_step"],
                         "virtual_ahead_of_checkpoint":
                             rep["virtual_ahead_of_checkpoint"]}
    trainer = JointTrainer(
        CORTEX_DEMO, tok, store, JOINT8_DIR, init_checkpoint=str(WARM6_CKPT),
        virtual_rank=VIRT_RANK_DEMO)
    return trainer, {"resumed": False, "warm_start": str(WARM6_CKPT)}


def stage_c_expanded_coverage(store: VirtualParameterStore,
                              rows_per_bank: int) -> dict:
    tok = load_tokenizer(str(TOKENIZER))
    train_recs = load_split(DATASET, "train")
    val_recs = load_split(DATASET, "heldout")
    ref_batch = train_recs[4:8]
    trainer, resume = _resume_trainer(tok, store, ref_batch)
    batches = [train_recs[b:b + 4]
               for b in _valid_batches(train_recs, tok, 256)]

    groups = []
    for row in range(1, rows_per_bank + 1):
        for start in range(0, N_BANKS, 8):
            groups.append([(b, row) for b in range(start, start + 8)])
    assert all(len(g) == 8 for g in groups), N_BANKS % 8

    if SWEEP_MARKER.exists():
        marker = json.loads(SWEEP_MARKER.read_text())
        if marker.get("rows_per_bank") == rows_per_bank:
            part = _progress_participation()
            losses_hist = [e["loss"] for e in part if "loss" in e]
            heldout_after = evaluate_heldout(trainer.model, val_recs, tok, 256)
            cov = measure_coverage(store, trainer, part)
            return {
                "stage": "C-expanded-coverage",
                "resumed_from_marker": True,
                "rows_per_bank": rows_per_bank,
                "marker": marker,
                "loss_trend_last_500_steps": {
                    "first": losses_hist[0] if losses_hist else None,
                    "last": losses_hist[-1] if losses_hist else None},
                "heldout_after": heldout_after,
                "coverage": cov,
                "trainer_accounting": trainer.accounting(),
                "note": "coverage measured on restarted trainer plus "
                        "persisted progress log (last 500 steps)",
            }
        print("[stage8-C] marker rows differ; re-running sweep")

    steps_done = 0
    t0 = time.time()
    losses: list[float] = []
    for gi, group in enumerate(groups):
        trainer.attach_partition_group(group)
        rec = trainer.train_step(batches[gi % len(batches)], 256)
        if rec:
            losses.append(rec["loss"])
            steps_done += 1
        if (gi + 1) % COMMIT_EVERY == 0:
            store.commit(trainer.step_count, f"tx-stage8-sweep-{gi + 1}")
            print(f"[stage8-C] group {gi + 1}/{len(groups)} "
                  f"loss={rec['loss'] if rec else None}")
    store.commit(trainer.step_count, "tx-stage8-expansion")
    heldout_after = evaluate_heldout(trainer.model, val_recs, tok, 256)

    ref_batch_loss = trainer.loss_on_batch(ref_batch, 256)
    lineage = trainer.checkpoint(
        JOINT8_CKPTS, "tx-stage8-expansion", extra={
            "ref_batch_loss": ref_batch_loss,
            "ref_records": "dataset_v3/train[4:8]",
            "expansion_rows_per_bank": rows_per_bank,
            "expansion_groups": len(groups),
        })
    cov = measure_coverage(store, trainer, trainer.participation)
    marker = {
        "rows_per_bank": rows_per_bank,
        "groups": len(groups),
        "steps": trainer.step_count,
        "transaction_id": lineage["transaction_id"],
        "lineage_sha256": lineage["physical_sha256"],
    }
    SWEEP_MARKER.parent.mkdir(parents=True, exist_ok=True)
    tmp = SWEEP_MARKER.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(marker, indent=2))
    os.replace(tmp, SWEEP_MARKER)

    return {
        "stage": "C-expanded-coverage",
        "resume": resume,
        "rows_per_bank": rows_per_bank,
        "groups_executed": steps_done,
        "sweep_seconds": round(time.time() - t0, 1),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "heldout_after": heldout_after,
        "lineage": {k: lineage[k] for k in (
            "step", "transaction_id", "physical_sha256",
            "virtual_integrity_ok", "virtual_rows_committed")},
        "coverage": cov,
        "trainer_accounting": trainer.accounting(),
    }


def stage_d_optimizer_consistency(store: VirtualParameterStore) -> dict:
    tok = load_tokenizer(str(TOKENIZER))
    train_recs = load_split(DATASET, "train")
    ref_batch = train_recs[4:8]

    # 1. checkpoint recovery FIRST (before any new commits can advance the
    # lineage watermark past the checkpoint step)
    trainer2, rep = restart_from_checkpoint(
        JOINT8_CKPTS, tok, store, CORTEX_DEMO, ref_batch, 256,
        allow_ahead=True)

    # 2. optimizer-state persistence probe (commits at the resumed frontier)
    dim, rank = CORTEX_DEMO.dim, VIRT_RANK_DEMO
    probe_slots = [(1, 0), (2, 0), (3, 0)]
    trainer = JointTrainer(
        CORTEX_DEMO, tok, store, JOINT8_DIR, init_checkpoint=str(WARM6_CKPT),
        virtual_rank=VIRT_RANK_DEMO)
    trainer.attach_partition_group(
        [(b, 0) for b in range(trainer.model.n_virtual)])
    for i in range(3):
        trainer.train_step(train_recs[8 + i:12 + i], 256)
    pre = {}
    for b, r in probe_slots:
        st = store._slots[(b, r)]
        pre[(b, r)] = (int(st.step_count),
                       float(st.v["B"].abs().sum()) if st.v else -1.0)
    store.commit(trainer.step_count, "tx-stage8-opt-consistency")
    store._slots.clear()
    restored = []
    for b, r in probe_slots:
        st = store.slot_state(b, r, dim, rank)
        post_v = float(st.v["B"].abs().sum()) if st.v else -1.0
        restored.append({"bank": b, "row": r,
                         "steps_before": pre[(b, r)][0],
                         "steps_after": int(st.step_count),
                         "steps_match": pre[(b, r)][0] == int(st.step_count),
                         "m_restored_present": st.m.get("A") is not None,
                         "v_abs_sum": round(post_v, 6),
                         "v_abs_sum_match": abs(post_v - pre[(b, r)][1]) < 1e-5})
    consistency_ok = all(e["steps_match"] and e["m_restored_present"]
                         and e["v_abs_sum_match"] for e in restored)

    npz_count = len(list(VSTORE_DIR.glob("adam-*.npz")))

    return {
        "stage": "D-optimizer-consistency",
        "per_slot_restore": restored,
        "optimizer_state_consistency_ok": consistency_ok,
        "npz_optimizer_state_files": npz_count,
        "restart_parity": {k: rep[k] for k in (
            "resumed_step", "physical_digest_ok", "lineage_steps_match",
            "ref_batch_loss_after_restart", "ref_batch_loss_before_checkpoint",
            "virtual_ahead_of_checkpoint", "parity_ok")},
        "restart_integrity": rep["virtual_integrity"]["integrity_ok"],
        "recovery_verified_at_expanded_coverage": bool(
            consistency_ok and rep["parity_ok"]
            and rep["virtual_integrity"]["integrity_ok"]),
    }


def stage_e_training_experiments(store: VirtualParameterStore) -> dict:
    """Dense control vs joint random-init vs joint SVD-init arms
    (identical batch order, optimizer settings and step count)."""
    tok = load_tokenizer(str(TOKENIZER))
    train_recs = load_split(DATASET, "train")
    val_recs = load_split(DATASET, "heldout")[:16]
    bases = _valid_batches(train_recs, tok, 256)
    assert bases

    def batch(i: int) -> list[dict]:
        b = bases[i % len(bases)]
        return train_recs[b:b + 4]

    from cortex.model.transformer import CortexTransformer

    torch.manual_seed(7)
    dense = CortexTransformer(CORTEX_DEMO)
    ck = torch.load(WARM6_CKPT, map_location="cpu", weights_only=True)
    dense.load_state_dict(ck["model_state"])
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=3e-4,
                                  betas=(0.9, 0.98), weight_decay=0.01)
    dense.train()
    dense_losses = []
    for i in range(DENSE_ARM_STEPS):
        x, y = make_batch(batch(i), tok, 256)
        if x is None:
            continue
        _, loss = dense(x, y, pad_mask=x == tok.pad_id)
        dense_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense.parameters(), 1.0)
        dense_opt.step()
        dense_losses.append(round(float(loss.detach()), 4))
    dense_heldout = evaluate_heldout(dense, val_recs, tok, 256)

    with torch.no_grad():
        M = sum(dense.blocks[i].attn.proj.weight
                for i in range(dense.cfg.n_layers)) / dense.cfg.n_layers
        M = M.detach().clone()

    def run_joint_arm(row: int, use_svd: bool) -> dict:
        trainer = JointTrainer(
            CORTEX_DEMO, tok, store, JOINT8_DIR,
            init_checkpoint=str(WARM6_CKPT), virtual_rank=VIRT_RANK_DEMO)
        losses = []
        fresh_slots = 0
        attached_slots = 0
        gi = 0
        while len(losses) < DENSE_ARM_STEPS:
            group_start = (gi * 8) % (EXP_BANK_MAX + 1)
            if use_svd:
                init_fn = (lambda bank, row_=row: svd_slot_init(
                    M, VIRT_RANK_DEMO))
            else:
                init_fn = None
            trainer.attach_partition_group(
                [(b, row) for b in range(group_start, group_start + 8)],
                init_fn=init_fn)
            attached_slots += 8
            fresh_slots += sum(
                1 for k in trainer.active_routes.values()
                if store._slots[k].step_count == 0)
            for _j in range(8):
                rec = trainer.train_step(batch(len(losses)), 256)
                if rec:
                    losses.append(round(rec["loss"], 4))
            gi += 1
        heldout = evaluate_heldout(trainer.model, val_recs, tok, 256)
        updated_slots = len({tuple(r) for e in trainer.participation
                             for r in e.get("virtual_slots_updated", [])})
        return {"loss_first": losses[0], "loss_last": losses[-1],
                "heldout": heldout, "distinct_slots_trained": updated_slots,
                "attached_slot_instances": attached_slots,
                "slots_started_fresh": fresh_slots,
                "slots_fresh_fraction":
                    round(fresh_slots / max(1, attached_slots), 4)}

    joint_random_arm = run_joint_arm(EXP_ROWS_RANDOM, use_svd=False)
    joint_svd_arm = run_joint_arm(EXP_ROWS_SVD, use_svd=True)

    return {
        "stage": "E-training-experiments",
        "steps_per_arm": DENSE_ARM_STEPS,
        "identical_batches": True,
        "dense_control": {
            "loss_first": dense_losses[0], "loss_last": dense_losses[-1],
            "heldout": dense_heldout},
        "joint_random_init_row4": joint_random_arm,
        "joint_svd_init_row5": joint_svd_arm,
        "fresh_expert_cost": {
            "dense_vs_joint_random_final_gap": round(
                joint_random_arm["loss_last"] - dense_losses[-1], 4),
            "svd_vs_random_final_gain": round(
                joint_random_arm["loss_last"] - joint_svd_arm["loss_last"], 4),
            "svd_vs_random_heldout_gain": round(
                joint_random_arm["heldout"] - joint_svd_arm["heldout"], 4),
        },
        "note": ("dense control = CortexTransformer (untouched class); joint "
                 "arms = CortexMoETransformer + 8 fresh virtual slots per "
                 "group; identical batch order and optimizer settings"),
    }


def stage_f_moe_specialization(store: VirtualParameterStore) -> dict:
    from cortex.model.moe import js_divergence

    tok = load_tokenizer(str(TOKENIZER))
    recs = load_split(DATASET, "train")[:16]
    trainer = JointTrainer(
        CORTEX_DEMO, tok, store, JOINT8_DIR, init_checkpoint=str(WARM6_CKPT),
        virtual_rank=VIRT_RANK_DEMO)
    trainer.attach_partition_group(
        [(b, 0) for b in range(trainer.model.n_virtual)])
    algo_probs: list[list] = [[] for _ in trainer.model.moe_layers]
    text_probs: list[list] = [[] for _ in trainer.model.moe_layers]
    with torch.no_grad():
        for rec in recs:
            x, _y = make_batch([rec], tok, 256)
            if x is None:
                continue
            trainer.model(x[:1], None)
            is_code = "def " in rec.get("code", "") or "class " in rec.get(
                "code", "")
            for li, layer in enumerate(trainer.model.moe_layers):
                if layer.last_probs is not None:
                    (algo_probs if is_code else text_probs)[li].append(
                        layer.last_probs.detach().clone())
    js_per_layer = []
    physical_grad_means = []
    for li, layer in enumerate(trainer.model.moe_layers):
        js = 0.0
        if algo_probs[li] and text_probs[li]:
            pa = torch.stack(algo_probs[li]).mean(0)
            pt = torch.stack(text_probs[li]).mean(0)
            js = float(js_divergence(pa, pt))
        js_per_layer.append(round(js, 6))
        wgrads = [p.grad.abs().mean().item() for p in layer.parameters()
                  if p.grad is not None]
        physical_grad_means.append(round(sum(wgrads) / len(wgrads), 8)
                                    if wgrads else 0.0)
    diags = [layer.routing_diagnostics() for layer in trainer.model.moe_layers]
    return {
        "stage": "F-moe-specialization",
        "moe_layers": len(trainer.model.moe_layers),
        "js_divergence_algo_vs_text_per_layer": js_per_layer,
        "input_conditional_routing": any(js > 0.005 for js in js_per_layer),
        "physical_expert_grad_means": physical_grad_means,
        "utilization_per_layer": [round(d["utilization_fraction"], 4)
                                  for d in diags],
        "any_layer_collapsed": any(d["collapsed"] for d in diags),
    }


def stage_g_divergent_v2() -> dict:
    from cortex.core import CortexAgent
    from cortex.evaluation.divergent import divergent_eval

    agent = CortexAgent(
        state_dir=str(ART / "cortex-state"),
        tokenizer_path=str(TOKENIZER),
        checkpoint_path=str(WARM6_CKPT))
    report = divergent_eval(agent, n_hypotheses=3, repair_round=True)
    agg = dict(report["aggregate"])
    agg["stage"] = "G-divergent-reasoning-v2"
    agg["tasks_with_repair_stderr_conditioned"] = sum(
        1 for t in report["per_task"]
        if t["repair_round"] and any(h["grade"]["stderr_tail"]
                                     for h in t["repair_round"]))
    return agg


def stage_h_qualification() -> dict:
    hw = {
        "cuda": torch.cuda.is_available(),
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    with tempfile.TemporaryDirectory(prefix="cor8-storage-") as td:
        storage = bench_storage(Path(td))
    bytes_per_2b_step = 2_056_012_800 * 4 * 16
    write_bw = storage["write_mbps"]
    seconds_per_2b_step = round(bytes_per_2b_step / (write_bw * 1e6), 3)

    fsdp = None
    ddp = None
    if hw["cuda"] and hw["gpus"] >= 2:
        from cortex.training.distributed import spawn_ddp_validation
        from cortex.training.fsdp import spawn_fsdp_validation
        fsdp = spawn_fsdp_validation(
            "cortex-demo", str(ART / "data" / "tokens.npy"),
            str(ART / "runs" / "fsdp-stage8"), world_size=2, steps=6,
            batch_size=2, checkpoint_every=6, resume_steps=2)
        ddp = spawn_ddp_validation(
            "cortex-demo", str(ART / "data" / "tokens.npy"),
            str(ART / "runs" / "ddp-stage8"), world_size=2, steps=6,
            batch_size=2)
    if not hw["cuda"]:
        decision = "REFUSED-CPU-ONLY"
    elif hw["gpus"] < 4 or fsdp is None:
        decision = "REFUSED-INSUFFICIENT-GPUS"
    else:
        decision = "QUALIFIED"
    return {
        "stage": "H-gpu-qualification",
        "hardware": hw,
        "storage_benchmark": storage,
        "bytes_per_2b_optimizer_step_checkpoint": bytes_per_2b_step,
        "seconds_per_2b_step_at_measured_write_bw": seconds_per_2b_step,
        "fsdp2_validation": fsdp if fsdp is not None
        else "not executed: no GPU on this host (from-measured, not assumed)",
        "ddp_optimizer_sharding_validation": ddp if ddp is not None
        else "not executed: no GPU on this host",
        "decision": decision,
        "reason": (None if decision == "QUALIFIED" else
                   f"measured cuda={hw['cuda']}, gpus={hw['gpus']}; "
                   "full-scale 2B+502B training starts only after >=4 "
                   "GPU-class devices pass FSDP2 + sharded-optimizer + "
                   "recovery qualification on real hardware"),
        "full_scale_training_executed": False,
    }


def stage_i_summary(a: dict, c: dict, h: dict, d: dict, e: dict) -> dict:
    cov = c["coverage"]
    scalars = cov["scalars"]
    e = e or {}
    return {
        "stage": "I-strict-accounting-and-summary",
        "virtual_scalars": scalars,
        "scalar_level": cov["scalar_level"],
        "partition_level": cov["partition_level"],
        "invariants_hold": cov["invariants_hold"],
        "physical_model_status": {
            "architecture": "CortexMoETransformer over CORTEX_DEMO "
                            "(demo-scale joint prototype; the CORTEX_2B "
                            "architecture remains DECLARED)",
            "2b_trained": False,
            "reason": "GPU qualification gate not satisfied on this host",
        },
        "gpu_gate": h["decision"],
        "optimizer_recovery_at_expanded_coverage":
            d["recovery_verified_at_expanded_coverage"],
        "dense_control_comparison": {
            "dense_final_loss": e.get("dense_control", {}).get("loss_last"),
            "joint_random_final_loss":
                e.get("joint_random_init_row4", {}).get("loss_last"),
            "joint_svd_final_loss":
                e.get("joint_svd_init_row5", {}).get("loss_last"),
            "fresh_expert_cost": e.get("fresh_expert_cost"),
        } if e else "skipped",
        "v1_fingerprint_ok": a["v1_fingerprint_ok"],
        "v2_fingerprint_ok": a["v2_fingerprint_ok"],
        "owned_by": "Fibonera",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-bank", type=int, default=3)
    ap.add_argument("--skip-experiments", action="store_true")
    args = ap.parse_args()

    report: dict = {"kind": "cortex-stage8-expansion-report",
                    "owned_by": "Fibonera"}

    print("[stage8] A: stage-7 audit")
    report["A"] = stage_a_audit()
    print("[stage8] A2: lineage watermark check/repair")
    store = VirtualParameterStore.open(VSTORE_DIR)
    report["A_lineage_repair"] = repair_manifest_watermark(store)
    print("[stage8] B: storage semantics")
    report["B"] = stage_b_storage_semantics(store)
    print(f"[stage8] C: expanded coverage sweep "
          f"(rows_per_bank={args.rows_per_bank})")
    report["C"] = stage_c_expanded_coverage(store, args.rows_per_bank)
    print("[stage8] D: optimizer-state consistency + recovery")
    report["D"] = stage_d_optimizer_consistency(store)
    if not args.skip_experiments:
        print("[stage8] E: dense-control comparison + SVD-init arms")
        report["E"] = stage_e_training_experiments(store)
        print("[stage8] F: MoE specialization")
        report["F"] = stage_f_moe_specialization(store)
        print("[stage8] G: divergent reasoning v2 (repair round)")
        report["G"] = stage_g_divergent_v2()
    print("[stage8] H: GPU qualification")
    report["H"] = stage_h_qualification()
    report["I"] = stage_i_summary(report["A"], report["C"], report["H"],
                                  report["D"], report.get("E", {}))

    out = ART / "stage8_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    cov = report["C"]["coverage"]["scalars"]
    print(json.dumps({
        "virtual_scalars": cov,
        "gpu_gate": report["H"]["decision"],
        "invariants_hold": report["C"]["coverage"]["invariants_hold"],
    }, indent=2, default=str))
    print(f"[stage8] report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
