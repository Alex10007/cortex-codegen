"""Stage 7 orchestrator: unified 502B trainable-parameter architecture.

Executes engineering stages A-I for real on this host. Full-scale 2B/502B
training is gated on qualified GPU infrastructure (measured, not assumed);
the hardware qualification result is part of the report. Strict accounting
separates allocated / initialized / activated / gradient-receiving /
optimizer-updated / demonstrably-trained virtual parameters.

Usage:
    python scripts/stage7_run.py [--sweep-limit N]

--sweep-limit caps the partition-coverage sweep to the first N groups of 8
banks (engineering-time control for local validation); the default covers
all 5,000 partitions.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import cortex.virtual.store as vstore  # noqa: E402
from cortex.evaluation.benchmarks import check_integrity as check_v1  # noqa: E402
from cortex.model.configs import CORTEX_2B, CORTEX_DEMO  # noqa: E402
from cortex.model.moe import CortexMoETransformer  # noqa: E402
from cortex.tokenizer.cortex_tokenizer import load_tokenizer  # noqa: E402
from cortex.training.sft import evaluate_heldout, load_split  # noqa: E402
from cortex.virtual.benchmarks import run_all as run_infra_benchmarks  # noqa: E402
from cortex.virtual.coordinator import (  # noqa: E402
    JointTrainer, coverage_plan, restart_from_checkpoint)
from cortex.virtual.store import (  # noqa: E402
    BANK_COLS, N_BANKS, ROWS_PER_BANK, TOTAL_VIRTUAL, VirtualParameterStore,
    exact_virtual_count)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
VSTORE_DIR = ART / "virtual-store-502b"
JOINT_DIR = ART / "runs" / "joint-502b-demo"
WARM_CKPT = ART / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt"
DATASET = ART / "sft" / "dataset_v3.jsonl"
TOKENIZER = ART / "tokenizer" / "tokenizer.json"

VIRT_RANK_DEMO = 16


V2_FINGERPRINT = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"


def stage_a_audit() -> dict:
    from cortex.evaluation.benchmarks import fingerprint_tasks

    dense_2b = CORTEX_2B.theoretical_param_count()
    dense_demo = CORTEX_DEMO.theoretical_param_count()
    moe_demo = CortexMoETransformer(CORTEX_DEMO).actual_param_count()
    return {
        "stage": "A-architecture-audit-and-specification",
        "dense_2b_param_count": dense_2b,
        "dense_demo_param_count": dense_demo,
        "moe_demo_param_count": moe_demo,
        "dense_demo_is_control": True,
        "objective_doc": str((ROOT / "docs" / "OBJECTIVE_502B.md").relative_to(ROOT)),
        "v1_fingerprint": check_v1(),
        "v1_fingerprint_value": fingerprint_tasks("v1"),
        "v2_suite_fingerprint": fingerprint_tasks("v2"),
        "v2_suite_fingerprint_ok": fingerprint_tasks("v2") == V2_FINGERPRINT,
        "hardware": {
            "cuda": torch.cuda.is_available(),
            "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": platform.platform(),
        },
    }


def stage_b_allocate() -> dict:
    t0 = time.time()
    store = VirtualParameterStore(VSTORE_DIR)
    manifest = store.allocate()
    count = exact_virtual_count()
    assert count == TOTAL_VIRTUAL == 500_000_000_000
    store.close()
    return {
        "stage": "B-virtual-tensor-storage",
        "allocated_parameters": count,
        "exact_500b": count == 500_000_000_000,
        "partitions_total": N_BANKS,
        "partition_shape": [vstore.BANK_ROWS, BANK_COLS],
        "dtype": manifest["dtype"],
        "allocation_seconds": round(time.time() - t0, 3),
        "logical_bytes": ROWS_PER_BANK * 4 * N_BANKS,
        "owned_by": "Fibonera",
    }


def stage_c_optimizer(store: VirtualParameterStore) -> dict:
    from cortex.model.virtual_slot import VirtualAdamW

    dim, rank = CORTEX_DEMO.dim, VIRT_RANK_DEMO
    st = store.slot_state(0, 0, dim, rank, seed=11)
    before = (st.A.detach().clone(), st.B.detach().clone())
    opt = VirtualAdamW(lr=1e-3)
    x = torch.randn(2, dim)
    steps = []
    for i in range(5):
        ((x @ st.A) @ st.B).sum().backward()
        rec = opt.step(st)
        st.A.grad = None
        st.B.grad = None
        steps.append(rec)
    after = (st.A.detach(), st.B.detach())
    delta = float((after[0] - before[0]).abs().max())
    return {
        "stage": "C-virtual-optimizer-engine",
        "optimizer": "VirtualAdamW (standard AdamW, bias-corrected, decoupled wd)",
        "updates": [s["step"] for s in steps],
        "max_abs_delta_after_5_steps": round(delta, 8),
        "parameters_changed": delta > 0,
        "grad_norm_final": steps[-1]["grad_norm"],
    }


def stage_d_joint_training(store: VirtualParameterStore, sweep_limit: int | None) -> dict:
    tok = load_tokenizer(str(TOKENIZER))
    train_recs = load_split(DATASET, "train")
    val_recs = load_split(DATASET, "heldout")
    trainer = JointTrainer(
        CORTEX_DEMO, tok, store, JOINT_DIR, init_checkpoint=str(WARM_CKPT),
        virtual_rank=VIRT_RANK_DEMO)

    warm_steps = 30
    warm_losses = []
    group0 = [(b, b % vstore.BANK_ROWS) for b in range(trainer.model.n_virtual)]
    trainer.attach_partition_group(group0)
    from cortex.training.sft import make_batch as _mb

    valid_bases = [b for b in range(0, len(train_recs) - 3, 4)
                   if _mb(train_recs[b:b + 4], tok, 256)[0] is not None]
    assert valid_bases, "no fitting training batches"

    def batch(i: int) -> list[dict]:
        b = valid_bases[i % len(valid_bases)]
        return train_recs[b:b + 4]

    for i in range(warm_steps):
        rec = trainer.train_step(batch(i), seq_len=256)
        if rec:
            warm_losses.append(rec["loss"])
    heldout_before = evaluate_heldout(trainer.model, val_recs, tok, 256)

    groups, uncovered = coverage_plan(
        store, trainer.model.n_virtual,
        step_budget=sweep_limit if sweep_limit is not None else 10**9)
    sweep_losses = []
    for gi, g in enumerate(groups):
        trainer.attach_partition_group([(b, r) for b, r in g])
        rec = trainer.train_step(batch(gi), seq_len=256)
        if rec:
            sweep_losses.append(rec["loss"])
    heldout_after = evaluate_heldout(trainer.model, val_recs, tok, 256)

    ref_recs = train_recs[4:8]
    lineage = trainer.checkpoint(
        JOINT_DIR / "checkpoints", "tx-stage7-joint",
        extra={
            "ref_batch_loss": trainer.loss_on_batch(ref_recs, seq_len=256),
            "ref_records": "dataset_v3/train[4:8]",
            "warm_steps": warm_steps,
            "sweep_groups": len(groups),
        })
    acc = trainer.accounting()
    integrity = store.verify_integrity()
    detail = {
        "stage": "D-joint-training-coordinator",
        "warm_checkpoint": str(WARM_CKPT.relative_to(ROOT)),
        "steps_executed": trainer.step_count,
        "warm_phase": {"steps": len(warm_losses),
                       "loss_first": round(warm_losses[0], 4),
                       "loss_last": round(warm_losses[-1], 4)},
        "coverage_sweep": {"groups_executed": len(groups),
                           "partitions_targeted": len(groups) * trainer.model.n_virtual,
                           "uncovered_partitions": uncovered,
                           "loss_first": round(sweep_losses[0], 4),
                           "loss_last": round(sweep_losses[-1], 4)},
        "heldout_loss_after_warm_phase": heldout_before,
        "heldout_loss_after_sweep": heldout_after,
        "heldout_note": ("post-sweep heldout is measured with the LAST "
                         "attached group, which took only 1 step; fresh "
                         "groups start as noise until trained - not a "
                         "regression of the trained slots"),
        "lineage": {k: lineage[k] for k in (
            "step", "transaction_id", "physical_sha256", "virtual_integrity_ok")},
        "store_integrity": integrity,
        "accounting": acc,
        "joint_training_executed": True,
    }
    return detail, trainer


def stage_e_moe_diagnostics(trainer: JointTrainer) -> dict:
    diags = [layer.routing_diagnostics() for layer in trainer.model.moe_layers]
    last = [l.last_counts for l in trainer.model.moe_layers]
    return {
        "stage": "E-moe-diagnostics",
        "moe_layers": len(trainer.model.moe_layers),
        "last_routing": [d for d in diags[-1:]],
        "all_layers_utilization": [round(d["utilization_fraction"], 4) for d in diags],
        "any_layer_collapsed": any(d["collapsed"] for d in diags),
        "aux_terms_last": [round(float(l.last_aux.detach()), 6) for l in trainer.model.moe_layers],
        "z_terms_last": [round(float(l.last_z.detach()), 6) for l in trainer.model.moe_layers],
        "physical_experts": [int(c) for c in (
            last[0][:len(last[0]) - trainer.model.n_virtual].tolist())],
    }


def stage_f_divergent_eval(art: Path = ART) -> dict:
    import os

    from cortex.core import CortexAgent
    from cortex.evaluation.divergent import divergent_eval

    agent = CortexAgent(
        state_dir=str(art / "cortex-state"),
        tokenizer_path=str(TOKENIZER),
        checkpoint_path=str(WARM_CKPT))
    report = divergent_eval(agent, n_hypotheses=3)
    agg = dict(report["aggregate"])
    agg["stage"] = "F-divergent-reasoning"
    return agg


def stage_g_infrastructure(store: VirtualParameterStore) -> dict:
    with tempfile.TemporaryDirectory(prefix="cor7-infra-") as td:
        bench = run_infra_benchmarks(Path(td))
    # joint checkpoint recovery on the real joint run; the reference batch
    # must be the exact batch the lineage recorded its ref loss on
    tok = load_tokenizer(str(TOKENIZER))
    train_recs = load_split(DATASET, "train")[4:8]
    t2, rep = restart_from_checkpoint(
        JOINT_DIR / "checkpoints", tok, store, CORTEX_DEMO,
        train_recs, 256)
    return {
        "stage": "G-distributed-infrastructure-and-recovery",
        "benchmarks": bench,
        "joint_restart": {k: rep[k] for k in (
            "resumed_step", "physical_digest_ok", "lineage_steps_match",
            "ref_batch_loss_after_restart", "ref_batch_loss_before_checkpoint",
            "parity_ok")},
        "recovery_verified": bool(
            rep["parity_ok"] and rep["virtual_integrity"]["integrity_ok"]),
    }


def stage_h_controlled_experiment(store: VirtualParameterStore) -> dict:
    """Joint (MoE + virtual slots) vs identical-step dense control."""
    tok = load_tokenizer(str(TOKENIZER))
    train_recs = load_split(DATASET, "train")[:32]
    val_recs = load_split(DATASET, "heldout")[:16]

    from cortex.model.transformer import CortexTransformer
    from cortex.training.sft import make_batch

    torch.manual_seed(7)
    dense = CortexTransformer(CORTEX_DEMO)
    ck = torch.load(WARM_CKPT, map_location="cpu", weights_only=True)
    dense.load_state_dict(ck["model_state"])
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=3e-4,
                                  betas=(0.9, 0.98), weight_decay=0.01)
    dense_losses = []
    from cortex.training.sft import make_batch as _mb2

    valid_bases2 = [b for b in range(0, len(train_recs) - 3, 4)
                    if _mb2(train_recs[b:b + 4], tok, 256)[0] is not None]
    dense.train()
    for i in range(30):
        base = valid_bases2[i % len(valid_bases2)]
        x, y = make_batch(train_recs[base:base + 4], tok, 256)
        if x is None:
            continue
        _, loss = dense(x, y, pad_mask=x == tok.pad_id)
        dense_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense.parameters(), 1.0)
        dense_opt.step()
        dense_losses.append(round(float(loss.detach()), 4))
    dense_heldout = evaluate_heldout(dense, val_recs, tok, 256)

    trainer = JointTrainer(
        CORTEX_DEMO, tok, store, ART / "runs" / "joint-502b-control",
        init_checkpoint=str(WARM_CKPT), virtual_rank=VIRT_RANK_DEMO)
    group0 = [(b, b % vstore.BANK_ROWS) for b in range(trainer.model.n_virtual)]
    trainer.attach_partition_group(group0)
    joint_losses = []
    for i in range(30):
        base = valid_bases2[i % len(valid_bases2)]
        rec = trainer.train_step(train_recs[base:base + 4], seq_len=256, log=False)
        if rec:
            joint_losses.append(rec["loss"])
    joint_heldout = evaluate_heldout(trainer.model, val_recs, tok, 256)
    return {
        "stage": "H-controlled-training-experiment",
        "steps": len(dense_losses),
        "dense_final_loss": dense_losses[-1],
        "joint_final_loss": round(joint_losses[-1], 4) if joint_losses else None,
        "dense_heldout": dense_heldout,
        "joint_heldout": joint_heldout,
        "note": ("identical batches, warm start and lr; joint arm additionally "
                 "learns 8 fresh virtual slots in 30 steps, while the dense "
                 "control starts from an already-SFT'd dense head; new-expert "
                 "cost visible in early joint loss"),
        "dense_control_untouched_class": "cortex.model.transformer.CortexTransformer",
    }


def stage_i_gate(stage_a: dict, stage_d: dict) -> dict:
    hw = stage_a["hardware"]
    gpu_ok = hw["cuda"] and hw["gpus"] >= 4
    return {
        "stage": "I-full-scale-gate",
        "decision": "PROCEED" if gpu_ok else "REFUSED-HARDWARE-GATE",
        "reason": (None if gpu_ok else
                   f"measured cuda={hw['cuda']}, gpus={hw['gpus']}; "
                   "2B physical + 500B virtual training requires >=4x80GB-class "
                   "GPUs with the validated FSDP/sharding path"),
        "what_would_qualify":
            "8x40GB or 4x80GB GPUs, measured FSDP2 validation, memory "
            "estimation margin, and this host's storage path",
        "full_scale_training_executed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-limit", type=int, default=None)
    args = ap.parse_args()

    report: dict = {"kind": "cortex-stage7-502b-report",
                    "owned_by": "Fibonera"}
    print("[stage7] A: audit")
    report["A"] = stage_a_audit()
    print("[stage7] B: allocate 500B virtual store")
    report["B"] = stage_b_allocate()
    store = VirtualParameterStore.open(VSTORE_DIR)
    print("[stage7] C: virtual optimizer")
    report["C"] = stage_c_optimizer(store)
    print("[stage7] D: joint training + coverage sweep")
    report["D"], trainer_final = stage_d_joint_training(store, args.sweep_limit)
    print("[stage7] E: MoE diagnostics")
    report["E"] = stage_e_moe_diagnostics(trainer_final)
    print("[stage7] F: divergent reasoning eval")
    report["F"] = stage_f_divergent_eval()
    print("[stage7] G: infrastructure + recovery")
    report["G"] = stage_g_infrastructure(store)
    print("[stage7] H: controlled experiment")
    report["H"] = stage_h_controlled_experiment(store)
    report["I"] = stage_i_gate(report["A"], report["D"])

    out = ART / "stage7_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    acc = report["D"]["accounting"]
    print(json.dumps({
        "virtual_accounting": {k: acc[k] for k in (
            "allocated", "initialized", "activated", "received_gradients",
            "optimizer_updated", "demonstrably_trained", "partitions_touched",
            "partitions_total")},
        "gate": report["I"]["decision"],
    }, indent=2))
    print(f"[stage7] report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
