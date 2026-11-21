"""Stage 7: unified 502B trainable-parameter architecture tests.

All tests exercise real components (no mocks): sparse tensor store paging,
real AdamW math verified against independent recomputation, real autograd
into virtual slot parameters, MoE routing diagnostics, joint training with
transactional checkpoint/recovery, and immutable benchmark fingerprints.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import cortex.virtual.store as vstore

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "artifacts" / "tokenizer" / "tokenizer.json"


@pytest.fixture()
def small_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL", 8 * vstore.ROWS_PER_BANK)
    from cortex.virtual.store import VirtualParameterStore

    store = VirtualParameterStore(tmp_path / "vstore")
    manifest = store.allocate()
    store.manifest = json.loads(store.manifest_path.read_text())
    return store


def test_virtual_total_is_exactly_500b():
    assert vstore.exact_virtual_count() == 500_000_000_000
    assert vstore.N_BANKS == 5_000
    assert vstore.BANK_ROWS == 2_000
    assert vstore.BANK_COLS == 50_000
    assert vstore.TOTAL_VIRTUAL == vstore.N_BANKS * vstore.BANK_ROWS * vstore.BANK_COLS


def test_full_scale_allocation_manifest(tmp_path):
    from cortex.virtual.store import (N_BANKS, ROWS_PER_BANK,
                                      VirtualParameterStore)
    store = VirtualParameterStore(tmp_path / "full")
    m = store.allocate()
    assert m["allocated_parameters"] == 500_000_000_000
    assert m["asserted_exact_total"] is True
    assert m["owned_by"] == "Fibonera"
    assert len(m["coverage_rows_per_bank"]) == N_BANKS
    disk = sum(f.stat().st_size for f in store.root.iterdir()
               if f.name.startswith("virtual-bank-"))
    assert disk == ROWS_PER_BANK * 4 * N_BANKS  # logical allocation exact
    import os
    blocks = sum(f.stat().st_blocks * 512 for f in store.root.iterdir()
                 if hasattr(f.stat(), "st_blocks"))
    assert blocks < ROWS_PER_BANK * 4  # sparse: far fewer physical blocks
    store2 = VirtualParameterStore.open(tmp_path / "full")
    assert store2.manifest["allocated_parameters"] == 500_000_000_000


def test_tensor_integrity_roundtrip(small_store):
    store = small_store
    st = store.slot_state(0, 1, dim=64, rank=8, seed=5)
    st2 = store.slot_state(0, 1, dim=64, rank=8, seed=999)  # cached, same slot
    assert st2 is st
    store.commit(step=1, txid="tx-1")
    committed = (st.A.detach().clone(), st.B.detach().clone())
    st.A.detach().mul_(2.0)  # out-of-band mutation then recommit
    store.dirty.add((0, 1))
    store.commit(step=2, txid="tx-2")
    assert store.verify_integrity()["integrity_ok"]
    # fresh read from disk reproduces committed values exactly
    store._pages.clear()
    store._slots.clear()
    st3 = store.slot_state(0, 1, dim=64, rank=8, seed=999)
    assert torch.equal(st3.A.detach(), st.A.detach())
    assert torch.equal(st3.B.detach(), committed[1])
    assert st3.A.shape == (64, 8) and st3.B.shape == (8, 64)


def test_corrupt_row_detected_by_digests(small_store):
    store = small_store
    store.slot_state(2, 2, dim=32, rank=4, seed=1)
    store.commit(step=1, txid="tx-1")
    page = store.page(2)
    page[2, 0] = 9.5
    page.flush()
    rep = store.verify_integrity()
    assert not rep["integrity_ok"]
    assert "2/2" in rep["bad_rows"]


def test_virtual_adamw_matches_independent_recomputation(small_store, tmp_path):
    from cortex.model.virtual_slot import SlotState, VirtualAdamW

    st = small_store.slot_state(3, 3, dim=16, rank=4, seed=2)
    p_before = st.A.detach().clone()
    x = torch.randn(4, 16)
    ((x @ st.A) @ st.B).pow(2).sum().backward()
    lr, b1, b2, eps, wd = 1e-2, 0.9, 0.98, 1e-8, 0.0
    opt = VirtualAdamW(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    g = st.A.grad.clone()
    opt.step(st)
    # independent AdamW recomputation: first step initializes m=(1-b1)g,
    # v=(1-b2)g^2, so bias correction cancels and update = -lr*g/(|g|+eps)
    expected = p_before - lr * g / (g.abs() + eps)
    assert torch.allclose(st.A.detach(), expected, atol=1e-6)
    assert st.step_count == 1


def test_slot_receives_real_autograd(small_store):
    st = small_store.slot_state(0, 0, dim=32, rank=4, seed=3)
    x = torch.randn(2, 32)
    y = st.forward(x, torch.tensor(1.0))
    y.sum().backward()
    assert st.A.grad is not None and st.B.grad is not None
    assert float(st.A.grad.abs().sum()) > 0


def test_moe_routing_and_virtual_gradients():
    from cortex.model.configs import CORTEX_DEMO
    from cortex.model.moe import CortexMoETransformer
    from cortex.virtual.store import VirtualParameterStore

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        vp = vstore.VirtualParameterStore(td)
        vp.manifest = vp.allocate()
        model = CortexMoETransformer(CORTEX_DEMO)
        slots = {r: vp.slot_state(0, r, CORTEX_DEMO.dim, 16, seed=r)
                 for r in range(model.n_virtual)}
        model.attach_virtual_slots(slots)
        x = torch.randint(0, CORTEX_DEMO.vocab_size, (2, 64))
        y = torch.randint(0, CORTEX_DEMO.vocab_size, (2, 64))
        _, loss = model(x, y, pad_mask=torch.zeros_like(x, dtype=torch.bool))
        loss.backward()
        total = 0
        for layer in model.moe_layers:
            d = layer.routing_diagnostics()
            assert sum(d["counts"]) == d["tokens_routed"]
            assert d["tokens_routed"] == 2 * 2 * 64  # top_k=2 selections/token
            assert 0 <= d["utilization_fraction"] <= 1.0
            assert not d["collapsed"]
            assert float(layer.last_aux) >= 0 and float(layer.last_z) >= 0
        # distinct slot receives gradient through at least one layer
        grad_any = sum(
            float(s.A.grad.abs().sum()) if s.A.grad is not None else 0
            for s in slots.values())
        assert grad_any > 0
        total = model.moe_param_count()
        assert total["total_actual"] == model.actual_param_count()


def test_joint_step_changes_both_systems(small_store, tmp_path):
    from cortex.model.configs import CORTEX_DEMO
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.virtual.coordinator import JointTrainer, coverage_plan

    tok = load_tokenizer(str(TOKENIZER))
    store = small_store
    trainer = JointTrainer(CORTEX_DEMO, tok, store, tmp_path / "j",
                           virtual_rank=16)
    groups, _ = coverage_plan(store, trainer.model.n_virtual, step_budget=1)
    trainer.attach_partition_group([(b, r) for b, r in groups[0]])
    slots_before = {k: (s.A.detach().clone(), s.B.detach().clone())
                    for k, s in store._slots.items()}
    phys_before = {n: p.detach().clone() for n, p in
                   trainer.model.named_parameters() if p.requires_grad}
    rec = trainer.train_step(
        [{"instruction": "add two numbers", "code": "c = a + b\nprint(c)"}],
        seq_len=256)
    assert rec is not None and rec["loss"] > 0
    # slots update only when the router routes tokens to them (Switch
    # semantics); a never-selected slot takes no step and is not counted
    assert 1 <= len(rec["virtual_slots_updated"]) <= trainer.model.n_virtual
    assert all(v > 0 for v in rec["virtual_grad_norms"].values())
    for k, s in store._slots.items():
        assert not torch.equal(s.A.detach(), slots_before[k][0])
    changed = 0
    for n, p in phys_before.items():
        cur = dict(trainer.model.named_parameters())[n].detach()
        changed += int(not torch.equal(cur, p))
    assert changed > 0
    part = trainer.participation[-1]
    assert part["virtual_grad_params"] == sum(
        store._slots[br].A.numel() + store._slots[br].B.numel()
        for br in part["virtual_slots_updated"])


def test_checkpoint_reload_parity_and_lineage_refusal(tmp_path, small_store):
    from cortex.model.configs import CORTEX_DEMO
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.virtual.coordinator import JointTrainer, coverage_plan, \
        restart_from_checkpoint

    tok = load_tokenizer(str(TOKENIZER))
    store = small_store
    trainer = JointTrainer(CORTEX_DEMO, tok, store, tmp_path / "j",
                           virtual_rank=16, seed=9)
    groups, _ = coverage_plan(store, trainer.model.n_virtual, step_budget=1)
    trainer.attach_partition_group([(b, r) for b, r in groups[0]])
    recs = [{"instruction": "double v", "code": "w = v * 2\nprint(w)"}]
    l1 = trainer.loss_on_batch(recs, seq_len=256)
    lin = trainer.checkpoint(tmp_path / "ckpt", "tx-1",
                             extra={"ref_batch_loss": l1})
    store2 = vstore.VirtualParameterStore(store.root)
    store2.manifest = json.loads(store2.manifest_path.read_text())
    t2, rep = restart_from_checkpoint(tmp_path / "ckpt", tok, store2,
                                      CORTEX_DEMO, recs, 256)
    assert rep["parity_ok"]
    assert abs(rep["ref_batch_loss_after_restart"] - l1) < 1e-6
    assert rep["resumed_step"] == trainer.step_count

    # stale/incorrect pair is refused: wrong virtual rank must not be accepted
    store3 = vstore.VirtualParameterStore(store.root)
    store3.manifest = json.loads(store3.manifest_path.read_text())
    bad = {k: v for k, v in store3.manifest.items()}
    bad["committed_step"] = trainer.step_count + 5
    (store.root / "tampered-manifest.json").write_text(json.dumps(bad))
    store3.manifest_path = store.root / "tampered-manifest.json"
    store3.manifest = bad
    with pytest.raises(RuntimeError):
        restart_from_checkpoint(tmp_path / "ckpt", tok, store3,
                                CORTEX_DEMO, recs, 256)


def test_strict_accounting_never_equates_allocated_with_trained(small_store):
    from cortex.model.virtual_slot import VirtualAdamW

    store = small_store
    st = store.slot_state(1, 1, dim=32, rank=4, seed=5)
    x = torch.randn(4, 32)
    ((x @ st.A) @ st.B).sum().backward()
    store.commit(1, "tx-a")
    acc = store.accounting()
    assert acc["allocated"] == 8 * vstore.ROWS_PER_BANK
    assert acc["allocated"] > acc["initialized"] > 0
    assert acc["activated"] == acc["received_gradients"] == 256
    assert acc["optimizer_updated"] == acc["demonstrably_trained"] == 0
    assert acc["demonstrably_trained"] < acc["allocated"]
    # after a real optimizer step the slot moves into the trained buckets
    VirtualAdamW().step(st)
    acc2 = store.accounting()
    assert acc2["optimizer_updated"] == acc2["demonstrably_trained"] == 256
    assert 0 < acc2["demonstrably_trained"] < acc2["allocated"]


def test_moe_dense_control_untouched():
    from cortex.model.configs import CORTEX_DEMO
    from cortex.model.transformer import CortexTransformer

    base = CortexTransformer(CORTEX_DEMO)
    assert base.__class__ is CortexTransformer
    # dense class still trains a pure completion-masked objective
    x = torch.randint(0, CORTEX_DEMO.vocab_size, (1, 8))
    y = torch.randint(-100, CORTEX_DEMO.vocab_size, (1, 8))
    _, loss = base(x, y)
    assert loss is not None


def test_shard_probe_two_processes(tmp_path):
    from cortex.virtual.benchmarks import run_shard_probe

    rep = run_shard_probe(tmp_path, banks=16)
    assert rep["exit_codes"] == [0, 0]
    assert rep["report"]["sharding_ok"]
    assert rep["report"]["ownership_overlap"] == []
    assert rep["report"]["banks_covered"] == 16


def test_benchmark_fingerprints_immutable():
    from cortex.evaluation.benchmarks import check_integrity, fingerprint_tasks

    v1 = check_integrity()
    assert v1["status"] == "unmodified"
    assert v1["fingerprint"] == (
        "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177")
    assert fingerprint_tasks("v2") == (
        "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3")
