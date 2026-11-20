"""Regression tests for Stage 16A architecture-correctness remediation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import cortex.virtual.store as vstore

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "artifacts" / "tokenizer" / "tokenizer.json"


@pytest.fixture()
def small_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL", 8 * vstore.ROWS_PER_BANK)
    store = vstore.VirtualParameterStore(tmp_path / "virtual")
    store.allocate()
    store.manifest = json.loads(store.manifest_path.read_text())
    return store


def test_moe_uses_renormalized_topk_gate():
    from cortex.model.moe import MoEFFN

    moe = MoEFFN(dim=4, n_physical=4, n_virtual=0, top_k=2,
                 expert_ffn=4)
    moe.experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    with torch.no_grad():
        moe.router.weight.zero_()
    x = torch.randn(2, 3, 4)
    y = moe(x)
    # Uniform router probabilities select two of four experts. Renormalized
    # selected gates sum to one, so identical expert functions reproduce x.
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_virtual_rank_defaults_to_bank_row_capacity():
    from cortex.model.configs import CORTEX_2B, CORTEX_DEMO
    from cortex.virtual.coordinator import JointTrainer
    from cortex.virtual.store import BANK_COLS

    assert JointTrainer.resolve_virtual_rank(CORTEX_DEMO.dim) == 64
    expected = BANK_COLS // (2 * CORTEX_2B.dim)
    assert expected == 9
    assert JointTrainer.resolve_virtual_rank(CORTEX_2B.dim) == expected
    with pytest.raises(ValueError, match="maximum rank"):
        JointTrainer.resolve_virtual_rank(CORTEX_2B.dim, requested=64)


def test_virtual_slot_rejects_shape_alias(small_store):
    store = small_store
    original = store.slot_state(0, 0, dim=64, rank=8)
    assert original.A.shape == (64, 8)
    with pytest.raises(ValueError, match="cached slot"):
        store.slot_state(0, 0, dim=64, rank=4)


def test_virtual_parameter_count_includes_both_factors_and_deduplicates(
        small_store):
    from cortex.model.configs import CORTEX_DEMO
    from cortex.model.moe import CortexMoETransformer

    model = CortexMoETransformer(CORTEX_DEMO)
    slots = {
        route: small_store.slot_state(0, route, CORTEX_DEMO.dim, rank=8)
        for route in range(model.n_virtual)
    }
    model.attach_virtual_slots(slots)
    expected = sum(s.A.numel() + s.B.numel() for s in slots.values())
    assert model.virtual_param_count() == expected


def test_joint_trainer_updates_shared_virtual_slot_once_and_clears_gradients(
        small_store, tmp_path):
    from cortex.model.configs import CORTEX_DEMO
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.virtual.coordinator import JointTrainer, coverage_plan

    tok = load_tokenizer(str(TOKENIZER))
    trainer = JointTrainer(CORTEX_DEMO, tok, small_store, tmp_path / "trainer",
                           virtual_rank=8, seed=17)
    groups, _ = coverage_plan(small_store, trainer.model.n_virtual, 1)
    trainer.attach_partition_group(groups[0])
    rec = trainer.train_step(
        [{"instruction": "add two values",
          "code": "def add(a, b):\n    return a + b\n"}],
        seq_len=256)
    assert rec is not None
    updated = {tuple(x) for x in rec["virtual_slots_updated"]}
    assert len(updated) == len(rec["virtual_slots_updated"])
    for key, slot in small_store._slots.items():
        assert slot.step_count == (1 if key in updated else 0)
        assert slot.A.grad is None
        assert slot.B.grad is None

def test_2b_memory_estimate_includes_quadratic_attention():
    from cortex.model.configs import CORTEX_2B
    from cortex.training.scaling import estimate_memory_bytes

    seq = CORTEX_2B.max_seq_len
    rep = estimate_memory_bytes(CORTEX_2B, batch_size=1, seq_len=seq)
    expected = (
        CORTEX_2B.n_layers * CORTEX_2B.n_heads * seq * seq * 4 * 2)
    assert rep["attention_matrix_bytes_est"] == expected
    assert rep["activation_bytes_est"] >= expected
    assert rep["total_bytes_est"] == (
        rep["param_bytes"] + rep["grad_bytes"]
        + rep["optimizer_bytes"] + rep["activation_bytes_est"])

def test_virtual_accounting_uses_slot_shapes_not_full_bank_rows(small_store):
    from cortex.virtual.coverage import measure, persisted_optimizer_scan

    slot = small_store.slot_state(2, 3, dim=64, rank=8)
    small_store.commit(step=0, txid="shape-accounting")
    expected = slot.A.numel() + slot.B.numel()
    accounting = small_store.accounting()
    coverage = measure(small_store)
    scan = persisted_optimizer_scan(small_store)
    assert accounting["initialized"] == expected
    assert accounting["initialized_exact"] is True
    assert accounting["initialized_unknown_rows"] == 0
    assert coverage["scalars"]["initialized"] == expected
    assert coverage["scalars"]["initialized_exact"] is True
    assert scan["n_allocated"] == vstore.exact_virtual_count()


def test_legacy_slot_without_shape_or_optimizer_is_reported_unknown(small_store):
    slot = small_store.slot_state(1, 2, dim=32, rank=4)
    small_store.commit(step=0, txid="legacy-shape")
    manifest = json.loads(small_store.manifest_path.read_text())
    manifest.pop("initialized_slot_shapes")
    small_store.manifest_path.write_text(json.dumps(manifest))
    small_store.manifest = manifest
    accounting = small_store.accounting()
    assert accounting["initialized"] == 0
    assert accounting["initialized_unknown_rows"] == 1
    assert accounting["initialized_exact"] is False

def test_fsdp_planner_refuses_2b_until_attention_fits_cluster_matrix():
    from scripts.train_2b_fsdp import required_gpus

    # The current explicit attention implementation materializes quadratic
    # sequence matrices; even the largest validated 80-GB device plan must
    # refuse rather than claim the old under-estimated 2B memory requirement.
    with pytest.raises(RuntimeError, match="no cluster size"):
        required_gpus("cortex-2b", batch_size=8, gpu_memory_gb=80.0)



def test_cluster_plans_never_emit_zero_microbatch():
    from cortex.model.configs import CORTEX_2B
    from cortex.training.scaling import cluster_plans

    plans = cluster_plans(CORTEX_2B, batch_size=4)
    assert plans["plans"]
    assert all(p["micro_batch"] >= 1 for p in plans["plans"])
    assert all(4 % p["world_size"] == 0 for p in plans["plans"])



def test_agent_loads_joint_moe_checkpoint_with_virtual_lineage(
        small_store, tmp_path, monkeypatch):
    from cortex.core import CortexAgent
    from cortex.model.configs import CORTEX_DEMO
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.virtual.coordinator import JointTrainer, coverage_plan

    tokenizer_path = str(TOKENIZER)
    tok = load_tokenizer(tokenizer_path)
    trainer = JointTrainer(
        CORTEX_DEMO, tok, small_store, tmp_path / "trainer",
        virtual_rank=8, seed=29)
    groups, _ = coverage_plan(small_store, trainer.model.n_virtual, 1)
    trainer.attach_partition_group(groups[0])
    checkpoint_dir = tmp_path / "joint-checkpoint"
    trainer.checkpoint(checkpoint_dir, "stage16a-test")
    checkpoint_path = checkpoint_dir / "joint-step-000000.pt"

    agent = CortexAgent(
        state_dir=str(tmp_path / "agent-state"),
        tokenizer_path=tokenizer_path,
        checkpoint_path=str(checkpoint_path),
        virtual_store_path=str(small_store.root))
    model = agent.model
    from cortex.model.moe import CortexMoETransformer
    assert isinstance(model, CortexMoETransformer)
    assert model.virtual_param_count() > 0
    assert model.moe_param_count()["physical_total_parameters"] == (
        model.actual_param_count())
