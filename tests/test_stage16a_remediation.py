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

def test_legacy_committed_slot_restores_from_optimizer_shape_witness(small_store):
    from cortex.model.virtual_slot import VirtualAdamW

    slot = small_store.slot_state(3, 4, dim=32, rank=4)
    ((torch.randn(2, 32) @ slot.A) @ slot.B).sum().backward()
    VirtualAdamW().step(slot)
    expected_a = slot.A.detach().clone()
    expected_b = slot.B.detach().clone()
    small_store.commit(step=1, txid="legacy-optimizer-witness")
    manifest = json.loads(small_store.manifest_path.read_text())
    manifest.pop("initialized_slot_shapes")
    small_store.manifest_path.write_text(json.dumps(manifest))

    restored_store = vstore.VirtualParameterStore.open(small_store.root)
    restored = restored_store.slot_state(3, 4, dim=32, rank=4)
    assert torch.equal(restored.A.detach(), expected_a)
    assert torch.equal(restored.B.detach(), expected_b)
    assert restored.step_count == 1


def test_orphan_optimizer_state_is_not_counted_as_trained(small_store):
    from cortex.model.virtual_slot import VirtualAdamW, slot_adam_states
    from cortex.virtual.coverage import measure

    slot = small_store.slot_state(4, 5, dim=32, rank=4)
    ((torch.randn(2, 32) @ slot.A) @ slot.B).sum().backward()
    VirtualAdamW().step(slot)
    npz_path = small_store.root / "adam-00004-000005.npz"
    import numpy as np
    np.savez(npz_path, **slot_adam_states(slot))
    small_store._slots.clear()
    small_store.dirty.clear()

    accounting = small_store.accounting()
    coverage = measure(small_store)
    assert accounting["orphan_optimizer_state_slots"] == 1
    assert accounting["optimizer_updated"] == 0
    assert accounting["invariants_hold"] is False
    assert coverage["persisted_optimizer_scan"]["orphan_optimizer_slots"] == 1
    assert coverage["scalars"]["optimizer_updated"] == 0
    with pytest.raises(RuntimeError, match="orphan optimizer state"):
        small_store.slot_state(4, 5, dim=32, rank=4)

def test_moe_physical_parameter_ledger_separates_dense_reference():
    from cortex.model.configs import CORTEX_DEMO
    from cortex.model.moe import CortexMoETransformer

    model = CortexMoETransformer(CORTEX_DEMO)
    ledger = model.moe_param_count()
    assert ledger["dense_reference_parameters"] == (
        CORTEX_DEMO.theoretical_param_count())
    assert ledger["physical_total_parameters"] == model.actual_param_count()
    assert ledger["physical_total_parameters"] > (
        ledger["dense_reference_parameters"])
    assert ledger["total_with_virtual"] == (
        ledger["physical_total_parameters"] + ledger["virtual_slot_parameters"])


def test_persisted_slot_shape_cannot_be_reinterpreted(small_store):
    store = small_store
    store.slot_state(0, 6, dim=64, rank=8)
    store.commit(step=0, txid="persisted-shape")
    store._slots.clear()
    with pytest.raises(ValueError, match="persisted slot"):
        store.slot_state(0, 6, dim=64, rank=4)



def test_generation_context_uses_only_verified_project_memory(tmp_path):
    from cortex.core import CortexAgent
    from cortex.interfaces import Provenance

    agent = CortexAgent(state_dir=str(tmp_path / "agent"))
    project = "sample-project"
    agent.memory.put_fact(project, "language", "Python", verified=True)
    agent.memory.put_fact(project, "untrusted-note", "do not trust", verified=False)
    agent.memory.record(
        project, "pipeline", {"status": "verified-change"},
        verified=True)
    agent.memory.record(
        project, "pipeline", {"status": "unverified-change"},
        verified=False)
    provenance = Provenance(
        source_uri="file:///verified.py", license="project-local",
        version="1", content_sha256="abc", collected_at="now",
        collector="test")
    agent.learning.ingest_verified_outcome(
        "def add(a, b):\\n    return a + b\\n",
        execution_ok=True, tests_passed=True, provenance=provenance)
    context = agent.retrieval_context(project, "add function")
    assert "language: Python" in context
    assert "untrusted-note" not in context
    assert "verified-change" in context
    assert "unverified-change" not in context
    assert "def add" in context


def test_moe_masks_unbound_virtual_routes_until_slots_are_attached():
    from cortex.model.moe import MoEFFN

    moe = MoEFFN(dim=4, n_physical=4, n_virtual=2, top_k=2,
                 expert_ffn=4)
    moe.experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    # Force the raw router to prefer virtual experts. They are not attached,
    # so routing must mask them and select only the available physical experts.
    with torch.no_grad():
        moe.router.weight.fill_(-10.0)
        moe.router.weight[4:].fill_(10.0)
    x = torch.ones(2, 3, 4)
    y = moe(x)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)
    assert int(moe.last_counts[4:].sum()) == 0
    diag = moe.routing_diagnostics()
    assert diag["active_experts"] == 4
    assert diag["utilization_fraction"] <= 1.0


def test_moe_refuses_when_active_experts_cannot_fill_topk():
    from cortex.model.moe import MoEFFN

    moe = MoEFFN(dim=4, n_physical=1, n_virtual=2, top_k=2,
                 expert_ffn=4)
    with pytest.raises(RuntimeError, match="active experts"):
        moe(torch.ones(1, 2, 4))



def test_agent_generation_uses_inference_mode(tmp_path):
    from types import SimpleNamespace
    from cortex.core import CortexAgent

    class Tokenizer:
        EOT = "<eot>"
        eot_id = 2

        def encode(self, text):
            return [1, 3]

        def decode(self, tokens):
            return "generated"

    class Model:
        cfg = SimpleNamespace(max_seq_len=32)

        def generate(self, tokens, **kwargs):
            assert torch.is_inference_mode_enabled()
            return tokens

    agent = CortexAgent(state_dir=str(tmp_path / "agent"))
    agent._tokenizer = Tokenizer()
    agent._model = Model()
    assert agent.generate("hello") == "generated"



def test_project_facts_are_excluded_until_explicitly_verified(tmp_path):
    from cortex.memory.engine import MemoryEngine

    memory = MemoryEngine(str(tmp_path / "memory.db"))
    memory.put_fact("project", "draft", "unverified value")
    assert memory.get_fact("project", "draft") is None
    assert memory.facts("project") == {}
    memory.put_fact("project", "accepted", "verified value", verified=True)
    assert memory.get_fact("project", "accepted") == "verified value"
    assert memory.facts("project") == {"accepted": "verified value"}



def test_moe_topk_load_balance_is_normalized():
    from cortex.model.moe import MoEFFN

    moe = MoEFFN(dim=4, n_physical=4, n_virtual=0, top_k=2,
                 expert_ffn=4)
    moe.experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    with torch.no_grad():
        moe.router.weight.zero_()
    moe(torch.ones(2, 3, 4))
    assert float(moe.last_aux) == pytest.approx(1.0, abs=1e-6)



def test_generation_metadata_reports_actual_context_usage(tmp_path):
    from types import SimpleNamespace
    from cortex.core import CortexAgent

    class Tokenizer:
        EOT = "<eot>"
        eot_id = 2

        def encode(self, text):
            return [1, 3]

        def decode(self, tokens):
            return "generated"

    class Model:
        cfg = SimpleNamespace(max_seq_len=32)

        def generate(self, tokens, **kwargs):
            assert torch.is_inference_mode_enabled()
            return tokens

    agent = CortexAgent(state_dir=str(tmp_path / "agent"))
    agent._tokenizer = Tokenizer()
    agent._model = Model()
    agent.retrieval_context = lambda project, query: ""
    without = agent.generate_with_metadata("hello", project="empty-project")
    assert without["project_context_used"] is False
    agent.retrieval_context = lambda project, query: "verified project fact"
    with_context = agent.generate_with_metadata("hello", project="known-project")
    assert with_context["project_context_used"] is True



def test_multi_slot_commit_publishes_one_manifest_transaction(
        small_store, monkeypatch):
    before = json.loads(small_store.manifest_path.read_text())
    slot_a = small_store.slot_state(0, 0, dim=32, rank=4, seed=7)
    slot_b = small_store.slot_state(0, 1, dim=32, rank=4, seed=8)
    expected_a = (slot_a.A.detach().clone(), slot_a.B.detach().clone())
    expected_b = (slot_b.A.detach().clone(), slot_b.B.detach().clone())

    def fail_manifest_write(*args, **kwargs):
        raise OSError("simulated manifest publish failure")

    monkeypatch.setattr(vstore, "_atomic_json", fail_manifest_write)
    with pytest.raises(OSError, match="manifest publish"):
        small_store.commit(step=1, txid="atomic-test")

    after = json.loads(small_store.manifest_path.read_text())
    assert after == before
    assert small_store.dirty == {(0, 0), (0, 1)}

    # Raw bank bytes written before the manifest boundary are not accepted as
    # committed state; a fresh process deterministically reinitializes them.
    reopened = vstore.VirtualParameterStore.open(small_store.root)
    restored_a = reopened.slot_state(0, 0, dim=32, rank=4, seed=7)
    restored_b = reopened.slot_state(0, 1, dim=32, rank=4, seed=8)
    assert torch.equal(restored_a.A.detach(), expected_a[0])
    assert torch.equal(restored_a.B.detach(), expected_a[1])
    assert torch.equal(restored_b.A.detach(), expected_b[0])
    assert torch.equal(restored_b.B.detach(), expected_b[1])



def test_2b_moe_parameter_estimate_is_separate_from_dense_2b():
    from cortex.model.configs import CORTEX_2B
    from cortex.model.moe import estimate_moe_physical_parameters

    report = estimate_moe_physical_parameters(CORTEX_2B)
    assert report["dense_reference_parameters"] == 2_056_012_800
    assert report["physical_total_parameters"] == 5_832_360_960
    assert report["physical_total_parameters"] > (
        report["dense_reference_parameters"])
    assert report["physical_total_parameters"] == (
        report["backbone_parameters"]
        + report["physical_expert_parameters"]
        + report["router_parameters"])
