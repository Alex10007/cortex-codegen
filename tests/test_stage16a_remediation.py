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
          "code": "def add(a, b):\\n    return a + b\\n"}],
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
