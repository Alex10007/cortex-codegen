"""Stage 8: virtual-parameter training expansion tests.

Real components only (no mocks of the system under test): AdamW optimizer
state round-trip through disk, commit-before-evict safety, deterministic
SVD slot initialization, scalar-level coverage invariants, update
distribution accounting, divergent-repair mechanics (grading stubbed at the
sandbox boundary only) and immutable fingerprints.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import cortex.virtual.store as vstore
import cortex.virtual.coverage as coverage

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "artifacts" / "tokenizer" / "tokenizer.json"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"


@pytest.fixture()
def small_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL", 8 * vstore.ROWS_PER_BANK)
    from cortex.virtual.store import VirtualParameterStore

    store = VirtualParameterStore(tmp_path / "vstore")
    store.allocate()
    store.manifest = json.loads(store.manifest_path.read_text())
    return store


def test_svd_slot_init_deterministic_and_scaled():
    from cortex.model.virtual_slot import svd_slot_init

    torch.manual_seed(3)
    M = torch.randn(64, 64)
    U, S, Vh = torch.linalg.svd(M)
    scale = 8.0 * (0.01 / 8.0) ** (torch.arange(64) / 63.0)
    M = (U * scale) @ Vh  # concentrated spectrum
    flat1 = svd_slot_init(M, rank=8)
    flat2 = svd_slot_init(M, rank=8)
    assert flat1.dtype == np.float32
    assert np.array_equal(flat1, flat2)
    A = torch.from_numpy(flat1[:64 * 8]).view(64, 8)
    B = torch.from_numpy(flat1[64 * 8:]).view(8, 64)
    x = torch.randn(256, 64)
    y = (x @ A) @ B
    assert abs(float(y.std()) - 0.02) < 0.01
    # A @ B is a scaled rank-8 approximation: SVD residual strictly smaller
    # than the input's own energy on the orthogonal complement
    sv = torch.linalg.svdvals(torch.from_numpy(
        (A.numpy() @ B.numpy())))
    assert sv[8:].abs().max() < 1e-4
    assert sv[0] > sv[7] > 0


def test_svd_slot_init_reduces_fresh_expert_cost(small_store):
    from cortex.model.virtual_slot import svd_slot_init

    store = small_store
    torch.manual_seed(5)
    M = torch.randn(64, 64)
    U, S, Vh = torch.linalg.svd(M)
    scale = 8.0 * (0.01 / 8.0) ** (torch.arange(64) / 63.0)
    M = ((U * scale) @ Vh).detach()
    st_rand = store.slot_state(0, 1, dim=64, rank=8, seed=7)
    st_svd = store.slot_state(0, 2, dim=64, rank=8,
                              init_values=svd_slot_init(M, 8))
    x = torch.randn(128, 64)
    out_rand = (x @ st_rand.A) @ st_rand.B
    out_svd = (x @ st_svd.A) @ st_svd.B
    # SVD slot output correlates with M's projection of x; random does not
    y_target = x @ M

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.detach().flatten(), b.detach().flatten(), dim=0))
    assert abs(cos(out_rand, y_target)) < 0.2
    assert cos(out_svd, y_target) > 0.5


def test_optimizer_state_roundtrip_through_disk(small_store):
    from cortex.model.virtual_slot import VirtualAdamW

    store = small_store
    st = store.slot_state(0, 0, dim=64, rank=8, seed=11)
    opt = VirtualAdamW(lr=1e-3)
    for _ in range(4):
        ((torch.randn(16, 64) @ st.A) @ st.B).sum().backward()
        opt.step(st)
        st.A.grad = None
        st.B.grad = None
    st_val = (st.A.detach().clone(), st.B.detach().clone())
    m_A = st.m["A"].clone()
    v_B = st.v["B"].clone()
    steps = st.step_count
    store.commit(step=4, txid="tx-rt")
    store._slots.clear()
    store._pages.clear()
    st2 = store.slot_state(0, 0, dim=64, rank=8, seed=11)
    assert st2.step_count == steps == 4
    assert torch.allclose(st2.m["A"], m_A)
    assert torch.allclose(st2.v["B"], v_B)
    assert torch.equal(st2.A.detach(), st_val[0])
    assert torch.equal(st2.B.detach(), st_val[1])


def test_optimizer_state_restore_guards_shape_mismatch(small_store):
    store = small_store
    st = store.slot_state(0, 3, dim=64, rank=8, seed=13)
    from cortex.model.virtual_slot import VirtualAdamW

    opt = VirtualAdamW()
    ((torch.randn(4, 64) @ st.A) @ st.B).sum().backward()
    opt.step(st)
    store.commit(step=1, txid="tx-shape")
    store._slots.clear()
    store._pages.clear()
    st2 = store.slot_state(0, 3, dim=32, rank=8, seed=13)
    assert st2.step_count == 0
    assert not st2.m and not st2.v


def test_evict_keeps_dirty_and_commit_before_evict_preserves_state(
        small_store):
    from cortex.model.virtual_slot import VirtualAdamW

    store = small_store
    st_keep = store.slot_state(0, 0, dim=64, rank=8, seed=1)
    st_dirty = store.slot_state(0, 1, dim=64, rank=8, seed=2)
    st_gone = store.slot_state(0, 2, dim=64, rank=8, seed=3)
    store.commit(step=1, txid="tx-evict")  # all three persisted; none dirty
    store.dirty.clear()
    evicted = store.evict_inactive({(0, 0), (0, 1)})
    assert evicted == 1
    assert (0, 2) not in store._slots
    assert (0, 0) in store._slots and (0, 1) in store._slots
    # dirty slots are never evicted (dirty (0,1) and active (0,0) both stay)
    st_dirty.A.detach().mul_(1.5)
    store.dirty.add((0, 1))
    assert store.evict_inactive({(0, 0), (0, 1)}) == 0
    # commit then evict: value restored from disk with optimizer state
    store.commit(step=2, txid="tx-evict2")
    expect = st_dirty.A.detach().clone()
    store.dirty.clear()
    store.evict_inactive(set())
    assert (0, 1) not in store._slots
    st_back = store.slot_state(0, 1, dim=64, rank=8)
    assert torch.equal(st_back.A.detach(), expect)
    assert (0, 0) not in store._slots  # evicted when nothing is active
    assert (0, 0) not in store.dirty


def test_coverage_measure_invariants(small_store):
    from cortex.model.virtual_slot import VirtualAdamW

    store = small_store
    dim, rank = 64, 8
    for row in (0, 1, 2):
        st = store.slot_state(0, row, dim, rank, seed=row)
        ((torch.randn(8, dim) @ st.A) @ st.B).sum().backward()
        VirtualAdamW().step(st)
    store.commit(step=3, txid="tx-cov")
    part = [{"update_scalar_max_abs_delta": 0.01,
             "update_scalar_mean_abs_delta": 0.002,
             "update_scalars_moved": 100, "update_scalars_total": 1024}] * 5
    rep = coverage.measure(store, None, part)
    sc = rep["scalars"]
    assert sc["allocated"] == 8 * vstore.ROWS_PER_BANK
    assert sc["initialized"] == 3 * vstore.BANK_COLS
    assert sc["demonstrably_trained"] <= sc["optimizer_updated"] \
        <= sc["received_gradients"] <= sc["activated"] < sc["allocated"]
    assert rep["invariants_hold"]
    assert rep["partition_level"]["partitions_with_committed_row"] == 1
    assert rep["scalar_level"]["rows_committed"] == 3
    # the persisted scan samples every committed row of the first 64 banks
    assert rep["scalar_level"]["sampled_rows_nonzero"] == 3
    assert "persisted_optimizer_scan" in rep


def test_coverage_partition_never_reads_uninitialized_rows(small_store):
    store = small_store
    st = store.slot_state(1, 0, dim=64, rank=8, seed=4)
    st.A.detach().zero_()
    store.commit(step=1, txid="tx-cov2")
    rep = coverage.measure(store, None, [])
    assert rep["scalars"]["initialized"] == vstore.BANK_COLS
    assert rep["update_distribution"]["steps_with_data"] == 0


def test_update_distribution_and_histogram():
    part = [{"update_scalar_max_abs_delta": float(i),
             "update_scalar_mean_abs_delta": i / 10,
             "update_scalars_moved": 10 * i,
             "update_scalars_total": 1000} for i in range(50)]
    d = coverage.update_distribution(part)
    assert d["steps_with_data"] == 50
    assert d["max_abs_delta"]["max"] == 49.0
    assert d["max_abs_delta"]["p50"] == pytest.approx(24.5, abs=1e-6)
    assert d["scalars_total_per_step"]["p95"] == 1000
    assert sum(d["hist_max_abs_delta"].values()) == 50
    empty = coverage.update_distribution([])
    assert empty == {"steps_with_data": 0}
    h = coverage.histogram([0.0] * 7, bins=3)
    assert sum(h.values()) == 7


def test_divergent_repair_round_mechanics(tmp_path, monkeypatch):
    import cortex.evaluation.divergent as div
    from cortex.evaluation.inference import extract_code

    class FakeTok:
        EOT = "<eot>"
        eot_id = 5

        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return "```python\npassed_code = True\n```"

    class FakeModel:
        def generate(self, x, max_new_tokens, temperature, eot_id):
            assert temperature in (0.2, 0.7, 1.0, 0.2)
            return x.new_zeros(x.shape[0], x.shape[1] + 4)

    class FakeAgent:
        tokenizer = FakeTok()
        model = FakeModel()

    calls = {"n": 0}

    def fake_grade(task, candidate):
        calls["n"] += 1
        round_tag = calls["n"]
        # round 1: three failures with informative stderr; repair: passes
        return {"passed": False, "exit_code": 1,
                "stderr_tail": f"AssertionError boom {round_tag}"}

    class Task:
        task_id = "t1"
        instruction = "write a function"

    captured_prefixes = []
    real_sample = div._sample_and_grade

    def spy_sample(agent, task, temperatures, max_new_tokens, prefix, tag):
        captured_prefixes.append((tag, prefix))
        return real_sample(agent, task, temperatures, max_new_tokens,
                           prefix, tag)

    monkeypatch.setattr(div, "grade_single", fake_grade)
    monkeypatch.setattr(div, "_sample_and_grade", spy_sample)
    monkeypatch.setattr(div, "extract_code", lambda text: "code")
    agent = FakeAgent()
    report = div.divergent_eval(agent, tasks=[Task()], n_hypotheses=3,
                                repair_round=True)
    assert report["aggregate"]["n_tasks"] == 1
    assert report["aggregate"]["pass_at_1_index0"] == 0.0
    assert report["aggregate"]["pass_at_n"] == 0.0
    tags = [t for t, _ in captured_prefixes]
    assert tags == ["divergent", "repair"]
    repair_prefix = captured_prefixes[1][1]
    assert "# Previous attempt failed:" in repair_prefix
    assert "AssertionError boom" in repair_prefix
    assert report["per_task"][0]["repair_round"], "repair round must run"


def test_divergent_no_repair_when_passed(monkeypatch):
    import cortex.evaluation.divergent as div

    class FakeTok:
        EOT = "<eot>"
        eot_id = 5

        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "x"

    class FakeModel:
        def generate(self, x, max_new_tokens, temperature, eot_id):
            return x.new_zeros(x.shape[0], x.shape[1] + 2)

    class FakeAgent:
        tokenizer = FakeTok()
        model = FakeModel()

    class Task:
        task_id = "t2"
        instruction = "write another function"

    monkeypatch.setattr(div, "grade_single",
                        lambda t, c: {"passed": True, "exit_code": 0,
                                      "stderr_tail": ""})
    monkeypatch.setattr(div, "extract_code", lambda text: "code")
    report = div.divergent_eval(FakeAgent(), tasks=[Task()],
                                n_hypotheses=2, repair_round=True)
    assert report["aggregate"]["pass_at_n"] == 1.0
    assert report["per_task"][0]["repair_round"] == []


def test_fingerprints_unchanged():
    from cortex.evaluation.benchmarks import fingerprint_tasks

    assert fingerprint_tasks("v1") == FINGERPRINT_V1
    assert fingerprint_tasks("v2") == FINGERPRINT_V2


def test_js_divergence_edges():
    from cortex.model.moe import js_divergence

    p = torch.tensor([0.5, 0.5, 0.0, 0.0])
    q = torch.tensor([0.5, 0.5, 0.0, 0.0])
    assert float(js_divergence(p, q)) == pytest.approx(0.0, abs=1e-8)
    r = torch.tensor([0.0, 0.0, 0.5, 0.5])
    assert float(js_divergence(p, r)) > 0.5
    assert float(js_divergence(p, r)) == pytest.approx(
        float(js_divergence(r, p)), abs=1e-8)


def test_stage8_report_shape_when_written(tmp_path):
    from scripts.stage8_run import stage_i_summary

    a = {"v1_fingerprint_ok": True, "v2_fingerprint_ok": True}
    c = {"coverage": {"scalars": {"allocated": 100, "initialized": 10,
                                  "activated": 5, "received_gradients": 5,
                                  "optimizer_updated": 5,
                                  "demonstrably_trained": 5},
                      "scalar_level": {}, "partition_level": {},
                      "invariants_hold": True}}
    h = {"decision": "REFUSED-CPU-ONLY"}
    d = {"recovery_verified_at_expanded_coverage": True}
    s = stage_i_summary(a, c, h, d, {})
    assert s["owned_by"] == "Fibonera"
    assert s["gpu_gate"] == "REFUSED-CPU-ONLY"
    assert s["physical_model_status"]["2b_trained"] is False
    assert s["invariants_hold"] is True
