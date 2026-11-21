"""Stage 9 tests: failure taxonomy, v3 suite, MoE probe, divergent-repair
targeting, audit integrity, and recovery mechanics (stale pairs, corrupt
optimizer state). Real components; only sandbox grading is stubbed, at the
same boundary as the stage-8 tests.
"""
from __future__ import annotations

import json

import pytest

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "artifacts" / "tokenizer" / "tokenizer.json"
DATASET = REPO / "artifacts" / "sft" / "dataset_v3.jsonl"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"

STATUSES = {"PASS", "FAIL", "BLOCKED", "NOT YET TESTED"}


@pytest.fixture(scope="module")
def tok():
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(TOKENIZER))


@pytest.fixture()
def small_store(tmp_path, monkeypatch):
    import cortex.virtual.store as vstore

    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL", 8 * vstore.ROWS_PER_BANK)
    from cortex.virtual.store import VirtualParameterStore

    store = VirtualParameterStore(tmp_path / "vstore")
    store.allocate()
    store.manifest = json.loads(store.manifest_path.read_text())
    return store


# ------------------------------------------------------------- failure taxonomy

def test_failure_tax_categories():
    from cortex.evaluation.failure_tax import classify

    assert classify({"passed": True, "exit_code": 0, "stderr_tail": ""}) == \
        "passed"
    assert classify({"passed": False, "exit_code": 1,
                     "stderr_tail": "  SyntaxError: bad token"}) == "syntax"
    assert classify({"passed": False, "exit_code": 1,
                     "stderr_tail": "IndentationError: unexpected indent"}) \
        == "syntax"
    assert classify({"passed": False, "exit_code": 1,
                     "stderr_tail": "ModuleNotFoundError: No module named "
                                    "nope"}) == "import"
    assert classify({"passed": False, "exit_code": 1,
                     "stderr_tail": "ZeroDivisionError: division by zero"}) \
        == "runtime"
    assert classify({"passed": False, "exit_code": 1,
                     "stderr_tail": "E   AssertionError: assert 1 == 2"}) == \
        "assertion"
    assert classify({"passed": False, "exit_code": 124,
                     "stderr_tail": "signature"}) == "timeout"
    assert classify({"passed": False, "exit_code": 0, "stderr_tail": ""}) == \
        "empty"
    assert classify({"passed": False, "exit_code": None,
                     "stderr_tail": "segfault at 0x0"}) == "unknown"


def test_repair_hints_do_not_leak_hidden_tests():
    from cortex.evaluation.failure_tax import CATEGORIES, repair_hint
    from cortex.evaluation.tasks import TASKS
    from cortex.evaluation.tasks_v3 import TASKS_V3

    secret_bits = [t.test_code for t in TASKS] + \
                  [t.test_code for t in TASKS_V3]
    assert len(CATEGORIES) == 8
    for cat in CATEGORIES:
        hint = repair_hint(cat)
        assert isinstance(hint, str)
        for test_code in secret_bits:
            # no hidden-test names, helpers or literal fragments may leak
            for line in test_code.splitlines():
                stripped = line.strip()
                if len(stripped) > 12 and stripped.startswith("assert "):
                    assert stripped not in hint, cat


def test_failure_breakdown_counts_and_requires_grade():
    from cortex.evaluation.failure_tax import failure_breakdown

    hyps = [{"grade": {"passed": False, "exit_code": 1,
                       "stderr_tail": "SyntaxError: x"}},
            {"grade": {"passed": True, "exit_code": 0, "stderr_tail": ""}},
            {"grade": None}]
    b = failure_breakdown(hyps)
    assert b["syntax"] == 1 and b["passed"] == 1 and b["empty"] == 1
    assert sum(b.values()) == 3


# ------------------------------------------------------------------- v3 suite

def test_tasks_v3_shape_fingerprint_and_families():
    from cortex.evaluation.tasks_v3 import (FAMILY, TASKS_V3, v3_fingerprint)

    assert len(TASKS_V3) == 12
    ids = [t.task_id for t in TASKS_V3]
    assert len(set(ids)) == len(ids)
    assert set(t.task_id for t in TASKS_V3) == set(FAMILY)
    assert len({t.task_id for t in TASKS_V3}
               & {"t-palindrome"}) == 0, "v3 must not overlap v1 ids"
    fp1, fp2 = v3_fingerprint(), v3_fingerprint()
    assert fp1 == fp2 and len(fp1) == 64
    from cortex.evaluation.benchmarks import fingerprint_tasks

    assert fingerprint_tasks("v3") == fp1
    with pytest.raises(ValueError):
        fingerprint_tasks("v9")


def test_v3_reference_solutions_pass_own_suites():
    from cortex.evaluation.runner import grade_single
    from cortex.evaluation.tasks_v3 import (FAMILY, TASKS_V3, build_report,
                                            v3_fingerprint)

    results = []
    for t in TASKS_V3:
        g = grade_single(t, t.reference_solution)
        g["family"] = FAMILY.get(t.task_id, "unknown")
        results.append(g)
    report = build_report("harness_sanity", results)
    assert report["pass_at_1"] == 1.0, [r["task_id"] for r in results
                                        if not r["passed"]]
    assert report["suite_fingerprint"] == v3_fingerprint()
    assert len(report["per_family"]) >= 6


def test_fingerprints_v1_v2_unchanged():
    from cortex.evaluation.benchmarks import fingerprint_tasks

    assert fingerprint_tasks("v1") == FINGERPRINT_V1
    assert fingerprint_tasks("v2") == FINGERPRINT_V2


# ------------------------------------------------------------------ MoE probe

def _demo_records():
    from cortex.training.sft import load_split

    return load_split(str(DATASET), "train")[:16]


def test_moe_probe_and_gradient_flow(tok, small_store):
    from cortex.evaluation.moe_probe import (assignment_stability,
                                             gradient_flow, probe)
    from cortex.model.configs import CORTEX_DEMO
    from cortex.model.moe import CortexMoETransformer
    from cortex.training.sft import make_batch

    store = small_store
    model = CortexMoETransformer(CORTEX_DEMO)
    slots = {r: store.slot_state(r, r, CORTEX_DEMO.dim, 16)
             for r in range(model.n_virtual)}
    model.attach_virtual_slots(slots)
    recs = _demo_records()
    batches = [recs[i:i + 4] for i in (0, 4, 8, 12)]
    batches = [b for b in batches if make_batch(b, tok, 256)[0] is not None]
    assert batches

    rep = probe(model, batches, tok, make_batch)
    assert rep["n_layers"] == len(model.moe_layers)
    for layer_stats in rep["per_layer"]:
        assert layer_stats["n_experts"] == 12
        assert layer_stats["top_k"] == 2
        assert 0.0 <= layer_stats["mean_max_prob"] <= 1.0
        assert layer_stats["mean_entropy_bits"] <= 3.585 + 1e-6
    assert "hypothesis" in rep["root_cause"]
    assert rep["root_cause"]["near_uniform_routing"] is True, \
        "untrained demo router must be detected as near-uniform"

    x, y = make_batch(batches[0], tok, 256)
    assert x is not None

    def loss_fn():
        _, loss = model(x, y, pad_mask=x == tok.pad_id)
        return loss

    model.train()
    grad = gradient_flow(model, loss_fn)
    model.zero_grad(set_to_none=True)
    assert grad["router_grad_nonzero"], "router must receive gradients"
    assert grad["expert_grad_nonzero"]
    assert grad["n_router_params"] > 0

    stab = assignment_stability(model, batches[0], tok, make_batch)
    assert stab["stable"], "top-k router assignments must be deterministic"


# -------------------------------------------------- divergent-repair targeting

class _FakeTok:
    EOT = "<eot>"
    eot_id = 5

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return "```python\nx = 1\n```"


class _FakeModel:
    def generate(self, x, max_new_tokens, temperature, eot_id):
        return x.new_zeros(x.shape[0], x.shape[1] + 4)


class _FakeAgent:
    tokenizer = _FakeTok()
    model = _FakeModel()


class _Task:
    task_id = "t9"
    instruction = "write a function"


def test_divergent_targets_dominant_failure_category(monkeypatch):
    import cortex.evaluation.divergent as div
    from cortex.evaluation.failure_tax import repair_hint

    grades = iter([
        {"passed": False, "exit_code": 1,
         "stderr_tail": "ModuleNotFoundError: nope"},      # import
        {"passed": False, "exit_code": 1, "stderr_tail": "SyntaxError: ^"},

        {"passed": False, "exit_code": 1, "stderr_tail": "SyntaxError: ^"},
        {"passed": False, "exit_code": 1, "stderr_tail": "SyntaxError: ^"},
    ])

    def fake_grade(task, candidate):
        return next(grades)

    captured = {}

    monkeypatch.setattr(div, "grade_single", fake_grade)
    monkeypatch.setattr(div, "extract_code", lambda t: "code")

    real_sample = div._sample_and_grade
    seen_prefixes = []

    def spy2(agent, task, temps, mnt, prefix, tag):
        seen_prefixes.append((tag, prefix))
        return real_sample(agent, task, temps, mnt, prefix, tag)

    monkeypatch.setattr(div, "_sample_and_grade", spy2)
    report = div.divergent_eval(_FakeAgent(), tasks=[_Task()],
                                n_hypotheses=3, repair_round=True,
                                temperatures=(0.2, 0.2, 0.2))

    assert report["aggregate"]["failure_categories_round1"]["syntax"] == 2
    assert report["aggregate"]["failure_categories_round1"]["import"] == 1
    tag, prefix = seen_prefixes[1]
    assert tag == "repair"
    # targeted repair: dominant category hint actually present, no leakage
    assert "did not PARSE" in prefix
    assert "Previous attempt failed" in prefix


def test_divergent_aggregate_backward_compatible(monkeypatch):
    import cortex.evaluation.divergent as div

    monkeypatch.setattr(div, "grade_single", lambda t, c: {
        "passed": True, "exit_code": 0, "stderr_tail": ""})
    monkeypatch.setattr(div, "extract_code", lambda t: "code")
    report = div.divergent_eval(_FakeAgent(), tasks=[_Task()],
                                n_hypotheses=2, temperatures=(0.2, 0.7),
                                repair_round=False)
    agg = report["aggregate"]
    for key in ("n_tasks", "pass_at_1_index0", "pass_at_n",
                "contradiction_rate", "mean_convergence",
                "distinct_candidate_rate", "tokens_consumed",
                "seconds_elapsed", "failure_categories_round1"):
        assert key in agg, key
    assert agg["pass_at_n"] == 1.0


# --------------------------------------------------------------- audit logic

def _toy_tasks():
    from cortex.evaluation.tasks import EvalTask

    return [
        EvalTask(task_id="a", instruction="ia", module_name="a.py",
                 test_code="def test_a():\n    assert True\n",
                 reference_solution="x = 1"),
        EvalTask(task_id="b", instruction="ib", module_name="b.py",
                 test_code="def test_b():\n    assert False\n",
                 reference_solution="y = 2"),
    ]


def test_task_v3_style_grading_categories_wire_together(tmp_path, monkeypatch):
    from cortex.evaluation import runner as runner_mod
    from cortex.evaluation.failure_tax import CATEGORIES
    from scripts.stage9_run import grade_tasks_fixed_decoding

    class _M:
        def generate(self, x, max_new_tokens, temperature, top_k, eot_id):
            return x.new_zeros(x.shape[0], x.shape[1] + 3)

    class _Sub:
        tokenizer = _FakeTok()
        model = _M()
        checkpoint_path = "toy"

    monkeypatch.setattr(runner_mod, "grade_single",
                        lambda t, c: {"passed": t.task_id == "a",
                                      "exit_code": 0 if t.task_id == "a"
                                      else 1, "stderr_tail": ""})
    results = grade_tasks_fixed_decoding(_Sub(), _toy_tasks())
    assert results[0]["passed"] and not results[1]["passed"]
    assert all(r["family"] == "v1" for r in results)
    assert set(r["failure_category"] for r in results) <= set(CATEGORIES)


def test_audit_mechanics_and_statuses():
    from scripts.stage9_audit import run_audit

    report = run_audit()
    checks = report["checks"]
    assert checks, "audit must produce its criterion table"
    for name, c in checks.items():
        assert c["status"] in STATUSES, name
    counts = report["summary_counts"]
    assert sum(counts.values()) == len(checks)
    assert checks["I_reproducibility_fingerprints"]["status"] == "PASS", \
        "v1/v2 fingerprints + lineage + digests must hold on the real store"
    assert report["independent_logic"]["virtual_accounting"][
        "corrupt_npz_files"] == []
    assert report["independent_logic"]["virtual_accounting"][
        "unique_trained_scalars_recomputed"] > 0


def test_audit_catches_tampered_fingerprint_constants(monkeypatch):
    import scripts.stage9_audit as audit

    monkeypatch.setattr(audit, "FINGERPRINT_V1", "0" * 64)
    report = audit.run_audit()
    fp = report["independent_logic"]["fingerprints"]
    assert fp["v1_matches_immutable"] is False, \
        "tampered constant must be flagged by recomputation"


# ------------------------------------------------------------------- recovery

def test_stale_pair_refused_and_behind_never_allowed(tmp_path, monkeypatch):
    import torch

    import cortex.virtual.store as vstore
    from cortex.virtual.coordinator import restart_from_checkpoint

    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL",
                        8 * vstore.ROWS_PER_BANK)
    root = tmp_path / "vstore"
    store = vstore.VirtualParameterStore(root)
    store.manifest = store.allocate()
    store.manifest = json.loads(store.manifest_path.read_text())

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    phys = ckpt_dir / "joint-step-000010.pt"
    torch.save({"model": "cortex-demo", "step": 10, "model_state": {},
                "phys_opt_state": {}}, phys)
    (ckpt_dir / "lineage.json").write_text(json.dumps({
        "step": 10, "transaction_id": "tx-test",
        "physical_checkpoint": phys.name,
        "physical_sha256": __import__("hashlib").sha256(
            phys.read_bytes()).hexdigest(),
        "virtual_routes": {}, "virtual_rank": 16, "virtual_seed": 7}))
    # manifest behind the checkpoint: partial commit pair
    store.manifest["committed_step"] = 4
    for allow in (False, True):
        with pytest.raises(RuntimeError, match="stale pair"):
            restart_from_checkpoint(ckpt_dir, None, store, None, [], 256,
                                    allow_ahead=allow)


def test_corrupt_optimizer_npz_is_detected(small_store):
    import numpy as np
    import torch

    from cortex.model.virtual_slot import VirtualAdamW

    store = small_store
    st = store.slot_state(0, 0, 192, 16)
    x = torch.randn(192)
    ((x @ st.A) @ st.B).sum().backward()
    VirtualAdamW().step(st)
    store.commit(3, "tx-corrupt-test")
    npz = store.root / "adam-00000-000000.npz"
    assert npz.exists()
    raw = npz.read_bytes()
    npz.write_bytes(raw[: len(raw) // 2])

    store._slots.clear()
    with pytest.raises(Exception):
        store.slot_state(0, 0, 192, 16)


def test_corrupt_virtual_row_digest_detected(small_store):
    import numpy as np
    import torch

    store = small_store
    st = store.slot_state(1, 2, 192, 16)
    with torch.no_grad():
        st.A.normal_()
    store.commit(1, "tx-digest-test")
    page = store.page(1)
    page[2, 0] += 1.5
    page.flush()
    out = store.verify_integrity()
    assert not out["integrity_ok"]
    assert "1/2" in out["bad_rows"]


def test_exact_2b_param_count():
    from cortex.model.configs import CORTEX_2B

    assert CORTEX_2B.theoretical_param_count() == 2_056_012_800


def test_hardware_report_blocked_without_gpus():
    from scripts.stage9_run import stage_a_infra

    report = stage_a_infra()
    assert report["kind"] == "cortex-stage9-hardware-report"
    assert report["owned_by"] == "Fibonera"
    if not report["hardware_inventory"]["has_cuda"]:
        assert report["decision"] == "BLOCKED"
        assert report["gpu_qualification_executed"] is False
        assert "NCCL" in " ".join(report["missing_requirements"])
