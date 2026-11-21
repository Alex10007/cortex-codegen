"""Stage 10 tests: deep failure diagnosis, dataset audit, capability
maturity, remediation batching split (legacy vs chunked), authoritative-store
read-only invariant, lineage-store staging, and audit tamper detection.
Real components; no sandbox execution is needed for these paths.
"""
from __future__ import annotations

import hashlib
import json

import pytest

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"


@pytest.fixture(scope="module")
def tok():
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(ARTIFACTS / "tokenizer" / "tokenizer.json"))


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


# ---------------------------------------------------------------- C1 diagnosis

def test_diagnose_passed():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    d = diagnose_candidate("```python\nx = 1\n```",
                           {"passed": True, "exit_code": 0,
                            "stderr_tail": ""}, "make `f(` work")
    assert d["diagnosis"] == "passed"


def test_diagnose_degenerate_repetition():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    body = "def f():\n    return 1\n" + ("return None\n" * 12)
    d = diagnose_candidate(body, {"passed": False, "exit_code": 1,
                                  "stderr_tail": ""}, "make `f(` work")
    assert d["diagnosis"] == "degenerate_generation"
    assert "repeated" in d["evidence"]


def test_diagnose_truncation_unclosed_fence():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    body = "text\n```python\ndef f():\n    return 1"
    d = diagnose_candidate(body, {"passed": False, "exit_code": 1,
                                  "stderr_tail": ""}, "make `f(` work")
    assert d["diagnosis"] == "truncation"
    assert "never closed" in d["evidence"]


def test_diagnose_syntax_error():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    body = "```python\nclass class class:\n    pass\n```"
    d = diagnose_candidate(body, {"passed": False, "exit_code": 1,
                                  "stderr_tail": "SyntaxError"}, "")
    assert d["diagnosis"] == "syntax_error"


def test_diagnose_undefined_dependency():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    body = ("```python\nimport nosuchmodule\n\ndef f():\n    return 1\n```")
    d = diagnose_candidate(
        body, {"passed": False, "exit_code": 1,
               "stderr_tail": "ModuleNotFoundError: No module named "
                              "'nosuchmodule'"},
        "make `f(` work")
    assert d["diagnosis"] == "undefined_dependency"
    assert "ModuleNotFoundError" in d["evidence"]


def test_diagnose_signature_mismatch():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    body = "```python\ndef totally_other():\n    return 1\n```"
    d = diagnose_candidate(body, {"passed": False, "exit_code": 1,
                                  "stderr_tail": ""},
                           "implement `intervals_merge(a)` merging ranges")
    assert d["diagnosis"] == "signature_mismatch"
    assert "intervals_merge" in d["evidence"]


def test_diagnose_incorrect_logic():
    from cortex.evaluation.failure_analysis import diagnose_candidate

    body = "```python\ndef add(x):\n    return x - 1\n```"
    d = diagnose_candidate(
        body,
        {"passed": False, "exit_code": 1,
         "stderr_tail": "AssertionError: assert add(1) == 2",
         "failure_category": "assertion"},
        "implement `add(x)`")
    assert d["diagnosis"] == "incorrect_logic"


def test_diagnose_rank_orders_by_severity():
    from cortex.evaluation.failure_analysis import root_cause_rank

    ranked = root_cause_rank({"incorrect_logic": 2, "truncation": 5,
                              "syntax_error": 1, "passed": 9})
    assert ranked[0][0] == "truncation"
    assert "passed" not in [k for k, _ in ranked]


# ---------------------------------------------------------------- C2 dataset audit

def _write_ds(tmp_path, n_short=2, n_long=1, split="train"):
    dur = {"train", "val", "heldout"}
    splits = (sorted(dur) if split == "train" else [split])

    def _split_for(i: int) -> str:
        return splits[i % len(splits)]

    rec = {"name": "f", "module": "functools", "kind": "function",
           "instruction": "Implement `f(x)` returning x plus one.",
           "code": "def f(x):\n    return x + 1\n",
           "doctests_attempted": 0, "provenance": {"source": "stdlib"},
           "split": split, "contamination_scores": {}}
    recs = []
    idx = 0
    for i in range(n_short):
        r = dict(rec)
        r["code"] = f"def f(x):\n    return x + {i + 1}\n"
        r["split"] = _split_for(idx)
        r["content_sha256"] = hashlib.sha256(
            json.dumps([i, r["code"]]).encode()).hexdigest()
        recs.append(r)
        idx += 1
    long_code = ("def g(x):\n"
                 + "".join("    x = x + 1\n" for _ in range(400))
                 + "    return x\n")
    for i in range(n_long):
        r = dict(rec)
        r["code"] = long_code
        r["split"] = _split_for(idx)
        r["content_sha256"] = hashlib.sha256(
            json.dumps(["L", i]).encode()).hexdigest()
        recs.append(r)
        idx += 1
    out = tmp_path / "ds.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in recs))
    return out


def test_dataset_audit_fit_and_duplicates(tmp_path, tok):
    from cortex.evaluation.dataset_audit import audit_dataset

    ds = _write_ds(tmp_path)
    rep = audit_dataset(ds, tok, 128)
    assert rep["n_records"] == 3
    assert rep["records_dropped_by_fit_filter"] >= 1
    assert 0 < rep["fit_rate_at_training_seq_len"] < 1
    assert rep["exact_code_duplicates"] == 0


def test_dataset_audit_detects_duplicates(tmp_path, tok):
    from cortex.evaluation.dataset_audit import audit_dataset

    ds = _write_ds(tmp_path)
    recs = [json.loads(l) for l in ds.read_text().splitlines()]
    recs.append(dict(recs[0]))
    ds.write_text("\n".join(json.dumps(r) for r in recs))
    rep = audit_dataset(ds, tok, 128)
    assert rep["exact_full_duplicates"] >= 1


def test_split_determinism_true_and_tamper(tmp_path):
    from cortex.evaluation.dataset_audit import split_is_deterministic

    ds = _write_ds(tmp_path)
    assert split_is_deterministic(ds) is True
    recs = [json.loads(l) for l in ds.read_text().splitlines()]
    # the same content hash appearing under two splits is not deterministic
    other = next(s for s in ("train", "val", "heldout")
                 if s != recs[0]["split"])
    recs.append(dict(recs[0], split=other))
    ds.write_text("\n".join(json.dumps(r) for r in recs))
    assert split_is_deterministic(ds) is False


def test_split_determinism_rejects_bad_label(tmp_path):
    from cortex.evaluation.dataset_audit import split_is_deterministic

    ds = _write_ds(tmp_path)
    recs = [json.loads(l) for l in ds.read_text().splitlines()]
    recs[0]["split"] = "dev"
    ds.write_text("\n".join(json.dumps(r) for r in recs))
    assert split_is_deterministic(ds) is False


# ------------------------------------------------------------ capability maturity

def test_maturity_cumulative_levels():
    from cortex.evaluation.capability_maturity import classify_capability

    assert classify_capability({}) == 0
    assert classify_capability({"implemented": True}) == 1
    assert classify_capability({"implemented": True, "executes": True}) == 2
    assert classify_capability({"implemented": True, "executes": True,
                                "learns": True}) == 3
    assert classify_capability({"learns": True}) == 0, \
        "gates are cumulative; skipping implemented must not pay off"


def test_maturity_blocked_reports_but_keeps_earned_levels():
    from cortex.evaluation.capability_maturity import classify_capability

    ev = {"implemented": True, "executes": True,
          "blocked": "pass@1 = 0.0"}
    assert classify_capability(ev) == 2


def test_maturity_build_report_next_gate_string():
    from cortex.evaluation.capability_maturity import build_report

    rep = build_report({
        "b": {"implemented": True, "executes": True},
        "a": {"implemented": True, "blocked": "no data"},
        "c": {g: True for g in (
            "implemented", "executes", "learns", "improves_heldout",
            "independent_reproduction", "outperforms_baseline", "transfers")},
    })
    rows = {r["capability"]: r for r in rep["capabilities"]}
    assert list(rows) == ["a", "b", "c"]
    assert rows["a"]["level"] == 1
    assert "blocked" in rows["a"]["status"]
    assert rows["b"]["status"] == "next gate unmet: learns"
    assert rows["c"]["level"] == 7
    assert rows["c"]["status"] == "max level"


# ------------------------------------- remediation batching (legacy vs chunked)

def test_chunked_batching_fits_long_records(tmp_path, tok):
    from cortex.training.sft import make_batch
    from cortex.training.longcontext import make_batch_chunked

    ds_path = _write_ds(tmp_path, n_short=0, n_long=2)
    recs = [json.loads(l) for l in ds_path.read_text().splitlines()]
    x1, y1 = make_batch(recs, tok, 256)
    x2, y2 = make_batch_chunked(recs, tok, 256)
    ds_path.unlink()
    assert x1 is None, "legacy path must skip over-long records"
    assert x2 is not None and x2.shape[0] >= 1


# ------------------------------------------- authoritative store read-only path

def test_warm_store_copy_is_read_only(small_store):
    import numpy as np
    import torch

    from cortex.model.virtual_slot import VirtualAdamW
    from scripts.stage10_run import _warm_store_copy

    auth = small_store
    st = auth.slot_state(0, 1, 192, 16)
    x = torch.randn(192)
    ((x @ st.A) @ st.B).sum().backward()
    VirtualAdamW().step(st)
    auth.commit(77, "tx-warm-src")
    npz_before = (auth.root / "adam-00000-000001.npz").read_bytes()
    manifest_before = json.loads(auth.manifest_path.read_text())

    warm = _warm_store_copy([(0, 1)], 192, 16, 4242, auth)
    try:
        assert json.loads(auth.manifest_path.read_text()) == manifest_before
        assert (auth.root / "adam-00000-000001.npz").read_bytes() == \
            npz_before
        assert int(json.loads(
            warm.manifest_path.read_text())["committed_step"]) == 0
        wst = warm.slot_state(0, 1, 192, 16)
        assert np.allclose(np.asarray(wst.A.detach()),
                           np.asarray(st.A.detach()))
    finally:
        warm.close()


def test_stage_store_for_lineage_sets_committed_step(small_store, tmp_path):
    import torch

    from scripts.stage10_run import _stage_store_for_lineage

    auth = small_store
    import numpy as np

    st = auth.slot_state(1, 2, 192, 16)
    page = auth.page(1)
    page[2, :4096] = np.linspace(0.1, 0.2, 4096)
    page.flush()
    auth.commit(10, "tx-lineage")
    skim = auth.slot_state(1, 2, 192, 16)

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    phys = ckpt_dir / "joint-step-000010.pt"
    torch.save({"model": "x", "step": 10, "model_state": {},
                "phys_opt_state": {}}, phys)
    (ckpt_dir / "lineage.json").write_text(json.dumps({
        "step": 10, "transaction_id": "tx-test",
        "physical_checkpoint": phys.name,
        "physical_sha256": hashlib.sha256(
            phys.read_bytes()).hexdigest(),
        "virtual_routes": {"0": [1, 2]}, "virtual_rank": 16,
        "virtual_seed": 4242}))

    staged = _stage_store_for_lineage(ckpt_dir, auth)
    try:
        manifest = json.loads(staged.manifest_path.read_text())
        assert manifest["committed_step"] == 10
        stored = staged.slot_state(1, 2, 192, 16)
        assert np.allclose(np.asarray(stored.A.detach()),
                           np.asarray(skim.A.detach()))
    finally:
        staged.close()


def test_staged_pair_passes_stale_guard(small_store, tmp_path, monkeypatch):
    import torch

    import cortex.virtual.store as vstore  # noqa: F401
    from scripts.stage10_run import _stage_store_for_lineage

    auth = small_store
    import numpy as np

    st = auth.slot_state(0, 1, 192, 16)
    page = auth.page(0)
    page[1, :4096] = np.linspace(0.3, 0.4, 4096)
    page.flush()
    auth.commit(10, "tx-lineage2")

    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    phys = ckpt_dir / "joint-step-000010.pt"
    torch.save({"model": "x", "step": 10, "model_state": {},
                "phys_opt_state": {}}, phys)
    (ckpt_dir / "lineage.json").write_text(json.dumps({
        "step": 10, "transaction_id": "tx-test",
        "physical_checkpoint": phys.name,
        "physical_sha256": hashlib.sha256(
            phys.read_bytes()).hexdigest(),
        "virtual_routes": {"0": [1, 1]}, "virtual_rank": 16,
        "virtual_seed": 4242}))

    staged = _stage_store_for_lineage(ckpt_dir, auth)
    try:
        # digest verification against the staged manifest must succeed
        integrity = staged.verify_integrity()
        assert integrity["integrity_ok"]
        manifest = json.loads(staged.manifest_path.read_text())
        assert manifest["committed_step"] >= 10, \
            "staged manifest must not sit behind the checkpoint"
    finally:
        staged.close()


# ------------------------------------------------------------------ stage J audit

def test_stage10_audit_flags_tampered_fingerprint(monkeypatch):
    import scripts.stage10_audit as audit

    monkeypatch.setattr(audit, "FINGERPRINT_V1", "0" * 64)
    report = audit.run_audit()
    fp = report["independent_logic"]["fingerprints"]
    assert fp["v1_matches_immutable"] is False
    assert report["checks"]["J_reproducibility_integrity"]["status"] == "FAIL"


def test_stage10_audit_virtual_recount_matches_stage9_final():
    import scripts.stage10_audit as audit

    report = audit.run_audit()
    acct = report["independent_logic"]["virtual_accounting_recount"]
    assert acct["matches_stage9_final"] is True
    assert acct["corrupt_npz_files"] == []


def test_stage10_artifacts_governance():
    import scripts.stage10_audit as audit

    report = audit.run_audit()
    checks = report["checks"]
    arts = checks["J_reproducibility_integrity"]["evidence"]["artifacts"]
    assert arts, "stage-10 artifact inventory must be evaluated"
    missing = [n for n, state in arts.items() if state != "OK"]
    assert missing == [], \
        f"missing/incomplete stage-10 artifacts: {missing}"
    for name in arts:
        doc = json.loads((audit.ART / name).read_text())
        assert doc.get("owned_by") == "Fibonera"
