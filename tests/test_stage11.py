"""Stage 11 tests: chunked-stream training gate (coordinator + dense),
stream-audit accounting, and artifact/audit integrity. Real components;
the coordinator test runs actual training steps.
"""
from __future__ import annotations

import json

import pytest

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
DATASET = ARTIFACTS / "sft" / "dataset_v3.jsonl"
WARM6 = ARTIFACTS / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt"
STAGE11_ALLOWED_DECISIONS = ("PROMOTED", "NOT PROMOTED")
STAGE11_ALLOWED_PROMOTION_DECISIONS = (
    "PROMOTED - first benchmark improvement over the executed baseline",
    "NOT PROMOTED - no immutable-benchmark improvement; negative result "
    "recorded")


@pytest.fixture(scope="module")
def tok():
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(ARTIFACTS / "tokenizer" / "tokenizer.json"))


@pytest.fixture(scope="module")
def data():
    from cortex.training.sft import load_split

    return (load_split(str(DATASET), "train"),
            load_split(str(DATASET), "heldout"))


# ------------------------------------------------- coordinator chunked path

def test_joint_train_step_chunked_trains_long_records(tmp_path, monkeypatch,
                                                      tok, data):
    import cortex.virtual.store as vstore
    from cortex.model.configs import CORTEX_DEMO
    from cortex.virtual.coordinator import JointTrainer
    from cortex.virtual.store import VirtualParameterStore

    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL", 8 * vstore.ROWS_PER_BANK)
    store = VirtualParameterStore(tmp_path / "vstore")
    store.allocate()

    train_recs, _ = data
    long_recs = sorted(train_recs, key=lambda r: -len(r["code"]))[:4]
    trainer = JointTrainer(CORTEX_DEMO, tok, store, tmp_path / "out",
                           init_checkpoint=str(WARM6), virtual_rank=16)
    trainer.attach_partition_group([(b, 1) for b in range(8)])

    legacy = trainer.train_step(long_recs, 256, log=False, chunked=False)
    assert legacy is None, \
        "legacy fit filter must skip these over-long records (stage-10 " \
        "measurement reproduced)"

    chunked = trainer.train_step(long_recs, 256, log=False, chunked=True)
    assert chunked is not None, \
        "chunked path must produce a real training step for the same " \
        "records the legacy path silently drops"
    assert chunked["ce_completion_tokens"] > 0
    assert chunked["physical_grad_norm"] >= 0


def test_train_step_default_semantics_unchanged(tmp_path, monkeypatch, tok,
                                                data):
    import cortex.virtual.store as vstore
    from cortex.model.configs import CORTEX_DEMO
    from cortex.virtual.coordinator import JointTrainer
    from cortex.virtual.store import VirtualParameterStore

    monkeypatch.setattr(vstore, "N_BANKS", 8)
    monkeypatch.setattr(vstore, "TOTAL_VIRTUAL", 8 * vstore.ROWS_PER_BANK)
    store = VirtualParameterStore(tmp_path / "vstore")
    store.allocate()
    train_recs, _ = data
    fit_recs = sorted(train_recs, key=lambda r: len(r["code"]))[:4]
    trainer = JointTrainer(CORTEX_DEMO, tok, store, tmp_path / "out",
                           init_checkpoint=str(WARM6), virtual_rank=16)
    trainer.attach_partition_group([(b, 1) for b in range(8)])
    # default call (no flag) must take the legacy path for fitting records
    rec_default = trainer.train_step(fit_recs, 256, log=False)
    assert rec_default is not None


# ------------------------------------------------------- stream audit facts

def test_stream_audit_counts(data, tok):
    from cortex.training.longcontext import chunk_record

    train_recs, heldout = data
    recs = train_recs + heldout
    legacy_fit = chunked_fit = 0
    for r in recs:
        p = tok.encode(r["instruction"] + "\n")
        c = tok.encode(r["code"].strip() + "\n" + tok.EOT)
        if len(p) + len(c) + 1 <= 256:
            legacy_fit += 1
        if chunk_record(r, tok, 256):
            chunked_fit += 1
    assert chunked_fit > legacy_fit, \
        "chunking must strictly increase record coverage"
    assert legacy_fit / len(recs) < 0.5, \
        "legacy coverage limitation (stage-10/11 measured) must hold"
    assert chunked_fit / len(recs) > 0.5


def test_chunk_record_rejects_over_long_instruction(tok):
    from cortex.training.longcontext import chunk_record

    rec = {"instruction": "word " * 900, "code": "def f():\n    return 1\n"}
    assert chunk_record(rec, tok, 256) is None


# --------------------------------------------------------- artifact / audit

def test_stage11_artifacts_and_promotion_rule():
    import scripts.stage11_run as s11

    rep = json.loads((ARTIFACTS / "stage11_report.json").read_text())
    assert rep["owned_by"] == "Fibonera"
    b = json.loads((ARTIFACTS / "stage11_extended_training.json").read_text())
    assert b["experiment"]["decision"] in STAGE11_ALLOWED_DECISIONS
    assert b["promotion_decision"] in STAGE11_ALLOWED_PROMOTION_DECISIONS
    assert (b["experiment"]["decision"] == "PROMOTED") == \
        b["promotion_decision"].startswith("PROMOTED - first benchmark")
    assert b["pre_registered_promotion_rule"] == s11.PROMOTION_RULE
    arms = b["arms"]
    assert arms["chunked_stream"][
        "steps_with_targets_total"] == arms["chunked_stream"][
        "steps_out_of_4000"]


def test_stage11_audit_integrity():
    audit = json.loads(
        (ARTIFACTS / "stage11_acceptance_audit.json").read_text())
    assert audit["owned_by"] == "Fibonera"
    checks = audit["checks"]
    assert checks["fingerprint_v1_recomputed_match"] is True
    assert checks["fingerprint_v2_recomputed_match"] is True
    assert checks["benchmark_files_untouched_vs_stage10"] is True
    assert checks["virtual_unique_trained_scalars_recount"] == 216_834_048


def test_stage11_checkpoints_recorded():
    b = json.loads((ARTIFACTS / "stage11_extended_training.json").read_text())
    found = {c["batching"]: c for c in b["checkpoints"]}
    assert set(found) == {"legacy_stream_control", "chunked_stream"}
    for c in found.values():
        assert len(c["sha256"]) == 64
        assert (REPO / c["path"].replace(str(REPO) + "/", "")) \
            .exists() or __import__("pathlib").Path(c["path"]).exists()
