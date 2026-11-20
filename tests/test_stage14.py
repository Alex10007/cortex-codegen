"""Stage 14 tests: execution-feedback verification correctness (real
sandbox execution + real stdlib doctest against reference docstrings),
-feedback-row surface form and target integrity, benchmark leakage
control, and post-run artifact integrity. Real components, real
tokenizer, no mocks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
DATASET = ARTIFACTS / "sft" / "dataset_v3.jsonl"

_ADD_REC = {
    "name": "add_two",
    "module": "synthetic",
    "instruction": "write a function that adds two numbers",
    "code": (
        "def add_two(a, b):\n"
        '    """Return the sum of a and b.\n'
        "\n"
        "    >>> add_two(1, 2)\n"
        "    3\n"
        "    >>> add_two(10, -4)\n"
        "    6\n"
        '    """\n'
        "    return a + b\n"
    ),
}

_GOOD_CANDIDATE = _ADD_REC["code"].strip()
_BROKEN_CANDIDATE = "def add_two(a, b):\n    return a + b\n  if:\n"
_WRONG_ALG_CANDIDATE = (
    "def add_two(a, b):\n"
    '    """Return the sum of a and b."""\n'
    "    return a - b\n"
)


def test_verify_candidate_passes_correct_implementation():
    from scripts.stage14_run import verify_candidate

    v = verify_candidate(_GOOD_CANDIDATE, _ADD_REC)
    assert v["executed"] is True
    assert v["exec_failure"] is None
    assert v["doctests_total"] == 2
    assert v["doctest_failures"] == 0
    assert v["passed"] is True


def test_verify_candidate_rejects_syntax_broken_code():
    from scripts.stage14_run import verify_candidate

    v = verify_candidate(_BROKEN_CANDIDATE, _ADD_REC)
    assert v["executed"] is False
    assert v["exec_failure"] == "sandbox_exec_failed"
    assert v["passed"] is False


def test_verify_candidate_fails_wrong_algorithm_on_reference_doctests():
    from scripts.stage14_run import verify_candidate

    v = verify_candidate(_WRONG_ALG_CANDIDATE, _ADD_REC)
    assert v["executed"] is True
    assert v["doctests_total"] > 0
    assert v["doctest_failures"] > 0
    assert v["passed"] is False


def test_verify_candidate_requires_doctest_evidence():
    from scripts.stage14_run import verify_candidate

    rec_no_doctests = {"name": "ping", "code": "def ping():\n    pass\n"}
    v = verify_candidate("def ping():\n    pass\n", rec_no_doctests)
    assert v["executed"] is True
    assert v["doctests_total"] == 0
    assert v["passed"] is False


def test_verify_candidate_empty_generation_cannot_pass():
    from scripts.stage14_run import verify_candidate

    v = verify_candidate("", _ADD_REC)
    assert v["passed"] is False
    assert v["doctest_failures"] > 0


def test_verify_candidate_accepts_record_doctests_against_namespace():
    from scripts.stage14_run import verify_candidate

    compliant = (
        "def add_two(a, b):\n"
        "    return a + b + 0\n"
    )
    v = verify_candidate(compliant, _ADD_REC)
    assert v["executed"] is True
    assert v["doctests_total"] == 2
    assert v["doctest_failures"] == 0
    assert v["passed"] is True


def test_verify_candidate_reproduces_harvest_module_namespace():
    """Dataset records were harvested with doctests run against the real
    module namespace (whole module source executed); verification must
    reproduce that, so a record's own CPython-provenance solution passes
    its reference doctests (real record 'once', module functools)."""
    from cortex.training.sft import load_split
    from scripts.stage14_run import verify_candidate

    recs = load_split(str(DATASET), "train")
    rec = next(r for r in recs if r["name"] == "once")
    v = verify_candidate(rec["code"].strip(), rec)
    assert v["executed"] is True
    assert v["doctests_total"] == 10
    assert v["doctest_failures"] == 0
    assert v["passed"] is True

    mutated = rec["code"].strip().replace(
        "wrapper.saved_result = func(*args, **kwargs)",
        "wrapper.saved_result = 0")
    v2 = verify_candidate(mutated, rec)
    assert v2["passed"] is False and v2["doctest_failures"] > 0


def test_feedback_row_surface_form_and_target_integrity():
    from scripts.stage14_run import make_feedback_row

    evidence = "Traceback (most recent call last):\nZeroDivisionError: p"
    row = make_feedback_row(_ADD_REC, evidence)
    assert row["name"] == "feedback-add_two"
    assert row["instruction"].startswith("# Task: ")
    assert row["instruction"].startswith("# Task: write a function that "
                                         "adds two numbers\n# Previous "
                                         "attempt failed:\n# ")
    assert "ZeroDivisionError" in row["instruction"]
    assert row["code"] == _ADD_REC["code"]
    assert row["split"] == "train"
    assert row["provenance"]["kind"] == "stage14-execution-feedback"
    assert row["provenance"]["parent"] == _ADD_REC.get("provenance")


def test_feedback_row_evidence_is_capped():
    from scripts.stage14_run import make_feedback_row

    row = make_feedback_row(_ADD_REC, "x" * 5000)
    evidence_part = row["instruction"].split("# Previous attempt failed:\n",
                                             1)[1]
    core = evidence_part.removesuffix("\n").removeprefix("# ")
    assert len(core) <= 400


def test_train_records_contain_no_benchmark_instruction_strings():
    from cortex.evaluation.tasks import TASKS
    from cortex.evaluation.tasks_v3 import TASKS_V3
    from cortex.training.sft import load_split

    recs = load_split(str(DATASET), "train")
    bench = {t.instruction for t in TASKS} | {t.instruction
                                              for t in TASKS_V3}
    hits = [r["name"] for r in recs if r["instruction"] in bench]
    assert hits == []


def test_sample_train_records_is_deterministic_subset():
    from cortex.training.sft import load_split
    from scripts.stage14_run import _sample_train_records, FB_RECORDS, \
        FB_SEED

    recs = load_split(str(DATASET), "train")
    a = _sample_train_records(recs, FB_RECORDS, FB_SEED)
    b = _sample_train_records(recs, FB_RECORDS, FB_SEED)
    assert a == b and len(a) == FB_RECORDS
    assert all(r in recs for r in a)


def test_experiment_header14_records_correct_parent_commit():
    from scripts.stage14_run import _experiment_header14, PARENT_COMMIT

    h = _experiment_header14("test-exp", "hyp", "acc")
    assert h["parent_commit"] == PARENT_COMMIT == "c30ceb5"
    assert h["experiment_id"] == "test-exp"
    assert h["pre_registered_hypothesis"] == "hyp"
    assert h["acceptance_criteria"] == "acc"
    assert "dataset_sha256" in h and len(h["dataset_sha256"]) == 64


STAGE14_EXPERIMENT_ARTIFACTS = [
    "stage14_gpu_qualification.json",
    "stage14_feedback_training.json",
    "stage14_horizon_arms.json",
    "stage14_promotion_audit.json",
]

STAGE14_PLAIN_ARTIFACTS = [
    "stage14_report.json",
    "stage14_independent_audit.json",
]


@pytest.mark.parametrize("name", STAGE14_EXPERIMENT_ARTIFACTS)
def test_stage14_artifacts_owned_and_complete(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
    assert "experiment" in obj and "decision" in obj["experiment"]


@pytest.mark.parametrize("name", STAGE14_PLAIN_ARTIFACTS)
def test_stage14_plain_artifacts_owned(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
