"""Stage 15 tests: execution-failure classification from real evidence,
compacted-feedback row surface form and target-coverage integrity,
unseen-suite harness sanity (every reference solution really passes its
own tests), leakage control, and post-run artifact integrity. Real
components, real tokenizer, real sandbox execution, no mocks.
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


def _tok():
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(ARTIFACTS / "tokenizer" / "tokenizer.json"))


# ------------------------------------------------- failure classification

def test_classify_success():
    from scripts.stage15_run import classify_execution_failure

    assert classify_execution_failure({"passed": True}) == "success"


def test_classify_syntax_failure_from_real_sandbox_evidence():
    from scripts.stage14_run import verify_candidate
    from scripts.stage15_run import classify_execution_failure

    v = verify_candidate("def add_two(a, b):\n    return a + b\n  if:\n",
                         _ADD_REC)
    assert classify_execution_failure(v) == "syntax_failure"


def test_classify_failed_reference_doctest_from_real_evidence():
    from scripts.stage14_run import verify_candidate
    from scripts.stage15_run import classify_execution_failure

    v = verify_candidate("def add_two(a, b):\n    return a - b\n",
                         _ADD_REC)
    assert v["doctest_failures"] > 0
    assert classify_execution_failure(v) == "failed_reference_doctest"


def test_classify_import_failure_from_stderr_evidence():
    from scripts.stage15_run import classify_execution_failure

    v = {"passed": False, "executed": False,
         "exec_failure": "sandbox_exec_failed",
         "exec_stderr": "ModuleNotFoundError: No module named 'numpy'",
         "doctests_total": 0, "doctest_failures": 0}
    assert classify_execution_failure(v) == "import_failure"


def test_classify_timeout_from_failure_evidence():
    from scripts.stage15_run import classify_execution_failure

    v = {"passed": False, "executed": False,
         "exec_failure": "TimeoutExpired", "exec_stderr": "",
         "doctests_total": 0, "doctest_failures": 0}
    assert classify_execution_failure(v) == "timeout"


def test_classify_failed_hidden_test_from_pytest_evidence():
    from scripts.stage15_run import classify_execution_failure

    v = {"passed": False, "executed": True, "exec_failure": None,
         "exec_stderr": "E   AssertionError: assert 5 == 4\n"
                        "== FAILED ==",
         "doctests_total": 0, "doctest_failures": 0}
    assert classify_execution_failure(v) == "failed_hidden_test"


def test_classify_runtime_exception_from_stderr_evidence():
    from scripts.stage15_run import classify_execution_failure

    v = {"passed": False, "executed": True, "exec_failure": None,
         "exec_stderr": "Traceback ...\nZeroDivisionError: division by zero",
         "doctests_total": 0, "doctest_failures": 0}
    assert classify_execution_failure(v) == "runtime_exception"


def test_classify_incomplete_generation():
    from scripts.stage15_run import classify_execution_failure

    v = {"passed": False, "executed": False, "exec_failure": None,
         "exec_stderr": "", "doctests_total": 0, "doctest_failures": 0}
    assert classify_execution_failure(v) == "incomplete_generation"


# ------------------------------------------------------- evidence compaction

def test_compact_evidence_empty_for_success():
    from scripts.stage15_run import compact_evidence
    from scripts.stage14_run import verify_candidate

    v = verify_candidate(_ADD_REC["code"].strip(), _ADD_REC)
    assert compact_evidence(v) == ""


def test_compact_evidence_carries_class_and_last_error_line():
    from scripts.stage15_run import classify_execution_failure, \
        compact_evidence

    v = {"passed": False, "executed": True, "exec_failure": None,
         "exec_stderr": "Traceback (most recent call last):\n"
                        "  File \"candidate\", line 2\n"
                        "ZeroDivisionError: division by zero",
         "doctests_total": 0, "doctest_failures": 0}
    ev = compact_evidence(v)
    cls = classify_execution_failure(v)
    assert ev.startswith(f"{cls}: ")
    assert "ZeroDivisionError" in ev


def test_compact_evidence_error_line_capped_at_160():
    from scripts.stage15_run import COMPACT_EVIDENCE_CHARS, compact_evidence

    assert COMPACT_EVIDENCE_CHARS == 160
    long_line = "SyntaxError: " + "E" * 900
    v = {"passed": False, "executed": False,
         "exec_failure": "sandbox_exec_failed", "exec_stderr": long_line,
         "doctests_total": 0, "doctest_failures": 0}
    ev = compact_evidence(v)
    assert "syntax_failure" in ev
    assert len(ev) <= COMPACT_EVIDENCE_CHARS + 32


def test_compact_evidence_falls_back_to_doctest_evidence():
    from scripts.stage15_run import compact_evidence

    v = {"passed": False, "executed": True, "exec_failure": None,
         "exec_stderr": "", "doctests_total": 2, "doctest_failures": 2,
         "doctest_evidence":
             "Failed example:\n    add_two(1, 2)\nExpected:\n    3\n"
             "Got:\n    2\n"}
    ev = compact_evidence(v)
    assert ev.startswith("failed_reference_doctest: ")
    assert "add_two(1, 2)" in ev


# ------------------------------------------------------- feedback rows

def test_feedback_row15_surface_form_and_provenance():
    from scripts.stage15_run import make_feedback_row15, verdict_stderr

    evidence = "failed_reference_doctest: Failed example"
    row = make_feedback_row15(_ADD_REC, evidence)
    assert row["name"] == "fb15-add_two"
    assert row["instruction"].startswith("# Task: write a function that "
                                         "adds two numbers\n"
                                         "# Attempt failed with ")
    assert evidence in row["instruction"]
    assert row["code"] == _ADD_REC["code"]
    assert row["split"] == "train"
    assert row["provenance"]["kind"] == "stage15-compacted-execution-feedback"
    assert row["provenance"]["error_class"] == \
        "failed_reference_doctest"
    assert row["provenance"]["parent"] == _ADD_REC.get("provenance")
    assert "original_stderr_preserved_in_record" in row["provenance"]
    assert verdict_stderr("add_two") is None or isinstance(
        verdict_stderr("add_two"), str)


def test_verdict_stderr_round_trip():
    from scripts.stage15_run import _VERDICT_STDERR, verdict_stderr

    _VERDICT_STDERR["t15-unit"] = "boom"
    try:
        assert verdict_stderr("t15-unit") == "boom"
        assert verdict_stderr("never-recorded") is None
    finally:
        _VERDICT_STDERR.pop("t15-unit", None)


# ------------------------------------------------- training-target coverage

def test_feedback_coverage_exact_match_on_real_batch():
    from cortex.training.sft import make_batch
    from scripts.stage15_run import feedback_coverage, make_feedback_row15

    tok = _tok()
    short = dict(_ADD_REC)
    short["code"] = "def add_two(a, b):\n    return a + b\n" * 2
    row = make_feedback_row15(short, "syntax_failure: SyntaxError (line 2)")
    x, y = make_batch([row], tok, 256)
    assert x is not None
    cov = feedback_coverage(row, tok)
    assert cov["fits"] is True
    assert cov["exact_match"] is True
    assert cov["n_supervised"] == cov["len_completion"] > 0


def test_feedback_coverage_nonfitting_row_is_measured_not_assumed():
    from scripts.stage15_run import feedback_coverage, make_feedback_row15

    tok = _tok()
    big = dict(_ADD_REC)
    big["code"] = "x = 1\n" * 4000
    row = make_feedback_row15(big, "syntax_failure: SyntaxError (line 2)")
    from cortex.training.sft import make_batch
    assert make_batch([row], tok, 256) is None or make_batch(
        [row], tok, 256)[0] is None
    cov = feedback_coverage(row, tok)
    assert cov["fits"] is False
    assert cov["exact_match"] is False
    assert cov["n_supervised"] == 0


# ------------------------------------------------------- unseen suite

def test_fingerprint_unseen_deterministic_64hex():
    from scripts.stage15_run import fingerprint_unseen

    a = fingerprint_unseen()
    b = fingerprint_unseen()
    assert a == b and len(a) == 64
    int(a, 16)


def test_unseen_task_count_and_harness_sanity():
    from scripts.stage15_run import UNSEEN_TASKS_STAGE15 as tasks
    from scripts.stage15_run import _grade_unseen, _validate_unseen_harness

    assert len(tasks) == 8
    harness = _validate_unseen_harness()
    assert harness["all_references_pass"] is True
    for r in harness["per_task"]:
        assert r["reference_passes"] is True
    for t in tasks:
        g = _grade_unseen(t, t.reference_solution)
        assert g["passed"] is True


def test_unseen_tasks_are_not_in_train_or_benchmark_instructions():
    from cortex.evaluation.tasks import TASKS
    from cortex.evaluation.tasks_v3 import TASKS_V3
    from cortex.training.sft import load_split
    from scripts.stage15_run import UNSEEN_TASKS_STAGE15

    bench = {t.instruction for t in TASKS} | {t.instruction for t in TASKS_V3}
    train_instr = {r["instruction"] for r in load_split(str(DATASET),
                                                        "train")}
    unseen = {t.instruction for t in UNSEEN_TASKS_STAGE15}
    assert not (unseen & bench)
    assert not (unseen & train_instr)
    ids = {t.task_id for t in UNSEEN_TASKS_STAGE15}
    bench_ids = {t.task_id for t in TASKS} | {t.task_id
                                              for t in TASKS_V3}
    assert not (ids & bench_ids)


# ------------------------------------------------- registered constants

def test_stage15_registered_constants():
    import scripts.stage15_run as m

    assert m.PARENT_COMMIT == "318e601"
    assert m.G_HELDOUT_LIMIT == 3.3015
    assert m.STAGE15_REFERENCE_ARM == "stage14-structural_h6000"
    assert m.STAGE14_CHAMPION_REF["heldout_eval_aligned"] == 2.9124
    assert m.STAGE14_CHAMPION_REF[
        "feedback_target_bearing_steps"] == 102
    assert m.DECODING15["temperature"] == 0.2
    assert m.STAGE14_REP1["control_heldout_eval_aligned"] == 2.9250
    assert m.STAGE14_REP1["structural_heldout_eval_aligned"] == 2.9124


def test_experiment_header15_records_correct_parent_commit():
    from scripts.stage15_run import PARENT_COMMIT, _experiment_header15

    h = _experiment_header15("test-exp", "hyp", "acc")
    assert h["parent_commit"] == PARENT_COMMIT == "318e601"
    assert h["experiment_id"] == "test-exp"
    assert h["pre_registered_hypothesis"] == "hyp"
    assert h["acceptance_criteria"] == "acc"


# ------------------------------------------------- artifact integrity

STAGE15_EXPERIMENT_ARTIFACTS = [
    "stage15_reproducibility.json",
    "stage15_structural_consolidation.json",
    "stage15_feedback_training.json",
    "stage15_execution_data.json",
    "stage15_generalization.json",
    "stage15_self_correction.json",
    "stage15_optimization_audit.json",
    "stage15_capacity_audit.json",
    "stage15_subsystem_contribution.json",
    "stage15_promotion_audit.json",
]

STAGE15_PLAIN_ARTIFACTS = [
    "stage15_report.json",
    "stage15_failure_taxonomy.json",
    "stage15_independent_audit.json",
]


@pytest.mark.parametrize("name", STAGE15_EXPERIMENT_ARTIFACTS)
def test_stage15_artifacts_owned_and_complete(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
    assert "experiment" in obj and "decision" in obj["experiment"]


@pytest.mark.parametrize("name", STAGE15_PLAIN_ARTIFACTS)
def test_stage15_plain_artifacts_owned(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
