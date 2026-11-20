"""Stage 16 tests: pre-registered constants and gate thresholds,
autoregressive-exposure alignment invariants, t16 unseen-suite harness
sanity (every reference solution really passes its own pytest tests),
contamination control, first-transition divergence helper, and post-run
artifact integrity. Real components, real tokenizer, real sandbox
execution, no mocks.
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


def _train_recs():
    from cortex.training.sft import load_split

    return load_split(str(DATASET), "train")


# ------------------------------------------------- registered constants

def test_parent_commit_is_stage15_head():
    from scripts.stage16_run import PARENT_COMMIT

    assert PARENT_COMMIT == "799a217"


def test_stage16_registered_constants_and_gate_thresholds():
    import scripts.stage16_run as m

    assert m.DECODING16 == {"temperature": 0.2, "top_k": 50,
                            "max_new_tokens": 180}
    assert m.PREFIX_DECODING16["max_new_tokens"] == 48
    assert m.EXPOSURE_PREFIX_TOKENS == 48
    assert m.EXPOSURE_ATTEMPT_RECORDS == 96
    assert m.EXPOSURE_GEN_SEED == 20261810
    assert m.STAGE16_PRIMARY_ARM == "exposure_sched"
    assert m.GATE_B_PARSE_DELTA == 0.25
    assert m.G_HELDOUT_LIMIT == 3.3015
    assert m.REPAIR_ROUNDS == 2
    assert m.S15_FEEDBACK_HELDOUT_REF == 3.5957
    assert m.FB16_RECORDS == 12
    assert m.FB16_STEPS == 600
    assert m.FB16_SEED == 20261821
    assert m.GEN_SEED16 == 20261830
    assert m.PERTURB_SEED_BASE16 == 20261900
    assert len(m.ARMS_SEEDS) == 7
    assert m.STAGE16_PRIMARY_ARM in m.ARMS_SEEDS
    assert m.ARMS_SEEDS["exposure_sched_rep2"] != m.ARMS_SEEDS["exposure_sched"]


def test_experiment_header16_records_correct_parent_commit():
    from scripts.stage16_run import PARENT_COMMIT, _experiment_header16

    h = _experiment_header16("stage16-test-exp", "hyp", "acc")
    assert h["parent_commit"] == PARENT_COMMIT == "799a217"
    assert h["experiment_id"] == "stage16-test-exp"
    assert h["pre_registered_hypothesis"] == "hyp"
    assert h["acceptance_criteria"] == "acc"


# --------------------------------------------- exposure rate schedules

def test_exposure_rate_schedules():
    from scripts.stage16_run import _exposure_rate

    assert _exposure_rate("zero", 0, 3000) == 0.0
    assert _exposure_rate("zero", 2999, 3000) == 0.0
    assert _exposure_rate("const", 0, 3000) == _exposure_rate("const", 2999, 3000)
    assert _exposure_rate("const", 0, 3000) == 0.5
    assert _exposure_rate("sched", 0, 3000) == pytest.approx(0.25)
    assert _exposure_rate("sched", 1500, 3000) == pytest.approx(0.5)
    assert _exposure_rate("sched", 3000, 3000) == pytest.approx(0.75)
    rates = [_exposure_rate("sched", s, 3000) for s in range(0, 3001, 250)]
    assert rates == sorted(rates)
    with pytest.raises(ValueError):
        _exposure_rate("unknown", 0, 3000)


# --------------------------------- exposure row alignment invariants

def test_rows_batch_padding_and_label_shift_with_real_tokenizer():
    from scripts.stage16_run import _rows_batch

    tok = _tok()
    prompt_ids = tok.encode(_ADD_REC["instruction"] + "\n")
    comp_ids = tok.encode(_ADD_REC["code"].strip() + "\n" + tok.EOT)
    n_pre = 3
    row = {
        "rec_name": _ADD_REC["name"],
        "ids": prompt_ids + comp_ids,
        "labels": [-100] * (len(prompt_ids) + n_pre) + comp_ids[n_pre:],
        "n_prefix_tokens": n_pre,
    }
    seq_len = 256
    x, y = _rows_batch([row, row], tok, seq_len)
    assert x.shape == (2, seq_len - 1)
    assert y.shape == (2, seq_len - 1)
    lab0 = y[0].tolist()
    supervised = [i for i, v in enumerate(lab0) if v != -100]
    assert supervised
    first = supervised[0]
    assert first == len(prompt_ids) + n_pre - 1
    assert x[0, first].item() == comp_ids[n_pre - 1]
    assert y[0, first].item() == comp_ids[n_pre]
    assert x[0, first - 1].item() == comp_ids[n_pre - 2]
    lab1 = y[1].tolist()
    pad_start = len(prompt_ids) + len(comp_ids)
    assert all(v == -100 for v in lab1[max(pad_start - 1, 0):])


def test_exposure_artifact_rows_align_to_real_dataset():
    path = ARTIFACTS / "stage16_autoregressive_exposure.json"
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    from cortex.training.longcontext import record_ids

    tok = _tok()
    obj = _json.loads(path.read_text())
    exposure = obj.get("exposure_dataset", obj)
    aligned = exposure.get("aligned_rows", [])
    early = exposure.get("early_rows", [])
    n_attempted = exposure["attempted"]
    skips = exposure.get("skip_reasons", {})
    assert n_attempted == 96
    assert n_attempted == (len(aligned) + sum(skips.values()))
    assert len(early) == skips.get(
        "prefix_diverged_truncated_to_early_row", 0)
    rows = aligned + early
    assert rows, "exposure rows must exist in artifact"
    by_name: dict = {}
    for r in _train_recs():
        by_name.setdefault(r["name"], []).append(r)
    for row in rows[:8]:
        cands = by_name[row["rec_name"]]
        matched = None
        for rec in cands:
            prompt, comp = record_ids(rec, tok, "legacy")
            if list(row["ids"]) == list(prompt) + list(comp):
                matched = (rec, prompt, comp)
                break
        assert matched, row["rec_name"]
        rec, prompt, comp = matched
        lab = row["labels"]
        n_pre = row["n_prefix_tokens"]
        assert n_pre >= 1
        assert len(lab) == len(row["ids"])
        assert lab[: len(prompt) + n_pre] == [-100] * (len(prompt) + n_pre)
        assert lab[len(prompt) + n_pre:] == comp[n_pre:]


# ------------------------------------- training-fragment overlap helper

def test_max_training_prefix_overlap_detects_exact_reproduction():
    from scripts.stage16_run import _max_training_prefix_overlap

    tok = _tok()
    rec = _train_recs()[0]
    comp = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
    overlap = _max_training_prefix_overlap(comp, [rec], tok)
    assert overlap["max_prefix_overlap_ratio"] == 1.0
    assert overlap["best_matching_record"] == rec["name"]
    overlap_gen = _max_training_prefix_overlap([999999] + comp[1:], [rec], tok)
    assert overlap_gen["max_prefix_overlap_ratio"] == 0.0


# --------------------------------------------------- t16 unseen suite

def test_fingerprint_unseen16_deterministic_64hex():
    from scripts.stage16_run import fingerprint_unseen16

    a = fingerprint_unseen16()
    b = fingerprint_unseen16()
    assert a == b and len(a) == 64
    int(a, 16)


def test_stage16_unseen_task_count_and_harness_sanity():
    from scripts.stage16_run import UNSEEN_TASKS_STAGE16 as tasks
    from scripts.stage16_run import _grade_unseen16, _validate_unseen_harness16

    assert len(tasks) == 8
    harness = _validate_unseen_harness16()
    assert harness["all_references_pass"] is True
    for r in harness["per_task"]:
        assert r["reference_passes"] is True
    for t in tasks:
        g = _grade_unseen16(t, t.reference_solution)
        assert g["passed"] is True
        assert g["executed"] is True


def test_stage16_unseen_tasks_reject_broken_solutions():
    from scripts.stage16_run import UNSEEN_TASKS_STAGE16 as tasks
    from scripts.stage16_run import _grade_unseen16

    for t in tasks:
        g = _grade_unseen16(t, t.reference_solution
                            + "\nthis is not valid python\n")
        assert g["passed"] is False
        broken = t.reference_solution.replace("return", "return_", 1)
        if broken != t.reference_solution:
            g2 = _grade_unseen16(t, broken)
            assert g2["passed"] is False


def test_stage16_unseen_tasks_not_in_train_or_benchmark_instructions():
    from cortex.evaluation.tasks import TASKS
    from cortex.evaluation.tasks_v3 import TASKS_V3
    from scripts.stage16_run import UNSEEN_TASKS_STAGE15, UNSEEN_TASKS_STAGE16

    unseen = {t.instruction for t in UNSEEN_TASKS_STAGE16}
    bench = {t.instruction for t in TASKS} | {t.instruction
                                              for t in TASKS_V3}
    train_instr = {r["instruction"] for r in _train_recs()}
    t15 = {t.instruction for t in UNSEEN_TASKS_STAGE15}
    assert not (unseen & bench)
    assert not (unseen & train_instr)
    assert not (unseen & t15)
    ids = {t.task_id for t in UNSEEN_TASKS_STAGE16}
    bench_ids = {t.task_id for t in TASKS} | {t.task_id for t in TASKS_V3}
    assert not (ids & bench_ids)


def test_stage16_contamination_counts_are_zero():
    from scripts.stage16_run import UNSEEN_TASKS_STAGE16, _contamination16

    cont = _contamination16(UNSEEN_TASKS_STAGE16, _train_recs())
    assert cont["n_tasks"] == 8
    assert cont["instruction_in_train"] == 0
    assert cont["instruction_in_benchmarks"] == 0
    assert cont["instruction_in_stage15_unseen"] == 0


# ------------------------------------------------- stage-15 references

def test_stage15_reference_constants_unchanged():
    import scripts.stage15_run as s15
    import scripts.stage16_run as m

    assert m.STAGE15_REFERENCE_ARM == s15.STAGE15_REFERENCE_ARM
    assert m.G_HELDOUT_LIMIT == s15.G_HELDOUT_LIMIT
    assert m.S15_FEEDBACK_HELDOUT_REF == 3.5957
    assert s15.STAGE14_CHAMPION_REF["heldout_eval_aligned"] == 2.9124


# ------------------------------------------------- artifact integrity

STAGE16_EXPERIMENT_ARTIFACTS = [
    "stage16_baseline.json",
    "stage16_autoregressive_exposure.json",
    "stage16_structural_learning.json",
    "stage16_complete_program_training.json",
    "stage16_objective_ablation.json",
    "stage16_execution_feedback.json",
    "stage16_self_correction.json",
    "stage16_generalization.json",
    "stage16_moe_divergent.json",
    "stage16_capacity_audit.json",
    "stage16_promotion.json",
]

STAGE16_PLAIN_ARTIFACTS = [
    "stage16_report.json",
    "stage16_independent_audit.json",
]


@pytest.mark.parametrize("name", STAGE16_EXPERIMENT_ARTIFACTS)
def test_stage16_artifacts_owned_and_registered(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
    exp = obj["experiment"]
    assert exp["parent_commit"] == "799a217"
    assert exp["pre_registered_hypothesis"]
    assert exp["acceptance_criteria"]


@pytest.mark.parametrize("name", STAGE16_PLAIN_ARTIFACTS)
def test_stage16_plain_artifacts_owned(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"


def test_promotion_artifact_derives_audit_gate():
    path = ARTIFACTS / "stage16_promotion.json"
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    gates = obj["gates"]
    expected = {f"gate{g}_{n}" for g, n in (
        ("A", "reproducibility"), ("B", "free_running_structure"),
        ("C", "complete_programs"), ("D", "functional_correctness"),
        ("E", "generalization"), ("F", "self_correction"),
        ("G", "regression_control"), ("H", "independent_audit"))}
    assert set(gates) == expected
    audit_path = ARTIFACTS / "stage16_independent_audit.json"
    if audit_path.exists():
        audit = _json.loads(audit_path.read_text())
        gate_h = gates["gateH_independent_audit"]
        assert (audit["all_pass"] is True) == bool(gate_h["ok"])
    else:
        assert gates["gateH_independent_audit"]["ok"] is None


def test_independent_audit_artifact_all_pass_when_present():
    path = ARTIFACTS / "stage16_independent_audit.json"
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj["all_pass"] is True
    assert obj["summary"].endswith("/ 0 FAIL") or obj["summary"].endswith(
        "0 FAIL")
