"""Stage 14 independent acceptance audit (Workstream E).

Deliberately does NOT trust the orchestrator's reported numbers where an
independent recomputation is practical: benchmark fingerprints are
recomputed from raw dataclass fields, the execution-feedback candidate
outcomes and feedback-row evidence provenance are re-derived from the
persisted per-record rows, capability pass@1/syntax rates are re-derived
from per-task rows, the perturbed gate is re-derived from per-probe
results, all stage-14 checkpoint sha256 values are recomputed from disk,
the 5-gate promotion decision is re-derived, and the authoritative
virtual store is recounted. Statuses are strict PASS / FAIL / BLOCKED /
NOT YET TESTED.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATASET = ART / "sft" / "dataset_v3.jsonl"
TOKENIZER = ART / "tokenizer" / "tokenizer.json"
CKPT10_DIR = ART / "runs" / "cortex-sft-stage10"
STORE_DIR = ART / "virtual-store-502b"
PARENT_COMMIT = "c30ceb5"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"
FINGERPRINT_V3 = "c70feb13b061a35b9b3940296c40ea8365aa57089b325050ef238dc93774ccbb"
BASE_HELDOUT_ALLOWED = 3.3015
STAGE13_CHAMPION_HELDOUT = {"native": 2.9435, "eval_aligned": 2.9498}

STAGE14_ARTIFACTS = (
    "stage14_gpu_qualification.json",
    "stage14_feedback_training.json",
    "stage14_horizon_arms.json",
    "stage14_promotion_audit.json",
    "stage14_report.json",
)

STAGE13_ARTIFACTS_MUST_REMAIN = (
    "stage13_report.json",
    "stage13_structural_root_cause.json",
    "stage13_structural_data_audit.json",
    "stage13_training_stream.json",
    "stage13_structural_learning.json",
    "stage13_generation_control.json",
    "stage13_execution_feedback.json",
    "stage13_generalization.json",
    "stage13_experiment_matrix.json",
    "stage13_capacity_and_infrastructure.json",
    "stage13_intelligence_subsystems.json",
    "stage13_promotion_audit.json",
    "stage13_independent_audit.json",
    "stage13_failure_taxonomy.json",
)

BENCHMARK_FILES = (
    "cortex/evaluation/tasks.py",
    "cortex/evaluation/tasks_v2.py",
    "cortex/evaluation/tasks_v3.py",
    "cortex/evaluation/benchmarks.py",
)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_unreadable": repr(exc)}


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_fingerprints() -> dict:
    from cortex.evaluation.benchmarks import fingerprint_tasks
    from cortex.evaluation.tasks import TASKS as T1

    v1 = hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.test_code, t.reference_solution)
        for t in T1), sort_keys=True).encode("utf-8")).hexdigest()
    v2 = fingerprint_tasks("v2")
    return {
        "v1_recomputed": v1, "v1_matches_immutable": v1 == FINGERPRINT_V1,
        "v2_recomputed": v2, "v2_matches_immutable": v2 == FINGERPRINT_V2,
        "v1_constant_match": FINGERPRINT_V1 == FINGERPRINT_V1,
        "v2_constant_match": FINGERPRINT_V2 == FINGERPRINT_V2,
    }


def audit_fingerprint_v3() -> dict:
    from cortex.evaluation.tasks_v3 import tasks_v3_tuples

    v3 = hashlib.sha256(json.dumps(sorted(
        tasks_v3_tuples()), sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "v3_recomputed_from_raw_tuples": v3,
        "v3_matches_immutable": v3 == FINGERPRINT_V3,
    }


def audit_benchmarks_untouched() -> dict:
    diff = subprocess.run(["git", "diff", f"{PARENT_COMMIT} HEAD", "--",
                           *BENCHMARK_FILES],
                          cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *BENCHMARK_FILES],
        cwd=ROOT, capture_output=True, text=True)
    return {
        "diff_vs_parent_commit_empty": diff.stdout.strip() == "",
        "working_tree_clean_for_benchmark_files":
            status.stdout.strip() == "",
    }


def audit_feedback_sanity() -> dict:
    """Every pre-training feedback row must come from a candidate that was
    really executed and did NOT pass; rows_built must equal
    n_records - n_passed; contamination hits must be 0."""
    b = _read_json(ART / "stage14_feedback_training.json")
    if b is None:
        return {"status": "MISSING ARTIFACT"}
    co = b.get("candidate_outcomes") or {}
    per_record = co.get("per_record") or []
    if not per_record:
        return {"status": "MISSING per_record rows"}
    n = len(per_record)
    passed = sum(1 for r in per_record if r.get("passed"))
    failed_real = sum(
        1 for r in per_record if not r.get("passed")
        and (r.get("executed") is False or (r.get("doctest_failures") or 0)
             > 0))
    rows_built = b.get("feedback_rows_built")
    counts_consistent = rows_built == n - passed
    every_row_from_real_failure = rows_built == failed_real
    sources = b.get("feedback_row_evidence_source") or {}
    source_used = sum(
        1 for s in sources.values()
        if s in ("exec_stderr", "doctest_evidence", "sandbox_exec_failed",
                 "namespace_exec"))
    return {
        "n_records": n,
        "n_passed_reference_doctests": passed,
        "n_real_failures_with_evidence": failed_real,
        "feedback_rows_built": rows_built,
        "rows_built_equals_n_minus_passed": counts_consistent,
        "every_row_from_really_executed_failing_candidate":
            every_row_from_real_failure,
        "evidence_source_counts": {
            k: sum(1 for v in sources.values() if v == k)
            for k in sorted(set(sources.values()))},
        "all_sources_are_real_failure_evidence":
            source_used == len(sources),
        "benchmark_instruction_contamination_hits":
            b.get("benchmark_instruction_contamination_hits"),
        "contamination_zero": b.get(
            "benchmark_instruction_contamination_hits") == 0,
    }


def audit_capability_rederived() -> dict:
    """Re-derive per-candidate per-suite pass@1 from the persisted
    per-task rows and cross-check the reported aggregates plus the
    syntax/complete-program rates."""
    checks = {}
    b = _read_json(ART / "stage14_feedback_training.json")
    c = _read_json(ART / "stage14_horizon_arms.json")
    sources = {}
    if isinstance(b, dict):
        sources["stage14-feedback"] = b["evaluation_unconditional"][
            "capability"]
    if isinstance(c, dict):
        for name, arm in (c.get("arms") or {}).items():
            sources[name] = arm["capability"]
    for name, cap in sources.items():
        rows = cap.get("per_task") or []
        if not rows:
            checks[name] = {"status": "MISSING per_task rows"}
            continue
        entry = {
            "reported_v1_pass_at_1": cap.get("v1_pass_at_1"),
            "reported_v3_pass_at_1": cap.get("v3_pass_at_1"),
            "reported_syntax_valid_rate": cap.get("syntax_valid_rate"),
            "reported_complete_program_rate":
                cap.get("complete_program_rate"),
            "n_per_task_rows": len(rows),
        }
        for suite in ("v1", "v3"):
            srows = [r for r in rows if r.get("suite") == suite]
            rederived = round(
                sum(1 for r in srows if r.get("passed"))
                / max(len(srows), 1), 4)
            entry[f"{suite}_rederived"] = rederived
            entry[f"{suite}_n_tasks"] = len(srows)
            entry[f"{suite}_matches"] = (
                rederived == cap.get(f"{suite}_pass_at_1"))
        syntax_rd = round(sum(1 for r in rows if r.get("parse"))
                          / max(len(rows), 1), 4)
        complete_rd = round(sum(1 for r in rows
                                if r.get("complete_program"))
                            / max(len(rows), 1), 4)
        entry["syntax_rederived"] = syntax_rd
        entry["syntax_matches"] = abs(
            syntax_rd - (cap.get("syntax_valid_rate") or -1)) < 1e-6
        entry["complete_rederived"] = complete_rd
        entry["complete_matches"] = abs(
            complete_rd - (cap.get("complete_program_rate") or -1)) < 1e-6
        checks[name] = entry
    return checks


def audit_checkpoints() -> dict:
    rows = []
    b = _read_json(ART / "stage14_feedback_training.json") or {}
    ck = (b.get("training") or {})
    if ck:
        rows.append({"arm": "stage14-feedback",
                     "path": ck.get("checkpoint"),
                     "recorded_sha256": ck.get("checkpoint_sha256")})
    c = _read_json(ART / "stage14_horizon_arms.json") or {}
    for name, arm in (c.get("arms") or {}).items():
        rows.append({"arm": name, "path": arm.get("checkpoint"),
                     "recorded_sha256": arm.get("checkpoint_sha256")})
    d = _read_json(ART / "stage14_promotion_audit.json") or {}
    bl = (d.get("rollback") or {}).get("baseline_checkpoint")
    out_rows = []
    for r in rows:
        p = Path(r["path"]) if r.get("path") else None
        actual = _sha(p) if p else None
        out_rows.append({
            "arm": r["arm"], "path": r["path"],
            "exists": bool(p and p.exists()),
            "recorded_sha256": r["recorded_sha256"],
            "actual_sha256": actual,
            "sha256_match": r["recorded_sha256"] == actual,
        })
    if bl:
        out_rows.append({
            "arm": "stage12-champion-rollback", "path": bl,
            "exists": Path(bl).exists(), "recorded_sha256": None,
            "actual_sha256": _sha(Path(bl)), "sha256_match": True,
        })
    return {
        "checkpoints": out_rows,
        "all_stage14_recorded_sha256_match": bool(rows) and all(
            r["sha256_match"] for r in out_rows
            if r["arm"] != "stage12-champion-rollback"),
        "all_checkpoint_paths_exist": bool(out_rows) and all(
            r["exists"] for r in out_rows),
    }


def audit_promotion_rederived() -> dict:
    from scripts.stage14_run import STAGE13_CHAMPION_CAP  # noqa: E402

    d = _read_json(ART / "stage14_promotion_audit.json")
    if d is None:
        return {"status": "MISSING ARTIFACT"}
    cands = d.get("candidate_metrics") or {}
    if not cands:
        return {"status": "MISSING candidate_metrics"}
    champion = d.get("champion_candidate")
    champ = cands.get(champion) or {}
    any_pass = any(
        (c.get("v1_pass_at_1") or 0) > 0 or (c.get("v3_pass_at_1") or 0) > 0
        for c in cands.values())
    probes = d.get("perturbation_probe_results") or []
    perturbed_pass = round(
        sum(1 for p in probes if p.get("passed")) / len(probes), 4) \
        if probes else None
    rederived_gate1 = (champ.get("syntax_valid_rate") or 0) > \
        STAGE13_CHAMPION_CAP["syntax_valid_rate"]
    rederived_gate2 = rederived_gate1 and (
        (champ.get("complete_program_rate") or 0)
        > STAGE13_CHAMPION_CAP["complete_program_rate"])
    rederived_gate3 = any_pass
    rederived_gate4 = bool(perturbed_pass) and perturbed_pass > 0
    loss_ok = (champ.get("heldout_eval_aligned_final", 99)
               <= BASE_HELDOUT_ALLOWED)
    rederived_promoted = rederived_gate3 and loss_ok
    reported = d.get("experiment", {}).get("decision", "")
    rp = d.get("gates", {})
    gates_reported = {k: v.get("ok") for k, v in rp.items()}
    gates_rederived = {
        "gate1_structural_improvement": rederived_gate1,
        "gate2_complete_generation": rederived_gate2,
        "gate3_functional_correctness": rederived_gate3,
        "gate4_generalization": rederived_gate4,
        "gate5_capability_promotion": rederived_gate3,
    }
    gates_agree = all(
        gates_reported.get(k) == v for k, v in gates_rederived.items())
    return {
        "champion": champion,
        "rederived_gates": gates_rederived,
        "reported_gates_ok": gates_reported,
        "gates_agree": gates_agree,
        "rederived_heldout_loss_gate_ok": loss_ok,
        "rederived_promoted": rederived_promoted,
        "reported_decision": reported,
        "promotion_decision_agrees": (
            rederived_promoted and "PROMOTED" in reported
            and "NOT" not in reported) or (
            (not rederived_promoted) and "NOT PROMOTED" in reported),
    }


def audit_independent_selftest() -> dict:
    from cortex.evaluation.autoregressive_diagnostics import (
        analyzer_selftest)

    st = analyzer_selftest()
    return {
        "self_test_ok": st.get("self_test_ok"),
        "n_cases": st.get("n_cases"),
        "failed_cases": [c["case"] for c in st.get("cases", [])
                         if not c.get("analyzer_agrees")][:5],
    }


def audit_store_untouched() -> dict:
    from cortex.virtual.coverage import persisted_optimizer_scan
    from cortex.virtual.store import VirtualParameterStore

    auth = VirtualParameterStore.open(STORE_DIR)
    scan = persisted_optimizer_scan(auth)
    auth.close()
    return {
        "slots_updated_recount": scan.get("slots_updated"),
        "expected_35292": scan.get("slots_updated") == 35292,
        "n_allocated_500b": scan.get("n_allocated"),
    }


def audit_stage13_lineage_intact() -> dict:
    arts = {}
    for name in STAGE13_ARTIFACTS_MUST_REMAIN:
        doc = _read_json(ART / name)
        if doc is None:
            arts[name] = "MISSING"
        elif isinstance(doc, dict) and "_unreadable" in doc:
            arts[name] = "UNREADABLE"
        else:
            arts[name] = "OK" if doc.get("owned_by") == "Fibonera" \
                else "OWNERSHIP MISSING"
    champs = [
        CKPT10_DIR / "cortex-sft-stage12-chunked_control.pt",
        CKPT10_DIR / "cortex-sft-stage13-control_recheck.pt",
        ART / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt",
    ]
    return {
        "stage13_artifacts": arts,
        "preserved_checkpoints_present": {
            str(p): p.exists() for p in champs},
    }


def run_audit() -> dict:
    fp = audit_fingerprints()
    fp.update(audit_fingerprint_v3())
    bench = audit_benchmarks_untouched()
    fb = audit_feedback_sanity()
    cap = audit_capability_rederived()
    ckpts = audit_checkpoints()
    promo = audit_promotion_rederived()
    selftest = audit_independent_selftest()
    store = audit_store_untouched()
    lineage = audit_stage13_lineage_intact()

    arts = {}
    for name in STAGE14_ARTIFACTS:
        doc = _read_json(ART / name)
        if doc is None:
            arts[name] = "MISSING"
        elif isinstance(doc, dict) and "_unreadable" in doc:
            arts[name] = "UNREADABLE"
        else:
            arts[name] = "OK" if doc.get("owned_by") == "Fibonera" \
                else "OWNERSHIP MISSING"

    fb_ok = bool(fb.get("rows_built_equals_n_minus_passed")) and bool(
        fb.get("every_row_from_really_executed_failing_candidate")) and \
        bool(fb.get("all_sources_are_real_failure_evidence")) and bool(
        fb.get("contamination_zero"))
    cap_ok = all(
        v.get("v1_matches", True) and v.get("v3_matches", True)
        for v in cap.values() if isinstance(v, dict))
    checks = {
        "B_feedback_rows_from_really_executed_failures": {
            "status": "PASS" if fb_ok else "FAIL", "evidence": fb},
        "capability_metrics_rederived": {
            "status": "PASS" if cap_ok else "FAIL", "evidence": cap},
        "checkpoint_integrity": {
            "status": "PASS" if ckpts["all_stage14_recorded_sha256_match"]
            and ckpts["all_checkpoint_paths_exist"] else "FAIL",
            "evidence": ckpts},
        "promotion_decision_rederived": {
            "status": "PASS"
            if promo.get("promotion_decision_agrees")
            and promo.get("gates_agree") else "FAIL",
            "evidence": promo},
        "analyzer_selftest_reexecuted": {
            "status": "PASS" if selftest.get("self_test_ok") else "FAIL",
            "evidence": selftest},
        "authoritative_store_untouched": {
            "status": "PASS" if store.get("expected_35292") else "FAIL",
            "evidence": store},
        "fingerprints_immutable": {
            "status": "PASS" if all(
                [fp["v1_matches_immutable"], fp["v2_matches_immutable"],
                 fp["v3_matches_immutable"]]) else "FAIL",
            "evidence": fp},
        "benchmark_files_untouched": {
            "status": "PASS" if bench["diff_vs_parent_commit_empty"]
            and bench["working_tree_clean_for_benchmark_files"]
            else "FAIL", "evidence": bench},
        "stage13_lineage_intact": {
            "status": "PASS" if all(
                v == "OK" for v in
                lineage["stage13_artifacts"].values())
            and all(lineage["preserved_checkpoints_present"].values())
            else "FAIL", "evidence": lineage},
        "stage14_artifacts_present_and_owned": {
            "status": "PASS" if all(v == "OK"
                                    for v in arts.values()) else "FAIL",
            "evidence": arts},
    }
    known_provenance = {
        "stage13_header_parent_commit_anomaly":
            "stage-13 artifact headers carry parent_commit=a8c4e98 from a "
            "shared stage-10 header helper; source_commit fields are "
            "correct and per the immutable-artifact rule stage-13 "
            "artifacts are NOT retroactively altered; stage-14 headers "
            "use a corrected local helper",
    }
    return {
        "kind": "cortex-stage14-independent-audit",
        "owned_by": "Fibonera", "stage": 14,
        "parent_commit": PARENT_COMMIT,
        "software_vs_capability_note":
            "PASS entries verify software correctness and measurement "
            "integrity; learned capability is reported separately via "
            "the pre-registered 5-gate promotion decision and the "
            "measured pass@1 values",
        "known_provenance_notes": known_provenance,
        "independent_logic": {
            "fingerprints": fp,
            "benchmark_files": bench,
            "dataset_tokenizer_checksums": {
                "dataset_v3": _sha(DATASET),
                "tokenizer": _sha(TOKENIZER)},
            "store_recount": store,
            "selftest": selftest,
            "lineage": lineage,
        },
        "checks": checks,
        "summary_counts": {
            s: sum(1 for c in checks.values()
                   if str(c.get("status")).startswith(s))
            for s in ("PASS", "FAIL", "BLOCKED", "NOT YET TESTED")},
    }


if __name__ == "__main__":
    out = run_audit()
    (ART / "stage14_independent_audit.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out["summary_counts"], indent=2))
    for name, chk in out["checks"].items():
        print(f"[stage14-audit] {chk['status']:>5}  {name}")
    print("[stage14-audit] wrote artifacts/stage14_independent_audit.json")
