"""Stage 15 independent acceptance audit (Workstream K).

Deliberately does NOT trust the orchestrator's reported numbers where an
independent recomputation is practical: benchmark fingerprints are
recomputed from raw dataclass fields, the unseen-suite fingerprint is
recomputed from raw tuples, the reproducibility decision is re-derived
from persisted held-out curves, feedback coverage is re-derived from the
persisted per-row rows, capability pass@1/syntax/complete rates are
re-derived from per-task rows, all stage-15 checkpoint sha256 values are
recomputed from disk, stage-14 checkpoints are re-verified against their
recorded hashes, the 6-gate promotion decision is re-derived, and the
authoritative virtual store is recounted. Statuses are strict PASS /
FAIL / BLOCKED / NOT YET TESTED.
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
PARENT_COMMIT = "318e601"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"
FINGERPRINT_V3 = "c70feb13b061a35b9b3940296c40ea8365aa57089b325050ef238dc93774ccbb"
G_HELDOUT_LIMIT = 3.3015

STAGE14_CHAMPION = {
    "heldout_eval_aligned": 2.9124,
    "feedback_target_bearing_steps": 102,
}

STAGE15_EXPERIMENT_ARTIFACTS = (
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
)

STAGE15_PLAIN_ARTIFACTS = (
    "stage15_report.json",
    "stage15_failure_taxonomy.json",
)

STAGE_10_TO_14_ARTIFACTS_MUST_REMAIN = (
    "stage10_report.json",
    "stage11_report.json",
    "stage12_report.json",
    "stage13_report.json",
    "stage14_report.json",
    "stage14_independent_audit.json",
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
    from cortex.evaluation.tasks_v3 import tasks_v3_tuples

    v1 = hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.test_code, t.reference_solution)
        for t in T1), sort_keys=True).encode("utf-8")).hexdigest()
    v3 = hashlib.sha256(json.dumps(sorted(
        tasks_v3_tuples()), sort_keys=True).encode("utf-8")).hexdigest()
    v2 = FINGERPRINT_V2 == fingerprint_tasks("v2")
    return {
        "v1_recomputed": v1, "v1_matches_immutable": v1 == FINGERPRINT_V1,
        "v2_matches_immutable": bool(v2),
        "v3_recomputed": v3, "v3_matches_immutable": v3 == FINGERPRINT_V3,
    }


def audit_unseen_fingerprint() -> dict:
    from scripts.stage15_run import UNSEEN_TASKS_STAGE15
    from scripts.stage15_run import fingerprint_unseen

    local = hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.module_name, t.test_code,
         t.reference_solution) for t in UNSEEN_TASKS_STAGE15),
        sort_keys=True).encode("utf-8")).hexdigest()
    mod = fingerprint_unseen()
    e = _read_json(ART / "stage15_generalization.json") or {}
    recorded = e.get("unseen_suite_fingerprint")
    return {
        "recomputed_from_raw_tuples": local,
        "module_helper_value": mod,
        "artifact_recorded_value": recorded,
        "recompute_matches_helper": local == mod,
        "artifact_matches_helper": recorded == mod if recorded else None,
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


def audit_reproducibility_rederived() -> dict:
    """Re-derive the reproducibility decision from the persisted rep-2
    held-out curves and the pre-registered stage-14 rep-1 constants."""
    a = _read_json(ART / "stage15_reproducibility.json")
    if a is None:
        return {"status": "MISSING ARTIFACT"}
    arms = a.get("arms") or {}
    if "control_rep2" not in arms or "structural_rep2" not in arms:
        return {"status": "MISSING rep-2 arm rows"}
    ctrl = arms["control_rep2"]
    struct = arms["structural_rep2"]
    delta_rep2 = round(struct["heldout_eval_aligned_final"]
                       - ctrl["heldout_eval_aligned_final"], 4)
    findings = a.get("findings") or {}
    curves_ok = all(
        len(arm.get("heldout_curve") or []) >= 2 for arm in
        (ctrl, struct))
    struct_le_control = all(
        struct[c] <= ctrl[c] for c in
        ("heldout_native_final", "heldout_eval_aligned_final"))
    rederived_reproduces = delta_rep2 < 0
    return {
        "rep2_control_heldout_eval_aligned": ctrl[
            "heldout_eval_aligned_final"],
        "rep2_structural_heldout_eval_aligned": struct[
            "heldout_eval_aligned_final"],
        "rep2_delta_rederived": delta_rep2,
        "reported_rep2_delta": findings.get("rep2_delta_stage15_seed"),
        "delta_matches_report": delta_rep2 == findings.get(
            "rep2_delta_stage15_seed"),
        "structural_le_control_native_and_aligned": struct_le_control,
        "curves_persisted": curves_ok,
        "reproduces_rederived": rederived_reproduces,
        "reproduces_matches_report": rederived_reproduces == findings.get(
            "reproduces"),
        "steps": a.get("steps"),
    }


def audit_feedback_coverage_rederived() -> dict:
    """Coverage invariants: rows built == real failures; every built row
    proves exact full-solution supervision; target-bearing steps span
    the whole run (the stage-14 102/600 defect must be gone)."""
    c = _read_json(ART / "stage15_feedback_training.json")
    if c is None:
        return {"status": "MISSING ARTIFACT"}
    co = c.get("candidate_outcomes") or {}
    cov = c.get("coverage_audit") or {}
    tr = c.get("training") or {}
    n = co.get("n") or 0
    passed = co.get("passed_reference_doctests") or 0
    rows_built = cov.get("rows_built")
    per_row = cov.get("per_row") or []
    all_exact = all(r.get("exact_match")
                    for r in per_row if r.get("fits")) \
        if per_row else None
    steps = tr.get("steps")
    tb = tr.get("target_bearing_steps")
    return {
        "n_candidates": n,
        "n_passed": passed,
        "rows_built": rows_built,
        "rows_built_equals_real_failures": rows_built == n - passed
        if rows_built is not None else None,
        "rows_fitting_seq": cov.get("rows_fitting_seq"),
        "rows_with_exact_match": cov.get(
            "rows_with_exact_solution_supervision"),
        "all_fitting_rows_exact_match": all_exact,
        "steps": steps,
        "target_bearing_steps": tb,
        "stage14_negative_control_target_bearing_steps": tr.get(
            "stage14_negative_control_target_bearing_steps"),
        "target_bearing_steps_equals_total": (
            tb is not None and steps is not None
            and tb == steps and tb > 0),
        "stage14_threshold_beaten": (
            tb is not None and tb > STAGE14_CHAMPION[
                "feedback_target_bearing_steps"]),
        "supervised_tokens_total": tr.get("supervised_tokens_total"),
        "contamination_zero": c.get(
            "benchmark_instruction_contamination_hits") == 0,
        "heldout_eval_aligned_final": (c.get("evaluation") or {}).get(
            "heldout_eval_aligned_final"),
        "delta_vs_stage14_negative_control": (c.get("evaluation") or {})
        .get("heldout_delta_vs_stage14_negative_feedback_control"),
    }


def audit_capability_rederived() -> dict:
    e = _read_json(ART / "stage15_generalization.json")
    if e is None:
        return {"status": "MISSING ARTIFACT"}
    rows = e.get("per_task") or []
    cap = e.get("capability") or {}
    if not rows:
        return {"status": "MISSING per_task rows"}
    out = {"n_per_task_rows": len(rows)}
    ok_all = True
    for suite in ("v1", "v3", "unseen"):
        srows = [r for r in rows if r.get("suite") == suite]
        if not srows:
            out[f"{suite}_pass_matches_report"] = None
            continue
        rd_pass = round(sum(1 for r in srows if r.get("passed"))
                        / len(srows), 4)
        rd_syntax = round(sum(1 for r in srows if r.get("parse"))
                          / len(srows), 4)
        rd_complete = round(sum(1 for r in srows
                                if r.get("complete_program"))
                            / len(srows), 4)
        out[f"{suite}_rederived"] = {
            "pass_at_1": rd_pass, "n_tasks": len(srows),
            "syntax_valid_rate": rd_syntax,
            "complete_program_rate": rd_complete}
        out[f"{suite}_pass_matches_report"] = (
            rd_pass == cap.get(f"{suite}_pass_at_1"))
        out[f"{suite}_syntax_matches_report"] = (
            rd_syntax == cap.get(f"{suite}_syntax_valid_rate"))
        ok_all = ok_all and out[f"{suite}_pass_matches_report"]
    out["all_pass_rates_match_report"] = ok_all
    harness = e.get("harness_sanity") or {}
    out["harness_all_references_pass"] = harness.get(
        "all_references_pass")
    return out


def audit_execution_data_rederived() -> dict:
    d = _read_json(ART / "stage15_execution_data.json")
    if d is None:
        return {"status": "MISSING ARTIFACT"}
    cands = d.get("candidates") or []
    if not cands:
        return {"status": "MISSING candidates"}
    rd: dict = {}
    for cand in cands:
        rd[cand["error_class"]] = rd.get(cand["error_class"], 0) + 1
    reported = d.get("error_class_distribution") or {}
    dist_match = all(reported.get(k) == v for k, v in rd.items()) and \
        sum(reported.values()) == sum(rd.values())
    return {
        "n_candidates": d.get("n_candidates"),
        "candidates_len": len(cands),
        "success_count_rederived": sum(1 for cand in cands
                                       if cand.get("passed")),
        "success_count_matches": sum(1 for cand in cands
                                     if cand.get("passed"))
        == d.get("success_count"),
        "error_distribution_rederived": rd,
        "error_distribution_matches_report": dist_match,
        "contaminated_candidates": d.get("contaminated_candidates"),
        "contamination_zero": d.get("contaminated_candidates") == 0,
    }


def audit_self_correction_rederived() -> dict:
    f = _read_json(ART / "stage15_self_correction.json")
    if f is None:
        return {"status": "MISSING ARTIFACT"}
    rows = f.get("per_task") or []
    rd_attempts = sum(1 for r in rows if "repair_passed" in r)
    rd_successes = sum(1 for r in rows if r.get("repair_passed"))
    rd_changed = sum(1 for r in rows
                     if r.get("repair_changed_candidate"))
    return {
        "n_tasks": f.get("n_tasks"),
        "per_task_rows": len(rows),
        "repair_attempts_rederived": rd_attempts,
        "repair_attempts_match": rd_attempts == f.get("repair_attempts"),
        "repair_successes_rederived": rd_successes,
        "repair_successes_match": rd_successes
        == f.get("repair_successes"),
        "revised_candidates_differed": rd_changed,
        "direct_successes": f.get("direct_successes"),
        "revised_executed": sum(1 for r in rows
                                if r.get("repair_executed")),
    }


def audit_checkpoints() -> dict:
    rows = []
    a = _read_json(ART / "stage15_reproducibility.json") or {}
    for name, arm in (a.get("arms") or {}).items():
        rows.append({"arm": name, "path": arm.get("checkpoint"),
                     "recorded_sha256": arm.get("checkpoint_sha256")})
    c = _read_json(ART / "stage15_feedback_training.json") or {}
    tr = c.get("training") or {}
    if tr:
        rows.append({"arm": "stage15-feedback", "path": tr.get("checkpoint"),
                     "recorded_sha256": tr.get("checkpoint_sha256")})
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
    return {
        "checkpoints": out_rows,
        "all_stage15_recorded_sha256_match": bool(rows) and all(
            r["sha256_match"] for r in out_rows),
        "all_checkpoint_paths_exist": bool(out_rows) and all(
            r["exists"] for r in out_rows),
    }


def audit_stage14_checkpoints_unchanged() -> dict:
    """Re-hash the stage-14 checkpoints against the sha256 values
    recorded in the persisted stage-14 artifacts (negative results and
    the champion must remain byte-identical)."""
    b14 = _read_json(ART / "stage14_feedback_training.json") or {}
    h14 = _read_json(ART / "stage14_horizon_arms.json") or {}
    rows = []
    tr = b14.get("training") or {}
    if tr:
        rows.append(("stage14-feedback", tr.get("checkpoint"),
                     tr.get("checkpoint_sha256")))
    for name, arm in (h14.get("arms") or {}).items():
        rows.append((name, arm.get("checkpoint"),
                     arm.get("checkpoint_sha256")))
    out = []
    for name, path, recorded in rows:
        p = Path(path) if path else None
        actual = _sha(p) if p else None
        out.append({"arm": name, "path": path,
                    "exists": bool(p and p.exists()),
                    "recorded_sha256": recorded, "actual_sha256": actual,
                    "sha256_match": recorded == actual})
    return {
        "checkpoints": out,
        "all_match": bool(out) and all(r["sha256_match"] and r["exists"]
                                       for r in out),
    }


def audit_recovery_reexecuted() -> dict:
    """Real torch.load + state_dict restore for every stage-15
    checkpoint (not trusting recorded recovery flags)."""
    import torch

    from scripts.stage10_run import dense_model_from_warm6

    results = []
    for name in ("stage15-control_rep2", "stage15-structural_rep2",
                 "stage15-feedback"):
        p = CKPT10_DIR / f"cortex-sft-{name}.pt"
        if not p.exists():
            results.append({"checkpoint": name, "exists": False})
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            model = dense_model_from_warm6()
            model.load_state_dict(ck["model_state"])
            results.append({"checkpoint": name, "exists": True,
                            "restores": True, "steps": ck.get("steps")})
            del model, ck
        except Exception as exc:
            results.append({"checkpoint": name, "exists": True,
                            "restores": False, "error": repr(exc)})
    return {
        "results": results,
        "all_restore": bool(results) and all(r.get("restores")
                                             for r in results),
    }


def audit_promotion_rederived() -> dict:
    a = _read_json(ART / "stage15_reproducibility.json")
    e = _read_json(ART / "stage15_generalization.json")
    f = _read_json(ART / "stage15_self_correction.json")
    j = _read_json(ART / "stage15_promotion_audit.json")
    if any(x is None for x in (a, e, f, j)):
        return {"status": "MISSING ARTIFACT"}
    arms = a.get("arms") or {}
    struct = arms.get("structural_rep2") or {}
    cap = e.get("capability") or {}
    pak = e.get("unseen_pass_at_k") or {}
    pert = e.get("perturbation") or {}
    v1 = cap.get("v1_pass_at_1") or 0
    v3 = cap.get("v3_pass_at_1") or 0
    unseen = cap.get("unseen_pass_at_1") or 0
    perturbed = pert.get("pass_at_1") or 0
    total = (pak.get("tasks_with_at_least_one_pass") or 0) \
        + (f.get("direct_successes") or 0)
    gate1 = bool((a.get("findings") or {}).get("reproduces"))
    gate2 = ((cap.get("v1_syntax_valid_rate") or 0) > 0
             or (cap.get("v3_syntax_valid_rate") or 0) > 0) and (
        (cap.get("v1_complete_program_rate") or 0) > 0
        or (cap.get("v3_complete_program_rate") or 0) > 0)
    gate3 = unseen > 0
    gate4 = perturbed > 0
    gate5 = (v1 > 0 or v3 > 0)
    gate6 = total >= 2
    heldout_ok = (struct.get("heldout_native_final", 99)
                  <= G_HELDOUT_LIMIT
                  and struct.get("heldout_eval_aligned_final", 99)
                  <= G_HELDOUT_LIMIT)
    rederived = gate5 and gate6 and heldout_ok
    gates_rederived = {
        "gate1_reproducible_structural_improvement": gate1,
        "gate2_complete_generation": gate2,
        "gate3_execution_improvement": gate3,
        "gate4_generalization": gate4,
        "gate5_independent_coding_success": gate5,
        "gate6_reliable_improvement": gate6,
    }
    reported_gates = {k: v.get("ok") for k, v in
                      (j.get("gates") or {}).items()}
    gates_agree = all(reported_gates.get(k) == v
                      for k, v in gates_rederived.items())
    reported = j.get("experiment", {}).get("decision", "")
    decision_agrees = (
        rederived and "PROMOTED" in reported and "NOT" not in reported
    ) or ((not rederived) and "NOT PROMOTED" in reported)
    return {
        "rederived_gates": gates_rederived,
        "reported_gates_ok": reported_gates,
        "gates_agree": gates_agree,
        "heldout_loss_gate_rederived": heldout_ok,
        "rederived_promoted": rederived,
        "reported_decision": reported,
        "promotion_decision_agrees": decision_agrees,
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


def audit_stage_lineage_intact() -> dict:
    arts = {}
    for name in STAGE_10_TO_14_ARTIFACTS_MUST_REMAIN:
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
        CKPT10_DIR / "cortex-sft-stage14-structural_h6000.pt",
        ART / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt",
    ]
    return {
        "stage10_to_14_artifacts": arts,
        "preserved_checkpoints_present": {
            str(p): p.exists() for p in champs},
    }


def run_audit() -> dict:
    fp = audit_fingerprints()
    bench = audit_benchmarks_untouched()
    unseen = audit_unseen_fingerprint()
    repro = audit_reproducibility_rederived()
    cov = audit_feedback_coverage_rederived()
    cap = audit_capability_rederived()
    exec_data = audit_execution_data_rederived()
    repair = audit_self_correction_rederived()
    ckpts = audit_checkpoints()
    ck14 = audit_stage14_checkpoints_unchanged()
    recovery = audit_recovery_reexecuted()
    promo = audit_promotion_rederived()
    store = audit_store_untouched()
    lineage = audit_stage_lineage_intact()

    arts = {}
    for name in STAGE15_EXPERIMENT_ARTIFACTS + STAGE15_PLAIN_ARTIFACTS:
        doc = _read_json(ART / name)
        if doc is None:
            arts[name] = "MISSING"
        elif isinstance(doc, dict) and "_unreadable" in doc:
            arts[name] = "UNREADABLE"
        else:
            arts[name] = "OK" if doc.get("owned_by") == "Fibonera" \
                else "OWNERSHIP MISSING"

    repro_ok = bool(repro.get("reproduces_matches_report")) and bool(
        repro.get("delta_matches_report")) and \
        bool(repro.get("curves_persisted"))
    cov_ok = bool(cov.get("rows_built_equals_real_failures")) and (
        cov.get("all_fitting_rows_exact_match") is True) and bool(
        cov.get("target_bearing_steps_equals_total")) and bool(
        cov.get("stage14_threshold_beaten")) and bool(
        cov.get("contamination_zero"))
    cap_ok = bool(cap.get("all_pass_rates_match_report")) and bool(
        cap.get("harness_all_references_pass"))
    exec_ok = bool(exec_data.get("error_distribution_matches_report")) \
        and bool(exec_data.get("success_count_matches")) and bool(
        exec_data.get("contamination_zero"))
    repair_ok = bool(repair.get("repair_attempts_match")) and bool(
        repair.get("repair_successes_match"))
    checks = {
        "A_reproducibility_rederived": {
            "status": "PASS" if repro_ok else "FAIL", "evidence": repro},
        "C_feedback_coverage_rederived": {
            "status": "PASS" if cov_ok else "FAIL", "evidence": cov},
        "capability_metrics_rederived": {
            "status": "PASS" if cap_ok else "FAIL", "evidence": cap},
        "execution_data_rederived": {
            "status": "PASS" if exec_ok else "FAIL", "evidence": exec_data},
        "self_correction_rederived": {
            "status": "PASS" if repair_ok else "FAIL", "evidence": repair},
        "checkpoint_integrity_stage15": {
            "status": "PASS" if ckpts.get(
                "all_stage15_recorded_sha256_match")
            and ckpts.get("all_checkpoint_paths_exist") else "FAIL",
            "evidence": ckpts},
        "stage14_checkpoints_unchanged": {
            "status": "PASS" if ck14.get("all_match") else "FAIL",
            "evidence": ck14},
        "checkpoint_recovery_reexecuted": {
            "status": "PASS" if recovery.get("all_restore") else "FAIL",
            "evidence": recovery},
        "promotion_decision_rederived": {
            "status": "PASS"
            if promo.get("promotion_decision_agrees")
            and promo.get("gates_agree") else "FAIL",
            "evidence": promo},
        "authoritative_store_untouched": {
            "status": "PASS" if store.get("expected_35292") else "FAIL",
            "evidence": store},
        "fingerprints_immutable": {
            "status": "PASS" if all(
                [fp["v1_matches_immutable"], fp["v2_matches_immutable"],
                 fp["v3_matches_immutable"],
                 unseen["recompute_matches_helper"],
                 unseen["artifact_matches_helper"]]) else "FAIL",
            "evidence": {"benchmarks": fp, "unseen_suite": unseen}},
        "benchmark_files_untouched": {
            "status": "PASS" if bench["diff_vs_parent_commit_empty"]
            and bench["working_tree_clean_for_benchmark_files"]
            else "FAIL", "evidence": bench},
        "stage10_to_14_lineage_intact": {
            "status": "PASS" if all(
                v == "OK" for v in
                lineage["stage10_to_14_artifacts"].values())
            and all(lineage["preserved_checkpoints_present"].values())
            else "FAIL", "evidence": lineage},
        "stage15_artifacts_present_and_owned": {
            "status": "PASS" if all(v == "OK"
                                    for v in arts.values()) else "FAIL",
            "evidence": arts},
    }
    known_provenance = {
        "stage13_header_parent_commit_anomaly":
            "stage-13 artifact headers carry parent_commit=a8c4e98 from a "
            "shared stage-10 header helper; stage-14/stage-15 headers use "
            "corrected local helpers with the right parent commit",
    }
    return {
        "kind": "cortex-stage15-independent-audit",
        "owned_by": "Fibonera", "stage": 15,
        "parent_commit": PARENT_COMMIT,
        "software_vs_capability_note":
            "PASS entries verify software correctness and measurement "
            "integrity; learned capability is reported separately via "
            "the pre-registered 6-gate promotion decision and the "
            "measured pass@1 values",
        "known_provenance_notes": known_provenance,
        "independent_logic": {
            "fingerprints": fp,
            "unseen_suite_fingerprint": unseen,
            "benchmark_files": bench,
            "dataset_tokenizer_checksums": {
                "dataset_v3": _sha(DATASET),
                "tokenizer": _sha(TOKENIZER)},
            "store_recount": store,
            "stage14_checkpoints": ck14,
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
    (ART / "stage15_independent_audit.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out["summary_counts"], indent=2))
    for name, chk in out["checks"].items():
        print(f"[stage15-audit] {chk['status']:>5}  {name}")
    print("[stage15-audit] wrote artifacts/stage15_independent_audit.json")
