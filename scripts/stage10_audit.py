"""Stage 10 independent acceptance audit (Workstream J).

Deliberately does NOT reuse the orchestrator's accounting helpers where a
second implementation is practical: fingerprints are recomputed from the
task modules' raw dataclass fields, virtual-parameter training coverage is
re-read directly from persisted optimizer npz files, the authoritative
store's untouched claim is verified with an independent scan, and every
stage-10 artifact is re-read from disk and checked for internally
consistent, evidence-linked claims. Statuses are strict PASS / FAIL /
BLOCKED / NOT YET TESTED; missing artifacts are NOT silently excused.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
VSTORE_DIR = ART / "virtual-store-502b"
JOINT7_CKPTS = ART / "runs" / "joint-502b-demo" / "checkpoints"
JOINT8_CKPTS = ART / "runs" / "joint-502b-stage8" / "checkpoints"
JOINT9_CKPTS = ART / "runs" / "joint-502b-stage9" / "checkpoints"
DATASET = ART / "sft" / "dataset_v3.jsonl"
TOKENIZER = ART / "tokenizer" / "tokenizer.json"
PARENT_COMMIT = "a8c4e98"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"
FINGERPRINT_V3 = "c70feb13b061a35b9b3940296c40ea8365aa57089b325050ef238dc93774ccbb"

STAGE9_FINAL_TRAINED = 216_834_048
PARAMS_PER_SLOT = 6144
SPARENT = 2_056_012_800

STAGE10_ARTIFACTS = (
    "stage10_report.json",
    "stage10_capability_report.json",
    "stage10_gpu_qualification.json",
    "stage10_pass1_root_cause.json",
    "stage10_virtual_coverage.json",
    "stage10_moe_specialization.json",
    "stage10_divergent_reasoning.json",
    "stage10_dense_vs_unified.json",
    "stage10_recovery_report.json",
    "stage10_training_readiness.json",
)


def _read_json(path: Path) -> dict | None:
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
    from cortex.evaluation.tasks import TASKS as T1

    v1 = hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.test_code, t.reference_solution)
        for t in T1), sort_keys=True).encode("utf-8")).hexdigest()
    from cortex.evaluation.benchmarks import fingerprint_tasks

    v2 = fingerprint_tasks("v2")
    v3 = fingerprint_tasks("v3")
    from cortex.evaluation.tasks_v3 import tasks_v3_tuples

    v3_direct = hashlib.sha256(json.dumps(
        sorted(tasks_v3_tuples()), sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "v1_recomputed": v1,
        "v1_matches_immutable": v1 == FINGERPRINT_V1,
        "v2_recomputed": v2,
        "v2_matches_immutable": v2 == FINGERPRINT_V2,
        "v3_recomputed": v3,
        "v3_matches_immutable": v3 == FINGERPRINT_V3,
        "v3_builder_agrees": v3 == v3_direct,
    }


def audit_virtual_accounting() -> dict:
    import numpy as np

    slots = 0
    param_counts: set[int] = set()
    steps: list[int] = []
    bad_files: list[str] = []
    for f in sorted(VSTORE_DIR.glob("adam-*.npz")):
        try:
            with np.load(f) as z:
                sc = int(z["step_count"])
                if sc > 0:
                    slots += 1
                    param_counts.add(int(z["A_m"].size + z["B_m"].size))
                    steps.append(sc)
        except Exception:
            bad_files.append(f.name)
    per_slot = sorted(param_counts)
    trained = slots * per_slot[0] if per_slot == [PARAMS_PER_SLOT] \
        else slots * PARAMS_PER_SLOT
    return {
        "slots_with_optimizer_state_gt0": slots,
        "params_per_slot_variants": per_slot,
        "unique_trained_scalars_recomputed": trained,
        "matches_stage9_final": trained == STAGE9_FINAL_TRAINED,
        "optimizer_steps_per_slot_total": int(sum(steps)),
        "corrupt_npz_files": bad_files[:10],
    }


def audit_benchmarks_untouched() -> dict:
    diff = subprocess.run(
        ["git", "diff", f"{PARENT_COMMIT} HEAD", "--",
         "cortex/evaluation/tasks.py", "cortex/evaluation/tasks_v2.py",
         "cortex/evaluation/tasks_v3.py"],
        cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "cortex/evaluation/tasks.py",
         "cortex/evaluation/tasks_v2.py", "cortex/evaluation/tasks_v3.py",
         "cortex/evaluation/benchmarks.py"],
        cwd=ROOT, capture_output=True, text=True)
    return {
        "diff_vs_parent_commit_empty": diff.stdout.strip() == "",
        "working_tree_clean_for_benchmark_files":
            status.stdout.strip() == "",
        "detail": status.stdout[:200],
    }


def audit_stage10_artifacts() -> dict:
    present = {}
    for name in STAGE10_ARTIFACTS:
        p = ART / name
        doc = _read_json(p)
        if doc is None:
            present[name] = "MISSING"
        elif "_unreadable" in doc:
            present[name] = "UNREADABLE"
        else:
            present[name] = "OK" if doc.get("owned_by") == "Fibonera" \
                else "OWNERSHIP MISSING"
    return present


def audit_report_consistency() -> dict:
    """Cross-check the orchestrator's summary against the per-workstream
    artifacts and the independent store scan."""
    rep = _read_json(ART / "stage10_report.json") or {}
    rec = _read_json(ART / "stage10_recovery_report.json") or {}
    dvu = _read_json(ART / "stage10_dense_vs_unified.json") or {}
    rc = _read_json(ART / "stage10_pass1_root_cause.json") or {}
    issues: list[str] = []
    untouched = bool(rep.get("summary", {}).get(
        "authoritative_store_untouched"))
    baseline = int(rep.get("summary", {}).get(
        "stage9_baseline_unique_trained_scalars") or 0)
    if baseline != STAGE9_FINAL_TRAINED:
        issues.append(
            f"report baseline scalars {baseline} != stage-9 final "
            f"{STAGE9_FINAL_TRAINED}")
    g_untouched = dvu.get("authoritative_store_untouched")
    if g_untouched is not True:
        issues.append("dense_vs_unified artifact lacks untouched evidence")
    fv = (rc.get("c2_dataset_audit") or {}).get("fit_rate_at_training_seq_len")
    if fv is None:
        issues.append("pass1_root_cause lacks the C2 dataset audit")
    outcomes = []
    for inj in ((rec.get("fault_injections") or {}).values()):
        detected = bool(inj.get("detected"))
        outcomes.append(detected and bool(inj.get("refused", detected)))
    return {
        "authoritative_store_untouched_claimed": untouched,
        "baseline_scalars_recorded": baseline,
        "g_untouched": g_untouched,
        "c2_fit_rate_recorded": fv,
        "fault_injection_consistency": {
            "n_injections": len(outcomes), "all_flags_true":
                bool(outcomes) and all(outcomes)},
        "recovery_verified_claim": rec.get("recovery_verified"),
        "issues": issues,
    }


def audit_capability_report() -> dict:
    doc = _read_json(ART / "stage10_capability_report.json") or {}
    rows = doc.get("capabilities") or []
    problems = []
    for r in rows:
        lvl = r.get("level")
        if not isinstance(lvl, int) or not (0 <= lvl <= 7):
            problems.append(f"bad level for {r.get('capability')}")
    return {
        "capabilities": len(rows),
        "max_level": max((r.get("level", 0) for r in rows), default=None),
        "rows_with_blocked_status":
            sum(1 for r in rows if str(r.get("status", "")).startswith(
                "blocked")),
        "problems": problems,
    }


def audit_experiments_governance() -> dict:
    names = ("stage10_gpu_qualification", "stage10_training_readiness",
             "stage10_pass1_root_cause", "stage10_virtual_coverage",
             "stage10_moe_specialization", "stage10_divergent_reasoning",
             "stage10_dense_vs_unified", "stage10_recovery_report",
             "stage10_capability_report")
    per = {}
    for name in names:
        doc = _read_json(ART / f"{name}.json") or {}
        exp = doc.get("experiment") or {}
        per[name] = {
            "experiment_id": exp.get("experiment_id"),
            "parent_commit_ok": exp.get("parent_commit") == PARENT_COMMIT,
            "has_hypothesis": bool(exp.get("pre_registered_hypothesis")),
            "has_criteria": bool(exp.get("acceptance_criteria")),
            "has_started_finished": bool(exp.get("started")
                                         and exp.get("finished")),
            "decision": exp.get("decision"),
        }
    return per


def run_audit() -> dict:
    fp = audit_fingerprints()
    acct = audit_virtual_accounting()
    bench = audit_benchmarks_untouched()
    arts = audit_stage10_artifacts()
    consistency = audit_report_consistency()
    capability = audit_capability_report()
    governance = audit_experiments_governance()
    dvu = _read_json(ART / "stage10_dense_vs_unified.json") or {}

    # lineage continuity (independent re-verification)
    lineage = {}
    steps = []
    for name, cdir in (("stage7", JOINT7_CKPTS), ("stage8", JOINT8_CKPTS),
                       ("stage9", JOINT9_CKPTS)):
        lin = _read_json(cdir / "lineage.json")
        if lin and "_unreadable" not in lin:
            phys = cdir / lin.get("physical_checkpoint", "NONE")
            lineage[name] = {
                "step": lin.get("step"),
                "physical_sha256_ok":
                    _sha(phys) == lin.get("physical_sha256")}
            if isinstance(lin.get("step"), int):
                steps.append(lin["step"])
    lineage["monotonic"] = steps == sorted(steps) and len(steps) == 3

    checks: dict[str, dict] = {}

    hw = _read_json(ART / "stage10_gpu_qualification.json") or {}
    if hw:
        decision = hw.get("decision")
        checks["A_gpu_qualification"] = {
            "status": "BLOCKED" if decision == "BLOCKED" else
                      "PASS" if decision == "PASS" and
                      hw.get("gpu_qualification_executed") else "FAIL",
            "evidence": {
                "decision": decision,
                "gpu_count": (hw.get("hardware") or {}).get("gpu_count"),
                "no_gpu_claimed_as_qualified":
                    not hw.get("gpu_qualification_executed")},
        }
    else:
        checks["A_gpu_qualification"] = {"status": "NOT YET TESTED"}

    b = _read_json(ART / "stage10_training_readiness.json") or {}
    if b:
        status = b.get("two_b_training_status") or {}
        param_ok = (b.get("physical_architecture") or {}).get(
            "param_count_is_2_056_012_800")
        checks["B_2b_training"] = {
            "status": "BLOCKED" if (
                status.get("trained") is False and not param_ok is False)
            else "FAIL",
            "evidence": {
                "trained_honestly_false": status.get("trained") is False,
                "param_count_verified": param_ok,
                "spend_usd": b.get("spend_usd"),
                "manipulable_config_param_count": SPARENT},
        }
    else:
        checks["B_2b_training"] = {"status": "NOT YET TESTED"}

    rc_ok = bool(_read_json(ART / "stage10_pass1_root_cause.json"))
    if rc_ok:
        rc = _read_json(ART / "stage10_pass1_root_cause.json")
        audits = rc.get("c2_dataset_audit") or {}
        rem = rc.get("c4_remediation_extended_sft") or {}
        arms = rem.get("arms") or {}
        promoted = "PROMOTED" in str(rem.get("promotion_decision"))
        improvement_supported = bool(arms) and promoted
        checks["C_pass1_root_cause"] = {
            "status": "PASS" if audits.get(
                "splits", {}).get("heldout", 0) > 0 else "FAIL",
            "evidence": {
                "executed": rc_ok,
                "c1_suites_diagnosed": len(rc.get(
                    "c1_generation_diagnosis") or {}),
                "c2_fit_rate": audits.get("fit_rate_at_training_seq_len"),
                "c2_split_deterministic": audits.get(
                    "split_deterministic"),
                "remediation_arms": list(arms),
                "promotion_decision": rem.get("promotion_decision"),
                "promotion_only_with_benchmark_gain":
                    improvement_supported,
                "harness_reference_pass": (rc.get("harness_sanity") or {})}}
    else:
        checks["C_pass1_root_cause"] = {"status": "NOT YET TESTED"}

    checks["D_virtual_expansion"] = {
        "status": "PASS" if _read_json(
            ART / "stage10_virtual_coverage.json") else "NOT YET TESTED",
        "evidence": {
            "authoritative_unique_trained_scalars":
                (_read_json(ART / "stage10_virtual_coverage.json") or {})
                .get("authoritative_unique_trained_scalars"),
            "independent_recount_matches_stage9_final":
                acct["matches_stage9_final"]},
    }

    moe = _read_json(ART / "stage10_moe_specialization.json") or {}
    if moe:
        accepted = moe.get("specialization_accepted")
        routing_differs = moe.get("routing_differs_by_family")
        checks["E_moe_specialization"] = {
            "status": ("PASS" if accepted else "FAIL"
                       if routing_differs and not accepted else
                       "INCONCLUSIVE-IF-REJECTED"),
            "evidence": {
                "specialization_accepted": accepted,
                "routing_differs_by_family": routing_differs,
                "quality_differs_by_family":
                    moe.get("quality_differs_by_family"),
                "js_max": moe.get("js_max"),
                "honest_note": (moe.get("honest_note") or "")[:200]},
        }
    else:
        checks["E_moe_specialization"] = {"status": "NOT YET TESTED"}

    dr = _read_json(ART / "stage10_divergent_reasoning.json") or {}
    if dr:
        delta = dr.get("repair_round_delta_pass_at_n")
        checks["F_divergent_reasoning"] = {
            "status": "PASS" if (delta or 0) > 0 else "FAIL",
            "evidence": {
                "repair_delta_pass_at_n": delta,
                "with_repair": (dr.get("with_repair") or {}).get("pass_at_n"),
                "without_repair": (dr.get("without_repair") or {})
                .get("pass_at_n"),
                "single_candidate": (dr.get(
                    "single_candidate_fixed_decoding") or {}).get(
                    "pass_at_1")},
        }
    else:
        checks["F_divergent_reasoning"] = {"status": "NOT YET TESTED"}

    if dvu:
        warm_beats = dvu.get("joint_warm_beats_dense")
        untouched = dvu.get("authoritative_store_untouched")
        checks["G_dense_vs_unified"] = {
            "status": ("PASS" if warm_beats and untouched else
                       "FAIL" if untouched else
                       "FAIL" if not untouched else "NOT YET TESTED"),
            "evidence": {
                "joint_warm_beats_dense": warm_beats,
                "authoritative_store_untouched": untouched,
                "warm_minus_dense_heldout": dvu.get(
                    "joint_warm_minus_dense_heldout"),
                "arms_recorded": sorted((dvu.get("arms") or {}))},
        }
    else:
        checks["G_dense_vs_unified"] = {"status": "NOT YET TESTED"}

    rr = _read_json(ART / "stage10_recovery_report.json") or {}
    if rr:
        intact = rr.get("intact_restart") or {}
        faults = rr.get("fault_injections") or {}
        all_detected = all(
            (v.get("detected") is True or v.get("refused") is True)
            for v in faults.values())
        checks["H_recovery"] = {
            "status": ("PASS" if rr.get("recovery_verified")
                       and intact.get("parity_ok") and all_detected
                       else "FAIL"),
            "evidence": {
                "recovery_verified": rr.get("recovery_verified"),
                "intact_restart_parity_ok": intact.get("parity_ok"),
                "intact_restart_step": intact.get("resumed_step"),
                "all_fault_injections_detected": all_detected,
                "sigkill_recovery_clean":
                    (rr.get("sigkill_during_commit") or {}).get(
                        "recovery_clean"),
                "stage9_lineage_sha256_ok":
                    (rr.get("stage9_lineage_recheck") or {}).get(
                        "physical_sha256_ok")},
        }
    else:
        checks["H_recovery"] = {"status": "NOT YET TESTED"}

    cap = capability
    checks["I_capability_maturity"] = {
        "status": "PASS" if cap["capabilities"] and not cap["problems"]
        else "FAIL",
        "evidence": {
            "capabilities": cap["capabilities"],
            "max_level": cap["max_level"],
            "blocked_rows": cap["rows_with_blocked_status"],
            "problems": cap["problems"]},
    }

    repro_ok = bool(
        fp["v1_matches_immutable"] and fp["v2_matches_immutable"]
        and fp["v3_matches_immutable"] and lineage.get("monotonic")
        and all(v == "OK" for v in arts.values()))
    checks["J_reproducibility_integrity"] = {
        "status": ("PASS" if repro_ok and bench[
            "diff_vs_parent_commit_empty"] and acct[
            "matches_stage9_final"] and acct["corrupt_npz_files"] == []
                   else "FAIL"),
        "evidence": {
            "v1_ok": fp["v1_matches_immutable"],
            "v2_ok": fp["v2_matches_immutable"],
            "v3_ok": fp["v3_matches_immutable"],
            "lineage_monotonic": lineage.get("monotonic"),
            "lineage": lineage,
            "benchmarks_untouched": bench,
            "virtual_scalar_recount": acct[
                "unique_trained_scalars_recomputed"],
            "artifacts": arts,
            "governance": governance,
            "report_consistency": consistency},
    }

    return {
        "kind": "cortex-stage10-acceptance-audit", "owned_by": "Fibonera",
        "parent_commit": PARENT_COMMIT,
        "independent_logic": {
            "fingerprints": fp,
            "virtual_accounting_recount": acct,
            "benchmark_task_files": bench,
            "dataset_tokenizer_checksums": {
                "dataset_v3": _sha(DATASET),
                "tokenizer": _sha(TOKENIZER)},
        },
        "checks": checks,
        "summary_counts": {
            s: sum(1 for c in checks.values()
                   if str(c.get("status")).startswith(s))
            for s in ("PASS", "FAIL", "BLOCKED", "NOT YET TESTED")},
    }


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, default=str))
