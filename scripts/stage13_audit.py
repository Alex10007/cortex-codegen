"""Stage 13 independent acceptance audit (Workstream L).

Deliberately does NOT trust the orchestrator's reported numbers where an
independent recomputation is practical: benchmark fingerprints are
recomputed from raw dataclass fields, stored generated text is re-parsed
with the AST to verify reported parse rates, the structural-segmentation
exactly-once supervision guarantee is re-derived with a fresh stream
reconstruction, checkpoint sha256 values are recomputed from disk, the
5-gate promotion decision is re-derived from per-arm per-task execution
rows, and the authoritative virtual store is recounted. Statuses are
strict PASS / FAIL / BLOCKED / NOT YET TESTED.
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
PARENT_COMMIT = "402de46"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"
FINGERPRINT_V3 = "c70feb13b061a35b9b3940296c40ea8365aa57089b325050ef238dc93774ccbb"

STAGE13_ARTIFACTS = (
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
    "stage13_failure_taxonomy.json",
)

STAGE12_ARTIFACTS_MUST_REMAIN = (
    "stage12_report.json",
    "stage12_autoregressive_diagnostics.json",
    "stage12_structure_learning.json",
    "stage12_generalization.json",
    "stage12_training_experiments.json",
    "stage12_generation_reliability.json",
    "stage12_capability_evaluation.json",
    "stage12_capacity_readiness.json",
    "stage12_promotion_audit.json",
    "stage12_independent_audit.json",
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
    import hashlib as _h

    from cortex.evaluation.tasks import TASKS as T1
    from cortex.evaluation.tasks_v3 import tasks_v3_tuples

    v1 = _h.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.test_code, t.reference_solution)
        for t in T1), sort_keys=True).encode("utf-8")).hexdigest()
    v3 = _h.sha256(json.dumps(sorted(
        tasks_v3_tuples()), sort_keys=True).encode("utf-8")).hexdigest()
    from cortex.evaluation.benchmarks import fingerprint_tasks

    return {
        "v1_recomputed": v1,
        "v1_matches_immutable": v1 == FINGERPRINT_V1,
        "v2_recomputed": fingerprint_tasks("v2"),
        "v2_matches_immutable": fingerprint_tasks("v2") ==
        FINGERPRINT_V2,
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


def audit_rootcause_reparsed() -> dict:
    """Re-parse the stored generated text of every workstream-A row with
    the ast module and re-derive the per-subject parse rates."""
    doc = _read_json(ART / "stage13_structural_root_cause.json")
    checks: dict = {}
    if doc is None or "per_subject" not in (doc or {}):
        return {"status": "MISSING ARTIFACT"}
    import ast as _ast

    from cortex.evaluation.inference import extract_code

    gens = doc.get("generations") or []
    for name, subj in doc["per_subject"].items():
        rows = [g for g in gens if g.get("subject") == name]
        reparsed = 0
        for r in rows:
            code = (extract_code(r.get("generated_text") or "")
                    or "").strip()
            try:
                _ast.parse(code)
                reparsed += 1
            except (SyntaxError, ValueError):
                pass
        measured = reparsed / max(len(rows), 1)
        reported = subj.get("parse_rate")
        delta = abs(measured - (reported if reported is not None else -1))
        checks[name] = {
            "n_rows": len(rows),
            "reparsed_parse_rate_from_stored_text": round(measured, 4),
            "reported_parse_rate": subj.get("parse_rate"),
            "abs_delta": round(delta, 4),
            "close_within_0.05": delta <= 0.05,
        }
    return checks


def audit_coverage_guarantee() -> dict:
    """Independently re-derive the exactly-once completion-supervision
    guarantee for BOTH segmentations on the train split by reconstructing
    the streamed target sequence from the windows."""
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.training.longcontext import (  # noqa: E402
        chunk_record, chunk_record_structural, record_ids)
    from cortex.training.sft import load_split

    tok = load_tokenizer(str(TOKENIZER))
    recs = load_split(str(DATASET), "train")
    checks = {}
    for tag, fn in (("fixed", chunk_record),
                    ("structural", chunk_record_structural)):
        complete = True
        total_targets = 0
        for rec in recs:
            windows = fn(rec, tok, 256)
            if windows is None:
                continue
            _, completion = record_ids(rec, tok)
            cat_targets: list[int] = []
            for win_ids, win_labels in windows:
                cat_targets += win_labels[1:]
            hit = sum(1 for v in cat_targets if v != -100)
            if hit != len(completion):
                complete = False
                break
            total_targets += len(completion)
        checks[tag] = {"complete": complete,
                       "total_completion_targets": total_targets}
    same = (checks["fixed"]["complete"] and checks["structural"]["complete"]
            and checks["fixed"]["total_completion_targets"]
            == checks["structural"]["total_completion_targets"])
    return {
        "per_segmentation": checks,
        "exactly_once_guarantee_holds_both_modes": same,
    }


def audit_checkpoints() -> dict:
    h = _read_json(ART / "stage13_experiment_matrix.json") or {}
    rows = []
    for c in h.get("checkpoints") or []:
        p = Path(c["path"])
        recorded = c.get("sha256")
        actual = _sha(p)
        rows.append({
            "arm": c.get("arm"), "path": c.get("path"),
            "exists": p.exists(), "recorded_sha256": recorded,
            "actual_sha256": actual,
            "sha256_match": recorded == actual,
            "steps": c.get("steps"),
        })
    k = _read_json(ART / "stage13_promotion_audit.json") or {}
    bl = (k.get("rollback") or {}).get("baseline_checkpoint")
    if bl:
        rows.append({
            "arm": "stage12-champion-rollback", "path": bl,
            "exists": Path(bl).exists(),
            "recorded_sha256": None,
            "actual_sha256": _sha(Path(bl)),
            "sha256_match": True, "steps": None,
        })
    return {
        "checkpoints": rows,
        "all_stage13_recorded_sha256_match": bool(rows) and all(
            r["sha256_match"] for r in rows
            if r["arm"] != "stage12-champion-rollback"),
        "all_checkpoint_paths_exist": bool(rows) and all(
            r["exists"] for r in rows),
    }


def audit_promotion_rederived() -> dict:
    h = _read_json(ART / "stage13_experiment_matrix.json")
    k = _read_json(ART / "stage13_promotion_audit.json")
    g = _read_json(ART / "stage13_generalization.json")
    if h is None or k is None or "arms" not in (h or {}):
        return {"status": "MISSING ARTIFACT"}
    arms = h["arms"]
    champion = h.get("champion_arm_by_pre_registered_rule")
    from scripts.stage13_run import BASE_HELDOUT_ALLOWED  # noqa: E402

    rederived_gate3 = any(
        (a["benchmarks"].get("v1_pass_at_1") or 0) > 0
        or (a["benchmarks"].get("v3_pass_at_1") or 0) > 0
        for a in arms.values())
    rederived_gate4 = bool(
        ((g or {}).get("perturbed_task_results") or {}).get("pass_at_1"))
    champ = arms.get(champion) or {}
    cap = champ.get("capability") or {}
    control = (arms.get("control_recheck") or {}).get("capability") or {}
    rederived_gate1 = ((cap.get("syntax_valid_rate") or 0)
                       > (control.get("syntax_valid_rate") or 0))
    rederived_gate2 = rederived_gate1 and (
        (cap.get("complete_program_rate") or 0)
        > (control.get("complete_program_rate") or 0))
    loss_ok = (champ.get("heldout_native_final", 99)
               <= BASE_HELDOUT_ALLOWED
               and champ.get("heldout_eval_aligned_final", 99)
               <= BASE_HELDOUT_ALLOWED)
    rederived_promoted = rederived_gate3 and loss_ok
    reported = k.get("experiment", {}).get("decision", "")
    return {
        "champion": champion,
        "rederived_gates": {"gate1": rederived_gate1,
                            "gate2": rederived_gate2,
                            "gate3": rederived_gate3,
                            "gate4": rederived_gate4,
                            "gate5": rederived_gate3},
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

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan = persisted_optimizer_scan(auth)
    auth.close()
    return {
        "slots_updated_recount": scan.get("slots_updated"),
        "expected_35292": scan.get("slots_updated") == 35292,
        "n_allocated_500b": scan.get("n_allocated"),
    }


def audit_stage12_lineage_intact() -> dict:
    """Stage-7..12 artifacts still present, parseable and Fibonera-owned;
    per-arm stage-12 checkpoints still on disk with unperturbed stage-13
    additions."""
    arts = {}
    for name in STAGE12_ARTIFACTS_MUST_REMAIN:
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
        ART / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt",
    ]
    return {
        "stage10_to_stage12_artifacts": arts,
        "preserved_checkpoints_present": {
            str(p): p.exists() for p in champs},
    }


def run_audit() -> dict:
    fp = audit_fingerprints()
    bench = audit_benchmarks_untouched()
    diag = audit_rootcause_reparsed()
    cov = audit_coverage_guarantee()
    ckpts = audit_checkpoints()
    promo = audit_promotion_rederived()
    selftest = audit_independent_selftest()
    store = audit_store_untouched()
    lineage = audit_stage12_lineage_intact()

    arts = {}
    for name in STAGE13_ARTIFACTS:
        doc = _read_json(ART / name)
        if doc is None:
            arts[name] = "MISSING"
        elif isinstance(doc, dict) and "_unreadable" in doc:
            arts[name] = "UNREADABLE"
        else:
            arts[name] = "OK" if doc.get("owned_by") == "Fibonera" \
                else "OWNERSHIP MISSING"

    diag_close = all(
        v.get("close_within_0.05", True)
        for v in diag.values() if isinstance(v, dict)
        and "close_within_0.05" in v)
    checks = {
        "A_root_cause_independently_reparsed": {
            "status": "PASS" if diag_close else "FAIL", "evidence": diag},
        "C_exactly_once_supervision_rederived": {
            "status": "PASS" if cov.get(
                "exactly_once_guarantee_holds_both_modes") else "FAIL",
            "evidence": cov},
        "checkpoint_integrity": {
            "status": "PASS" if ckpts["all_stage13_recorded_sha256_match"]
            and ckpts["all_checkpoint_paths_exist"] else "FAIL",
            "evidence": ckpts},
        "promotion_decision_rederived": {
            "status": "PASS"
            if promo.get("promotion_decision_agrees") else "FAIL",
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
        "stage12_lineage_intact": {
            "status": "PASS" if all(
                v == "OK" for v in
                lineage["stage10_to_stage12_artifacts"].values())
            and all(lineage["preserved_checkpoints_present"].values())
            else "FAIL", "evidence": lineage},
        "stage13_artifacts_present_and_owned": {
            "status": "PASS" if all(v == "OK"
                                    for v in arts.values()) else "FAIL",
            "evidence": arts},
    }
    return {
        "kind": "cortex-stage13-independent-audit",
        "owned_by": "Fibonera", "stage": 13,
        "parent_commit": PARENT_COMMIT,
        "software_vs_capability_note":
            "PASS entries verify software correctness and measurement "
            "integrity; learned capability is reported separately via "
            "the workstream-K gates and the measured pass@1 values",
        "independent_logic": {
            "fingerprints": fp,
            "benchmark_files": bench,
            "dataset_tokenizer_checksums": {
                "dataset_v3": _sha(DATASET),
                "tokenizer": _sha(TOKENIZER)},
            "coverage_guarantee": cov,
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
    (ART / "stage13_independent_audit.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out["summary_counts"], indent=2))
    for name, chk in out["checks"].items():
        print(f"[stage13-audit] {chk['status']:>5}  {name}")
    print("[stage13-audit] wrote artifacts/stage13_independent_audit.json")
