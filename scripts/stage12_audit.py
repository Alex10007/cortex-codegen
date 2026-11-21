"""Stage 12 independent acceptance audit (Workstream I).

Deliberately does NOT reuse the orchestrator's helpers where a second
implementation is practical: benchmark fingerprints are recomputed from
the raw dataclass fields, the EOT-in-context claim is re-derived with a
independent make_batch/x-row scan, stored generation text_heads are
re-parsed with the AST to verify the reported syntax rates, checkpoint
sha256 values are recomputed from disk, and the promotion decision is
re-derived from per-task execution rows. Statuses are strict PASS / FAIL
/ BLOCKED / NOT YET TESTED.
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
PARENT_COMMIT = "082c5eb"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"
FINGERPRINT_V3 = "c70feb13b061a35b9b3940296c40ea8365aa57089b325050ef238dc93774ccbb"

STAGE12_ARTIFACTS = (
    "stage12_report.json",
    "stage12_autoregressive_diagnostics.json",
    "stage12_structure_learning.json",
    "stage12_generalization.json",
    "stage12_training_experiments.json",
    "stage12_generation_reliability.json",
    "stage12_capability_evaluation.json",
    "stage12_capacity_readiness.json",
    "stage12_promotion_audit.json",
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


def audit_diagnostics_reparsed() -> dict:
    """Re-parse the stored text_head of every task row with the Python
    ast module and re-derive the per-subject parse rates."""
    doc = _read_json(ART / "stage12_autoregressive_diagnostics.json")
    checks: dict = {}
    if doc is None or "per_subject" not in (doc or {}):
        return {"status": "MISSING ARTIFACT"}
    import ast as _ast

    for name, subj in doc["per_subject"].items():
        rows = subj.get("per_task") or []
        reparsed = 0
        for r in rows:
            code = (r.get("text_head") or "").strip()
            try:
                _ast.parse(code)
                reparsed += 1
            except SyntaxError:
                pass
        measured = reparsed / max(len(rows), 1)
        delta = abs(measured - subj.get("parse_rate", -1))
        checks[name] = {
            "n_rows": len(rows),
            "reparsed_parse_rate_from_stored_text_head":
                round(measured, 4),
            "reported_parse_rate": subj.get("parse_rate"),
            "abs_delta": round(delta, 4),
            "close_within_0.05": delta <= 0.05,
        }
    eot = doc.get("eot_in_training_context") or {}
    checks["eot_claim_report"] = {
        "legacy_total": (eot.get("legacy") or {}).get(
            "eot_in_context_total"),
        "eval_aligned_total": (eot.get("eval_aligned") or {}).get(
            "eot_in_context_total"),
    }
    return checks


def audit_eot_claim_with_make_batch() -> dict:
    """Independently re-derive the EOT-in-context claim with the
    supervision-relevant definition: an EOT token in the padded x row
    only matters when it PRECEDES a supervised (unmasked) target. The
    trailing EOT of a fitting row sits at the last x slot with a -100
    label after it, so it conditions nothing."""
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer
    from cortex.training.sft import load_split, make_batch

    tok = load_tokenizer(str(TOKENIZER))
    recs = load_split(str(DATASET), "train")[:32]
    counts = {}
    for style in ("legacy", "eval_aligned"):
        x, y = make_batch(recs, tok, 256, prompt_style=style)
        if x is None:
            counts[style] = None
            continue
        tot = 0
        for row_x, row_y in zip(x, y):
            sup = (row_y != -100).nonzero()
            if sup.numel() == 0:
                continue
            first_sup = int(sup[0, 0])
            tot += int(((row_x == tok.eot_id)[:first_sup]).sum())
        counts[style] = tot
    return {
        "recomputed_eot_in_context_before_first_target_counts": counts,
        "claim_supported": counts.get("legacy") == 0
        and (counts.get("eval_aligned") or 0) > 0,
    }


def audit_checkpoints() -> dict:
    d = _read_json(ART / "stage12_training_experiments.json") or {}
    rows = []
    for c in d.get("checkpoints") or []:
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
    parent = _read_json(ART / "stage12_promotion_audit.json") or {}
    bl = (parent.get("baseline_checkpoint") or {})
    if bl:
        rows.append({
            "arm": "warm6-parent-baseline", "path": bl.get("path"),
            "exists": Path(bl["path"]).exists(),
            "recorded_sha256": bl.get("sha256"),
            "actual_sha256": _sha(Path(bl["path"])),
            "sha256_match": bl.get("sha256") ==
            _sha(Path(bl["path"])), "steps": None,
        })
    return {
        "checkpoints": rows,
        "all_recorded_sha256_match": bool(rows)
        and all(r["sha256_match"] for r in rows),
    }


def audit_capability_and_promotion() -> dict:
    f = _read_json(ART / "stage12_capability_evaluation.json")
    h = _read_json(ART / "stage12_promotion_audit.json")
    if f is None or h is None or "results" not in (f or {}):
        return {"status": "MISSING ARTIFACT"}
    derived = {}
    for name, r in f["results"].items():
        rows = r.get("per_task") or []
        derived[name] = {
            "n_rows": len(rows),
            "recomputed_pass_at_1": round(sum(
                1 for x in rows if x.get("passed")) / max(len(rows), 1),
                4),
            "reported_pass_at_1": r.get("pass_at_1"),
            "agrees": abs(sum(1 for x in rows if x.get("passed"))
                          / max(len(rows), 1)
                          - r.get("pass_at_1", -1)) < 0.001,
        }
    any_pass = any(v["recomputed_pass_at_1"] > 0
                   for v in derived.values())
    decision = h.get("promotion_decision", "")
    return {
        "per_subject_repass": derived,
        "recomputed_any_pass": any_pass,
        "reported_promotion_decision": decision,
        "promotion_decision_agrees": (
            "NOT PROMOTED" in decision) != any_pass,
    }


def audit_independent_selftest() -> dict:
    from cortex.evaluation.autoregressive_diagnostics import \
        analyzer_selftest

    st = analyzer_selftest()
    wrong = [c for c in st.get("cases", [])
             if not bool(c.get("expectation_held", True))]
    return {
        "self_test_ok": st.get("self_test_ok"),
        "n_cases": len(st.get("cases", [])),
        "failed_cases": wrong[:5],
    }


def audit_structure_findings() -> dict:
    b = _read_json(ART / "stage12_structure_learning.json") or {}
    fnd = b.get("findings") or {}
    return {
        "windows_ending_inside_open_construct": fnd.get(
            "chunked_windows_ending_inside_open_construct"),
        "complete_program_rate_recorded": fnd.get(
            "complete_program_rate"),
        "single_function_dominance_claimed": fnd.get(
            "dataset_is_single_function_dominated"),
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
    }


def run_audit() -> dict:
    fp = audit_fingerprints()
    bench = audit_benchmarks_untouched()
    diag = audit_diagnostics_reparsed()
    eot = audit_eot_claim_with_make_batch()
    ckpts = audit_checkpoints()
    cap = audit_capability_and_promotion()
    selftest = audit_independent_selftest()
    struct = audit_structure_findings()
    store = audit_store_untouched()

    arts = {}
    for name in STAGE12_ARTIFACTS:
        doc = _read_json(ART / name)
        if doc is None:
            arts[name] = "MISSING"
        elif isinstance(doc, dict) and "_unreadable" in doc:
            arts[name] = "UNREADABLE"
        else:
            arts[name] = "OK" if doc.get("owned_by") == "Fibonera" \
                else "OWNERSHIP MISSING"

    diag_close = bool(diag.get("status") is None) and all(
        v.get("close_within_0.05", True)
        for v in diag.values() if isinstance(v, dict)
        and "close_within_0.05" in v)
    checks = {
        "A_diagnostics_independently_reparsed": {
            "status": "PASS" if diag_close else "FAIL",
            "evidence": diag},
        "EOT_context_claim_rederived": {
            "status": "PASS" if eot.get("claim_supported") else "FAIL",
            "evidence": eot},
        "checkpoint_integrity": {
            "status": "PASS" if ckpts[
                "all_recorded_sha256_match"] else "FAIL",
            "evidence": ckpts},
        "capability_promotion_rederived": {
            "status": "PASS"
            if cap.get("promotion_decision_agrees") else "FAIL",
            "evidence": cap},
        "analyzer_selftest_reexecuted": {
            "status": "PASS"
            if selftest.get("self_test_ok") else "FAIL",
            "evidence": selftest},
        "structure_findings_recorded": {
            "status": "PASS" if struct.get(
                "windows_ending_inside_open_construct") is not None
            else "FAIL", "evidence": struct},
        "authoritative_store_untouched": {
            "status": "PASS" if store.get("expected_35292")
            else "FAIL", "evidence": store},
        "fingerprints_immutable": {
            "status": "PASS" if all(
                [fp["v1_matches_immutable"], fp["v2_matches_immutable"],
                 fp["v3_matches_immutable"]]) else "FAIL",
            "evidence": fp},
        "benchmark_files_untouched": {
            "status": "PASS" if bench[
                "diff_vs_parent_commit_empty"] and bench[
                "working_tree_clean_for_benchmark_files"] else "FAIL",
            "evidence": bench},
        "artifacts_present_and_owned": {
            "status": "PASS" if all(v == "OK"
                                    for v in arts.values()) else "FAIL",
            "evidence": arts},
    }
    return {
        "kind": "cortex-stage12-independent-audit",
        "owned_by": "Fibonera", "stage": 12,
        "parent_commit": PARENT_COMMIT,
        "independent_logic": {
            "fingerprints": fp,
            "benchmark_files": bench,
            "dataset_tokenizer_checksums": {
                "dataset_v3": _sha(DATASET),
                "tokenizer": _sha(TOKENIZER)},
            "structure_findings": struct,
            "store_recount": store,
            "selftest": selftest,
        },
        "checks": checks,
        "summary_counts": {
            s: sum(1 for c in checks.values()
                   if str(c.get("status")).startswith(s))
            for s in ("PASS", "FAIL", "BLOCKED", "NOT YET TESTED")},
    }


if __name__ == "__main__":
    out = run_audit()
    (ART / "stage12_independent_audit.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out["summary_counts"], indent=2))
    print("[stage12-audit] wrote artifacts/stage12_independent_audit.json")
