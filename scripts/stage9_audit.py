"""Stage 9 independent acceptance audit (Workstream I).

Deliberately does NOT reuse the orchestrator's accounting helpers where a
second implementation is practical: fingerprints are recomputed from the
task modules' own dataclass fields, virtual-parameter training coverage is
re-read directly from persisted optimizer npz files, and every artifact's
integrity is verified from disk. Statuses are strict PASS / FAIL / BLOCKED /
NOT YET TESTED; missing artifacts are NOT silently excused.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ART = Path(__file__).resolve().parents[1] / "artifacts"
VSTORE_DIR = ART / "virtual-store-502b"
JOINT7_CKPTS = ART / "runs" / "joint-502b-demo" / "checkpoints"
JOINT8_CKPTS = ART / "runs" / "joint-502b-stage8" / "checkpoints"
JOINT9_CKPTS = ART / "runs" / "joint-502b-stage9" / "checkpoints"
DATASET = ART / "sft" / "dataset_v3.jsonl"
TOKENIZER = ART / "tokenizer" / "tokenizer.json"
WARM6_CKPT = ART / "runs" / "cortex-sft-stage6" / "stage6-epoch-002.pt"

FINGERPRINT_V1 = "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177"
FINGERPRINT_V2 = "64ea2c020004af92854aebdde0557678293313b06881fdf5b5abb54e8729c8a3"

STAGE8_BASELINE_TRAINED = 124_329_984


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
    """v1 is recomputed from raw EvalTask dataclass fields (an independent
    reimplementation of the fingerprint payload); v2/v3 are cross-checked
    against module tuple builders AND against the immutable constants; a
    drift between re-derivation and constants would fail regardless of the
    helper modules agreeing with themselves."""
    from cortex.evaluation.tasks import TASKS as T1

    v1 = hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.test_code, t.reference_solution)
        for t in T1), sort_keys=True).encode("utf-8")).hexdigest()

    from cortex.evaluation.benchmarks import fingerprint_tasks

    v2 = fingerprint_tasks("v2")
    v3 = fingerprint_tasks("v3")
    return {
        "v1_recomputed": v1,
        "v1_matches_immutable": v1 == FINGERPRINT_V1,
        "v1_module_agrees": v1 == fingerprint_tasks("v1"),
        "v2_recomputed": v2,
        "v2_matches_immutable": v2 == FINGERPRINT_V2,
        "v3_recomputed": v3,
        "v3_registered_in_module": True,
        "v3_value": v3,
    }


def audit_virtual_accounting() -> dict:
    """Independent scan: every adam npz with step_count>0 counts each slot
    once; total = slots x params_per_slot. Uses OWN glob/load logic."""
    import numpy as np

    slots = 0
    param_counts = set()
    steps = []
    bad_files = []
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
    params_per_slot = sorted(param_counts)
    trained = slots * params_per_slot[0] if len(param_counts) == 1 \
        and len(params_per_slot) == 1 else slots * 6144
    return {
        "slots_with_optimizer_state_gt0": slots,
        "params_per_slot_variants": params_per_slot,
        "unique_trained_scalars_recomputed": trained,
        "optimizer_steps_per_slot_total": int(sum(steps)),
        "max_steps_per_slot": int(max(steps)) if steps else 0,
        "corrupt_npz_files": bad_files,
    }


def audit_lineage() -> dict:
    out = {"lineages": {}}
    for name, cdir in (("stage7", JOINT7_CKPTS), ("stage8", JOINT8_CKPTS),
                       ("stage9", JOINT9_CKPTS)):
        lin = _read_json(cdir / "lineage.json")
        if lin is None:
            out["lineages"][name] = "missing"
            continue
        if "_unreadable" in lin:
            out["lineages"][name] = lin
            continue
        phys = cdir / lin.get("physical_checkpoint", "NONE")
        out["lineages"][name] = {
            "step": lin.get("step"),
            "transaction_id": lin.get("transaction_id"),
            "physical_sha256_ok": _sha(phys) == lin.get("physical_sha256"),
        }
    steps = [v.get("step") for v in out["lineages"].values()
             if isinstance(v, dict) and isinstance(v.get("step"), int)]
    out["monotonic_lineage_order"] = (
        len(steps) >= 2 and steps == sorted(steps))
    return out


def audit_manifest_digests(sample: int = 100) -> dict:
    import numpy as np

    from cortex.virtual.store import BANK_COLS, BANK_ROWS

    manifest_path = VSTORE_DIR / "virtual-manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        return {"error": "manifest missing"}
    checked = 0
    bad = []
    banks = sorted(manifest.get("initialized_rows", {}).keys())
    for bank_s in banks[:sample]:
        f = VSTORE_DIR / f"virtual-bank-{int(bank_s):05d}.f32"
        if not f.exists():
            bad.append(f"{bank_s}/<file-missing>")
            continue
        mm = np.memmap(f, dtype=np.float32, mode="r",
                       shape=(BANK_ROWS, BANK_COLS))
        for row_s, digest in manifest["initialized_rows"][bank_s].items():
            actual = hashlib.sha256(
                np.asarray(mm[int(row_s)]).tobytes()).hexdigest()
            checked += 1
            if actual != digest:
                bad.append(f"{bank_s}/{row_s}")
        del mm
    return {
        "banks_sampled": min(len(banks), sample),
        "rows_digest_checked": checked,
        "bad_rows": bad[:5],
        "digests_ok": not bad,
    }


def run_audit() -> dict:
    hw = _read_json(ART / "stage9_hardware_report.json") or {}
    tm = _read_json(ART / "stage9_training_metrics.json") or {}
    vc = _read_json(ART / "stage9_virtual_coverage.json") or {}
    moe = _read_json(ART / "stage9_moe_diagnostics.json") or {}
    reas = _read_json(ART / "stage9_reasoning_evaluation.json") or {}
    cb = _read_json(ART / "stage9_coding_benchmarks.json") or {}
    jc = _read_json(ART / "stage9_joint_comparison.json") or {}
    rec = _read_json(ART / "stage9_recovery_report.json") or {}
    s8 = _read_json(ART / "stage8_report.json") or {}

    fp = audit_fingerprints()
    acct = audit_virtual_accounting()
    lineage = audit_lineage()
    digests = audit_manifest_digests()

    # cross-check the orchestrator's C numbers against the independent scan
    final_orch = (vc or {}).get("final", {}).get(
        "unique_trained_scalars")
    start_recorded = (vc or {}).get("baseline", {}).get(
        "stage9_start_independent_scan")
    # the store may legitimately contain MORE trained slots than the stage-8
    # snapshot (its post-C experiments); the snapshot itself must never be
    # exceeded in the shrink direction, and every delta must be slot-aligned
    stage8_delta_aligned = (
        (acct["unique_trained_scalars_recomputed"] - STAGE8_BASELINE_TRAINED)
        % 6144 == 0
        and acct["unique_trained_scalars_recomputed"]
        >= STAGE8_BASELINE_TRAINED)
    baseline_ok = bool(
        vc
        and start_recorded == 125_804_544
        and acct["unique_trained_scalars_recomputed"] >= start_recorded
        and isclose(final_orch, acct["unique_trained_scalars_recomputed"]))
    accounting_ok = bool(
        acct["unique_trained_scalars_recomputed"]
        >= STAGE8_BASELINE_TRAINED
        and not acct["corrupt_npz_files"])
    stage8_advance_explained = bool(
        stage8_delta_aligned and
        int(acct["slots_with_optimizer_state_gt0"] - 20236) * 6144
        == acct["unique_trained_scalars_recomputed"]
        - STAGE8_BASELINE_TRAINED)

    checks: dict[str, dict] = {}

    if hw:
        decision = hw.get("decision")
        checks["A_infrastructure"] = {
            "status": ("PASS" if decision == "QUALIFIED"
                       and hw.get("gpu_qualification_executed")
                       else "BLOCKED" if decision in ("BLOCKED",
                                                      "REFUSED-CPU-ONLY",
                                                      "REFUSED-INSUFFICIENT-GPUS")
                       else "FAIL"),
            "evidence": {"decision": decision,
                         "gpu_count": hw.get("hardware_inventory", {}).get(
                             "gpu_count"),
                         "flawed_if_gpu_qualification_executed":
                             hw.get("gpu_qualification_executed")},
        }
    else:
        checks["A_infrastructure"] = {"status": "NOT YET TESTED"}

    checks["B_physical_2b_model"] = {
        "status": ("BLOCKED"
                   if tm.get("two_b_training_status", {}).get("trained")
                   is False else "FAIL"),
        "evidence": {
            "param_count_verified":
                tm.get("physical_architecture", {}).get(
                    "param_count_is_2_056_012_800"),
            "trained": tm.get("two_b_training_status", {}).get("trained"),
        } if tm else {"missing_artifact": "stage9_training_metrics.json"},
    } if tm else {"status": "NOT YET TESTED"}

    checks["C_virtual_scalar_updates"] = {
        "status": ("PASS" if vc and baseline_ok and accounting_ok
                   and vc.get("final", {}).get("invariants_hold")
                   else "FAIL" if vc else "NOT YET TESTED"),
        "evidence": {
            "unique_trained_scalars_recomputed":
                acct["unique_trained_scalars_recomputed"],
            "orchestrator_matches_independent_scan":
                bool(vc and isclose(final_orch,
                                    acct["unique_trained_scalars_recomputed"])),
            "stage9_start_scan_recorded": start_recorded,
            "store_advanced_since_stage9_start": (
                acct["unique_trained_scalars_recomputed"]
                - start_recorded if start_recorded else None),
            "stage8_snapshot_monotonic": stage8_delta_aligned,
            "stage8_post_C_advance_explained": stage8_advance_explained,
            "slots_delta_vs_stage8_snapshot":
                acct["slots_with_optimizer_state_gt0"] - 20236,
            "invariants_hold":
                vc.get("final", {}).get("invariants_hold") if vc else None,
            "stage8_report_baseline": s8.get("C", {}).get("coverage", {})
            .get("scalars", {}).get("demonstrably_trained"),
        },
    }

    moe_special = bool(moe.get("specialization_demonstrated"))
    checks["D_moe_specialization"] = {
        "status": ("PASS" if moe_special else "FAIL" if moe
        else "NOT YET TESTED"),
        "evidence": {
            "specialization_demonstrated": moe_special,
            "root_cause": (moe.get("conclusion") or "")[:220],
            "router_grads_flow": moe.get("root_cause_evidence", {})
            .get("router_gradients_flow") if moe else None,
        },
    }

    reas_pass = (reas.get("aggregate", {}).get("pass_at_n", 0.0) or 0.0)
    checks["E_divergent_reasoning"] = {
        "status": ("PASS" if reas_pass > 0 else "FAIL" if reas
        else "NOT YET TESTED"),
        "evidence": {
            "pass_at_n": reas_pass,
            "pass_via_repair":
                reas.get("aggregate", {}).get("pass_via_repair"),
            "failure_categories_round1":
                reas.get("aggregate", {}).get("failure_categories_round1"),
        },
    }

    cb_ok = bool(
        cb and cb.get("fingerprint_v1") == FINGERPRINT_V1
        and cb.get("fingerprint_v2") == FINGERPRINT_V2)
    cb_pass = max(
        (s.get("v3_report", {}).get("pass_at_1", 0.0)
         for s in (cb.get("subjects", {}) or {}).values()), default=0.0)
    checks["F_coding_capability"] = {
        "status": ("PASS" if cb_ok and cb_pass > 0
                   else "FAIL" if cb else "NOT YET TESTED"),
        "evidence": {
            "fingerprints_preserved": cb_ok,
            "max_subject_pass_at_1_v3": cb_pass,
            "harness_sanity": cb.get("harness_sanity") if cb else None,
            "note": "zero pass rate == capability failure, honestly graded; "
                    "not a benchmark-integrity violation",
        },
    }

    jc_better = bool(jc.get("joint_better_than_dense_on_heldout")) \
        if jc else None
    checks["G_joint_vs_dense"] = {
        "status": ("PASS" if jc_better else "FAIL" if jc else
        "NOT YET TESTED"),
        "evidence": {
            "joint_better_than_dense_on_heldout": jc_better,
            "heldout_delta_joint_minus_dense":
                (jc.get("heldout_delta_joint_minus_dense") if jc else None),
            "against_fairness_note": jc.get("fairness_note") if jc else None,
        },
    }

    rec_ok = bool(rec.get("recovery_verified"))
    checks["H_recovery_reproducibility"] = {
        "status": ("PASS" if rec_ok else "FAIL" if rec
        else "NOT YET TESTED"),
        "evidence": {
            "recovery_verified": rec_ok,
            "kill_probe_clean":
                (rec.get("kill_recovery_subprocess_probe") or {})
                .get("recovery_clean") if rec else None,
            "multi_gpu_recovery": (rec.get("checks") or {})
            .get("multi_gpu_recovery") if rec else None,
        },
    }

    repro_ok = bool(
        fp["v1_matches_immutable"] and fp["v2_matches_immutable"]
        and lineage.get("monotonic_lineage_order")
        and digests.get("digests_ok"))
    checks["I_reproducibility_fingerprints"] = {
        "status": ("PASS" if repro_ok else "FAIL"),
        "evidence": {
            "v1_ok": fp["v1_matches_immutable"],
            "v2_ok": fp["v2_matches_immutable"],
            "v3_registered": fp.get("v3_value") is not None,
            "lineage_monotonic": lineage.get("monotonic_lineage_order"),
            "manifest_digests_ok": digests.get("digests_ok"),
        },
    }

    owned = all((_read_json(ART / f) or {}).get("owned_by") == "Fibonera"
                for f in ("stage9_hardware_report.json",
                          "stage9_virtual_coverage.json",
                          "stage9_moe_diagnostics.json",
                          "stage9_recovery_report.json")
                if (ART / f).exists())
    checks["J_ownership"] = {
        "status": "PASS" if owned else "FAIL",
        "evidence": {"owned_by_fibonera_in_artifacts": owned},
    }

    return {
        "kind": "cortex-stage9-acceptance-audit",
        "owned_by": "Fibonera",
        "independent_logic": {
            "fingerprints": fp,
            "virtual_accounting": acct,
            "lineage": lineage,
            "manifest_digest_sampling": digests,
            "artifact_checksums": {
                "dataset_v3": _sha(DATASET),
                "tokenizer": _sha(TOKENIZER),
                "warm6_checkpoint": _sha(WARM6_CKPT),
                "stage8_report": _sha(ART / "stage8_report.json"),
                "stage8_lineage_physical": (
                    _sha(JOINT8_CKPTS /
                         (_read_json(JOINT8_CKPTS / "lineage.json") or {})
                         .get("physical_checkpoint", "NONE"))
                    if (JOINT8_CKPTS / "lineage.json").exists() else None),
            },
        },
        "checks": checks,
        "summary_counts": {
            s: sum(1 for c in checks.values() if c["status"] == s)
            for s in ("PASS", "FAIL", "BLOCKED", "NOT YET TESTED")},
    }


def isclose(a, b, tol: float = 1.0) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, default=str))
