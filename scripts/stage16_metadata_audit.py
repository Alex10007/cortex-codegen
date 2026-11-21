"""Supplemental, non-destructive Stage 16 artifact metadata audit.

This audit never edits historical Stage 16 experiment artifacts. It reports
missing provenance explicitly rather than inventing seeds, configurations,
checkpoint hashes, or execution timings.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
EXPERIMENT_ARTIFACTS = (
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
)
SUMMARY_ARTIFACTS = (
    "stage16_report.json",
    "stage16_independent_audit.json",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield child_path, key, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield from _walk(child, child_path)


def _timestamp(value: Any):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def inspect_artifact(name: str, artifact: dict) -> dict:
    exp = artifact.get("experiment") or {}
    rows = list(_walk(artifact))
    configs = [
        {"path": path, "value": value}
        for path, key, value in rows
        if key.lower() in {"config", "configuration", "decoding",
                           "training_config"}
        and isinstance(value, dict)
    ]
    seeds = [
        {"path": path, "value": value}
        for path, key, value in rows
        if "seed" in key.lower()
        and isinstance(value, (int, float, str))
        and not isinstance(value, bool)
    ]
    checkpoint_hashes = [
        {"path": path, "value": value}
        for path, key, value in rows
        if "checkpoint" in path.lower()
        and ("sha256" in key.lower() or "hash" in key.lower())
        and isinstance(value, str) and HEX64.fullmatch(value)
    ]
    dataset_fingerprints = [
        {"path": path, "value": value}
        for path, key, value in rows
        if ("fingerprint" in key.lower() or "dataset_sha256" in key.lower())
        and isinstance(value, str) and HEX64.fullmatch(value)
    ]
    started = exp.get("started") or exp.get("started_at")
    finished = exp.get("finished") or exp.get("finished_at")
    start_dt, finish_dt = _timestamp(started), _timestamp(finished)
    duration = None
    if start_dt is not None and finish_dt is not None:
        duration = (finish_dt - start_dt).total_seconds()

    failure_records = [
        {"path": path, "value": value}
        for path, key, value in rows
        if any(term in key.lower() for term in
               ("failure", "failures", "skip_reasons", "blocked",
                "regression", "error_class", "rejected"))
    ]
    raw_metrics_present = any(
        key not in {"kind", "owned_by", "workstream", "experiment"}
        for key in artifact
    )
    decision = exp.get("decision") or artifact.get("promotion_decision")
    provenance = {
        "parent_commit": exp.get("parent_commit") or artifact.get("parent_commit"),
        "source_commit": exp.get("source_commit"),
        "dataset_sha256": exp.get("dataset_sha256"),
    }

    fields = {
        "configuration": bool(configs),
        "dataset_fingerprints": bool(dataset_fingerprints),
        "checkpoint_hashes": bool(checkpoint_hashes),
        "random_seeds": bool(seeds),
        "started_at": start_dt is not None,
        "finished_at": finish_dt is not None,
        "actual_duration_seconds": duration is not None and duration >= 0,
        "raw_metrics": raw_metrics_present,
        "decision": bool(decision),
        "failure_conditions": bool(failure_records),
        "provenance": all(provenance.values()),
    }
    missing = [key for key, present in fields.items() if not present]
    return {
        "artifact": name,
        "artifact_kind": artifact.get("kind"),
        "experiment_id": exp.get("experiment_id"),
        "parent_commit": provenance["parent_commit"],
        "source_commit": provenance["source_commit"],
        "dataset_sha256": provenance["dataset_sha256"],
        "started_at": started,
        "finished_at": finished,
        "actual_duration_seconds": duration,
        "decision": decision,
        "configuration_records": configs,
        "random_seed_records": seeds,
        "checkpoint_hash_records": checkpoint_hashes,
        "dataset_fingerprint_records": dataset_fingerprints,
        "failure_condition_records": failure_records,
        "field_status": fields,
        "missing_fields": missing,
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "historical_artifact_modified": False,
    }


def build_report() -> dict:
    inspected = []
    missing_files = []
    for name in EXPERIMENT_ARTIFACTS:
        path = ART / name
        if not path.is_file():
            missing_files.append(name)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            inspected.append({
                "artifact": name, "status": "UNREADABLE",
                "error": str(exc), "missing_fields": ["valid_json"],
                "historical_artifact_modified": False,
            })
            continue
        inspected.append(inspect_artifact(name, data))

    summaries = []
    for name in SUMMARY_ARTIFACTS:
        path = ART / name
        if not path.is_file():
            missing_files.append(name)
            summaries.append({"artifact": name, "status": "MISSING"})
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({
                "artifact": name,
                "status": "AGGREGATE_SUMMARY",
                "owned_by": data.get("owned_by"),
                "stage": data.get("stage"),
                "parent_commit": data.get("parent_commit"),
                "historical_artifact_modified": False,
            })
        except (OSError, json.JSONDecodeError) as exc:
            summaries.append({"artifact": name, "status": "UNREADABLE",
                              "error": str(exc)})

    complete = sum(row.get("status") == "COMPLETE" for row in inspected)
    incomplete = sum(row.get("status") == "INCOMPLETE" for row in inspected)
    unreadable = sum(row.get("status") == "UNREADABLE" for row in inspected)
    return {
        "kind": "cortex-stage16-metadata-compliance-audit",
        "owned_by": "Fibonera",
        "stage": 16,
        "schema_version": "1.0",
        "policy": "Historical artifacts are read-only; absent provenance is reported, never inferred.",
        "required_fields": [
            "configuration", "dataset_fingerprints", "checkpoint_hashes",
            "random_seeds", "started_at", "finished_at",
            "actual_duration_seconds", "raw_metrics", "decision",
            "failure_conditions", "provenance",
        ],
        "experiment_artifacts_expected": len(EXPERIMENT_ARTIFACTS),
        "experiment_artifacts_inspected": len(inspected),
        "complete": complete,
        "incomplete": incomplete,
        "unreadable": unreadable,
        "missing_files": missing_files,
        "metadata_complete": (
            complete == len(EXPERIMENT_ARTIFACTS)
            and incomplete == 0 and unreadable == 0
            and not missing_files
        ),
        "experiment_artifacts": inspected,
        "summary_artifacts": summaries,
        "capability_decision_unchanged": "NOT PROMOTED",
        "stage12_champion_preservation_required": True,
        "no_historical_artifacts_rewritten": True,
    }


def main() -> int:
    report = build_report()
    out = ART / "stage16_metadata_compliance.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "complete": report["complete"],
        "incomplete": report["incomplete"],
        "unreadable": report["unreadable"],
        "missing_files": report["missing_files"],
        "metadata_complete": report["metadata_complete"],
        "output": str(out),
    }, indent=2))
    return 0 if report["metadata_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
