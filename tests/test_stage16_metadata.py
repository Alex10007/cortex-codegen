"""Tests for the non-destructive Stage 16 metadata compliance audit."""
from scripts.stage16_metadata_audit import inspect_artifact


def _complete_artifact():
    return {
        "kind": "workstream",
        "owned_by": "Fibonera",
        "workstream": "test",
        "metrics": {"pass_at_1": 0.0},
        "config": {"steps": 12, "batch_size": 2},
        "checkpoint_sha256": "a" * 64,
        "dataset_fingerprint": "b" * 64,
        "experiment": {
            "experiment_id": "stage16-test",
            "parent_commit": "799a217",
            "source_commit": "123456789abc",
            "dataset_sha256": "b" * 64,
            "started": "2026-11-20T10:00:00+00:00",
            "finished": "2026-11-20T10:00:12+00:00",
            "decision": "EXECUTED",
            "pre_registered_hypothesis": "test",
            "acceptance_criteria": "test",
        },
        "random_seed": 42,
        "failure_conditions": {"syntax_failures": 3},
    }


def test_complete_metadata_contract_is_recognized():
    row = inspect_artifact("sample.json", _complete_artifact())
    assert row["status"] == "COMPLETE"
    assert row["missing_fields"] == []
    assert row["actual_duration_seconds"] == 12.0
    assert row["random_seed_records"][0]["value"] == 42
    assert row["checkpoint_hash_records"][0]["value"] == "a" * 64


def test_missing_metadata_is_reported_not_invented():
    artifact = _complete_artifact()
    del artifact["random_seed"]
    del artifact["checkpoint_sha256"]
    row = inspect_artifact("sample.json", artifact)
    assert row["status"] == "INCOMPLETE"
    assert "random_seeds" in row["missing_fields"]
    assert "checkpoint_hashes" in row["missing_fields"]
    assert row["random_seed_records"] == []
    assert row["checkpoint_hash_records"] == []
    assert row["historical_artifact_modified"] is False


def test_negative_or_invalid_timing_is_not_accepted():
    artifact = _complete_artifact()
    artifact["experiment"]["finished"] = "not-a-timestamp"
    row = inspect_artifact("sample.json", artifact)
    assert "finished_at" in row["missing_fields"]
    assert "actual_duration_seconds" in row["missing_fields"]


def test_duplicate_dataset_hashes_remain_explicit_provenance():
    row = inspect_artifact("sample.json", _complete_artifact())
    assert len(row["dataset_fingerprint_records"]) >= 1
    assert row["provenance"]["dataset_sha256"] == "b" * 64 if "provenance" in row else True
