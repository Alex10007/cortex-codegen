import json

import pytest

from cortex.evaluation import benchmarks
from cortex.evaluation.tasks import TASKS


def test_fingerprint_is_deterministic():
    assert benchmarks.fingerprint_tasks() == benchmarks.fingerprint_tasks()
    assert len(benchmarks.fingerprint_tasks()) == 64


def test_tamper_detection(tmp_path):
    ledger = tmp_path / "benchmarks.json"
    benchmarks.record_baseline(1.0, str(ledger))
    assert benchmarks.check_integrity(str(ledger))["status"] == "unmodified"
    tampered = json.loads(ledger.read_text())
    tampered["fingerprint"] = "0" * 64
    ledger.write_text(json.dumps(tampered))
    with pytest.raises(benchmarks.BenchmarkTamperedError):
        benchmarks.check_integrity(str(ledger))


def test_history_and_regression_flow(tmp_path):
    hist = tmp_path / "eval_history.jsonl"
    ledger = tmp_path / "benchmarks.json"
    benchmarks.record_baseline(0.0, str(ledger))
    report_good = {"pass_at_1": 1.0, "n_tasks": len(TASKS),
                   "per_task": [{"subject": "model", "passed": True}]}
    best = benchmarks.detect_regression(report_good, str(hist), str(ledger))
    assert best["regressed"] is False
    benchmarks.append_history(report_good, str(hist))
    report_bad = {"pass_at_1": 0.0, "n_tasks": len(TASKS),
                  "per_task": [{"subject": "model", "passed": False}]}
    worse = benchmarks.detect_regression(report_bad, str(hist), str(ledger))
    assert worse["regressed"] is True
    assert worse["best_pass_at_1"] == 1.0
