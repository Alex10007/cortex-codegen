"""Immutable benchmark ledger + regression detection.

The evaluation task set is fingerprinted; if a fingerprint stored from a
previous run no longer matches, evaluation results are flagged as
incomparable (benchmarks must not move under the model). Pass-rate history
is appended and regressions are detected against the best recorded score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cortex.evaluation.tasks import TASKS

DEFAULT_LEDGER = "artifacts/benchmarks.json"
DEFAULT_HISTORY = "artifacts/eval_history.jsonl"


class BenchmarkTamperedError(RuntimeError):
    pass


def fingerprint_tasks() -> str:
    payload = json.dumps(
        sorted((t.task_id, t.instruction, t.test_code, t.reference_solution)
               for t in TASKS),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_baseline(pass_at_1: float, ledger_path: str = DEFAULT_LEDGER) -> dict[str, Any]:
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    ledger = {"fingerprint": fingerprint_tasks(), "pass_at_1": pass_at_1}
    Path(ledger_path).write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def check_integrity(ledger_path: str = DEFAULT_LEDGER) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        return {"status": "no-ledger"}
    ledger = json.loads(path.read_text(encoding="utf-8"))
    current = fingerprint_tasks()
    if ledger.get("fingerprint") != current:
        raise BenchmarkTamperedError(
            "evaluation task set changed since the stored baseline; results are "
            "not comparable — establish a new benchmark version explicitly"
        )
    return {"status": "unmodified", "fingerprint": current}


def append_history(report: dict[str, Any],
                   history_path: str = DEFAULT_HISTORY) -> list[dict[str, Any]]:
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "subject": report.get("per_task", [{}])[0].get("subject", "model")
        if report.get("per_task") else "model",
        "pass_at_1": report["pass_at_1"],
        "n_tasks": report["n_tasks"],
        "fingerprint": fingerprint_tasks(),
    }
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return read_history(history_path)


def read_history(history_path: str = DEFAULT_HISTORY) -> list[dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return []
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def detect_regression(report: dict[str, Any],
                      history_path: str = DEFAULT_HISTORY,
                      ledger_path: str = DEFAULT_LEDGER) -> dict[str, Any]:
    history = read_history(history_path)
    best = max((h["pass_at_1"] for h in history
                if h["fingerprint"] == fingerprint_tasks()), default=None)
    check_integrity(ledger_path)
    current = report["pass_at_1"]
    return {
        "current_pass_at_1": current,
        "best_pass_at_1": best,
        "regressed": best is not None and current < best,
        "history_entries": len(history),
    }
