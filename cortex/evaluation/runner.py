"""Evaluation runner: grades code by executing hidden pytest suites in the sandbox.

Two distinct result labels are reported:
- `harness_sanity`: reference solutions pass their own tests (verifies the
  harness itself works);
- `model`: candidate code from the language model (whatever the real score is).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from cortex.evaluation.tasks import TASKS, EvalTask
from cortex.execution.sandbox import run


def grade_single(task: EvalTask, candidate_code: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cortex-eval-") as ws:
        ws_path = Path(ws)
        (ws_path / task.module_name).write_text(candidate_code, encoding="utf-8")
        (ws_path / "test_task.py").write_text(task.test_code, encoding="utf-8")
        result = run(
            argv=[sys.executable, "-m", "pytest", "-q"],
            workspace=str(ws_path),
            timeout_s=task.timeout_s,
        )
    return {
        "task_id": task.task_id,
        "passed": result.ok,
        "exit_code": result.exit_code,
        "stdout_tail": (result.stdout or "")[-800:],
        "stderr_tail": (result.stderr or "")[-800:],
    }


def evaluate(candidate: Optional[str] = None, reference: bool = False,
             audit: Optional[Callable[[str, dict], None]] = None) -> dict[str, Any]:
    results = []
    for task in TASKS:
        if reference:
            if not task.reference_solution:
                continue
            rec = grade_single(task, task.reference_solution)
            rec["subject"] = "harness_sanity"
        else:
            if not candidate:
                raise ValueError("candidate code required unless reference=True")
            rec = grade_single(task, candidate)
            rec["subject"] = "model"
        if audit:
            audit("eval.task", {"task_id": rec["task_id"], "passed": rec["passed"],
                                "subject": rec["subject"]})
        results.append(rec)
    passes = sum(1 for r in results if r["passed"])
    return {
        "pass_at_1": round(passes / len(results), 4) if results else 0.0,
        "per_task": results,
        "n_tasks": len(results),
    }
