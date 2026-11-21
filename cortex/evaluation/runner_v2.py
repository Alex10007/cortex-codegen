"""Suite-v2 grader: workspace-level grading with stage metrics.

Measures, per task: compilation (each produced file parses and py_compiles),
execution (the hidden pytest process actually ran), correctness (all hidden
tests passed). pass@1 aggregates correctness over tasks.
"""
from __future__ import annotations

import ast
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from cortex.evaluation.tasks_v2 import TASKS_V2, TaskV2
from cortex.execution.sandbox import run

FENCE_RE = re.compile(r"```(?:(\S+)\n)(.*?)```", flags=re.S)


def parse_multifile(text: str, allowed: list[str]) -> dict[str, str]:
    """Extract fenced code blocks whose label matches an allowed edit target.
    Falls back to the whole text when there is exactly one target."""
    edits: dict[str, str] = {}
    for label, body in FENCE_RE.findall(text):
        if label in allowed and label not in edits:
            edits[label] = body.rstrip("\n") + "\n"
    if not edits and len(allowed) == 1:
        edits[allowed[0]] = text.rstrip("\n") + "\n"
    return edits


def _compiles(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def grade_v2(task: TaskV2, candidate_edits: dict[str, str],
             subject: str = "model") -> dict[str, Any]:
    candidate_edits = {k: v for k, v in candidate_edits.items()
                       if k in task.edit_targets}
    with tempfile.TemporaryDirectory(prefix="cortex-evalv2-") as ws:
        ws_path = Path(ws)
        for rel, content in task.base_files.items():
            target = ws_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for rel, content in candidate_edits.items():
            target = ws_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (ws_path / "test_hidden.py").write_text(task.test_code, encoding="utf-8")

        produced = list(Path(ws).glob("**/*.py"))
        compile_ok = all(_compiles(p) for p in produced)
        result = run(argv=["python3", "-m", "pytest", "-q"],
                     workspace=str(ws_path), timeout_s=task.timeout_s)
    tail = (result.stdout or "")[-1500:]
    return {
        "task_id": task.task_id,
        "tier": task.tier,
        "subject": subject,
        "compile_ok": compile_ok,
        "executed": bool(result.stdout or result.stderr) or result.exit_code != 0,
        "passed": result.ok,
        "exit_code": result.exit_code,
        "missing_files": [t for t in task.edit_targets
                          if t not in candidate_edits],
        "stdout_tail": tail,
        "stderr_tail": (result.stderr or "")[-800:],
    }


def candidate_to_edits(task: TaskV2, candidate_text: Optional[str],
                       edits: Optional[dict[str, str]]) -> dict[str, str]:
    if edits is not None:
        return edits
    assert candidate_text is not None
    return parse_multifile(candidate_text, task.edit_targets)


def evaluate_v2(report_subject: str = "model",
                reference: bool = False,
                audit: Optional[Callable[[str, dict], None]] = None,
                candidate_texts: Optional[dict[str, str]] = None,
                candidate_edits: Optional[dict[str, dict[str, str]]] = None
                ) -> dict[str, Any]:
    results = []
    for task in TASKS_V2:
        if reference:
            rec = grade_v2(task, task.reference_edits, subject="harness_sanity")
        else:
            if candidate_edits is not None and task.task_id in candidate_edits:
                edits = candidate_edits[task.task_id]
            elif candidate_texts is not None and task.task_id in candidate_texts:
                edits = candidate_to_edits(task, candidate_texts[task.task_id], None)
            else:
                edits = {}
            rec = grade_v2(task, edits, subject=report_subject)
        if audit:
            audit("eval.v2", {"task_id": rec["task_id"], "passed": rec["passed"],
                              "tier": rec["tier"], "subject": rec["subject"]})
        results.append(rec)
    tiers: dict[str, list[bool]] = {}
    for r in results:
        tiers.setdefault(r["tier"], []).append(r["passed"])
    return {
        "pass_at_1": round(sum(1 for r in results if r["passed"]) / len(results), 4)
        if results else 0.0,
        "n_tasks": len(results),
        "compile_ok_rate": round(sum(1 for r in results if r["compile_ok"])
                                 / len(results), 4) if results else 0.0,
        "executed_rate": round(sum(1 for r in results if r["executed"])
                               / len(results), 4) if results else 0.0,
        "per_tier": {k: round(sum(v) / len(v), 4) for k, v in tiers.items()},
        "per_task": results,
    }
