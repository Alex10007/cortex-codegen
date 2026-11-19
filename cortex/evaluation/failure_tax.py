"""Failure taxonomy for execution-graded code generation (stage 9).

Classifies actual pytest/process outcomes into actionable categories so
that repair rounds can target the observed failure mode instead of blind
re-sampling. Classification is derived from the real stderr/exit code of
the sandbox run - never from the candidate text alone.
"""
from __future__ import annotations

from typing import Any

CATEGORIES = ("passed", "syntax", "import", "runtime", "assertion",
              "timeout", "empty", "unknown")


def classify(grade: dict[str, Any]) -> str:
    if grade.get("passed"):
        return "passed"
    stderr = (grade.get("stderr_tail") or "")
    if grade.get("exit_code") == 124 or "timeout" in stderr.lower():
        return "timeout"
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax"
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return "import"
    if "AssertionError" in stderr or "assert" in stderr:
        return "assertion"
    if grade.get("exit_code") and grade.get("exit_code") != 0:
        return "runtime"
    if not stderr.strip():
        return "empty"
    return "unknown"


def repair_hint(category: str) -> str:
    """Category-specific generation guidance; does not leak hidden tests."""
    hints = {
        "syntax": "The previous module did not PARSE. Produce one complete, "
                  "syntactically valid Python module with no prose.",
        "import": "The previous module imported something unavailable. Use "
                  "the Python standard library only.",
        "runtime": "The previous module raised an uncaught exception at "
                   "runtime. Handle None/empty inputs defensively.",
        "assertion": "The previous module was importable but produced wrong "
                     "results. Re-read the specification and cover edge "
                     "cases (empty input, single element, case sensitivity).",
        "timeout": "The previous module exceeded the time limit. Use an "
                   "efficient algorithm.",
        "empty": "The previous attempt produced no code. Emit exactly one "
                 "Python code block defining the required functions.",
    }
    return hints.get(category, "")


def failure_breakdown(hypotheses: list[dict]) -> dict[str, int]:
    breakdown = {c: 0 for c in CATEGORIES}
    for h in hypotheses:
        c = classify(h.get("grade") or {})
        breakdown[c] += 1
    return breakdown
