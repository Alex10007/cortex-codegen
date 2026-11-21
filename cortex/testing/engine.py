"""Testing engine: run a command's test suite inside the sandbox and record
machine-verifiable results (exit code + captured output)."""
from __future__ import annotations

import sys
from typing import Any, Callable

from cortex.execution.sandbox import run


def run_test_command(argv: list[str], workspace: str,
                     timeout_s: float = 120.0,
                     audit: Callable[[str, dict], None] | None = None,
                     require_full_isolation: bool = False) -> dict[str, Any]:
    result = run(argv=argv, workspace=workspace, timeout_s=timeout_s, audit=audit,
                 require_full_isolation=require_full_isolation)
    tail = (result.stdout or "")[-4000:]
    return {
        "command": argv,
        "ok": result.ok,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "isolation": result.isolation,
        "duration_s": result.duration_s,
        "stdout_tail": tail,
        "stderr_tail": (result.stderr or "")[-4000:],
        "limits": result.limits,
    }


def run_pytest(workspace: str, targets: list[str] | None = None,
               timeout_s: float = 120.0,
               audit: Callable[[str, dict], None] | None = None,
               require_full_isolation: bool = False) -> dict[str, Any]:
    argv = [sys.executable, "-m", "pytest", "-q"]
    if targets:
        argv += targets
    report = run_test_command(argv, workspace, timeout_s=timeout_s, audit=audit,
                               require_full_isolation=require_full_isolation)
    report["summary"] = _parse_pytest_summary(report["stdout_tail"])
    return report


def _parse_pytest_summary(stdout_tail: str) -> dict[str, Any]:
    import re

    line = [l for l in stdout_tail.splitlines() if " passed" in l or "failed" in l
            or "error" in l or "no tests ran" in l]
    m = re.search(r"(\d+) passed", stdout_tail)
    f = re.search(r"(\d+) failed", stdout_tail)
    e = re.search(r"(\d+) error", stdout_tail)
    return {
        "passed": int(m.group(1)) if m else 0,
        "failed": int(f.group(1)) if f else 0,
        "errors": int(e.group(1)) if e else 0,
        "tail": line[-1] if line else "",
    }
