"""Testing engine: run a command's test suite inside the sandbox and record
machine-verifiable results (exit code + captured output)."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Callable

from cortex.execution.sandbox import run



def select_regression_tests(workspace: str, impact_report: dict[str, Any],
                            max_tests: int = 200) -> dict[str, Any]:
    """Recommend conservative pytest targets from impact paths and test imports.

    This is a selection hint, not proof of complete coverage. If no test can be
    linked confidently, return the full discovered suite as the safe fallback.
    """
    if not 1 <= max_tests <= 2000:
        raise ValueError("max_tests must be between 1 and 2000")
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace must be a directory")
    changed = set(impact_report.get("changed_paths") or [])
    changed.update(
        row.get("path") for row in impact_report.get("impacted_nodes", [])
        if row.get("path"))
    target_modules = set()
    target_stems = set()
    for raw in changed:
        path = Path(raw)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(root)
            except ValueError:
                continue
        if path.suffix == ".py":
            parts = list(path.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            if parts:
                target_modules.add(".".join(parts))
                target_stems.add(parts[-1].lower())

    test_files: list[Path] = []
    excluded_dirs = {".git", ".venv", "venv", "node_modules", ".tox",
                     "__pycache__", "dist", "build", ".mypy_cache",
                     ".pytest_cache", ".cortex-state"}
    discovery_truncated = False
    for current, dirs, filenames in os.walk(root):
        dirs[:] = sorted(d for d in dirs
                         if d not in excluded_dirs and not d.startswith("."))
        for filename in filenames:
            if ((filename.startswith("test_") and filename.endswith(".py"))
                    or filename.endswith("_test.py")):
                test_files.append(Path(current) / filename)
                if len(test_files) >= 10000:
                    discovery_truncated = True
                    break
        if discovery_truncated:
            break
    test_files.sort()
    selected: set[Path] = set()
    parse_failures: list[str] = []
    for test_path in test_files:
        rel = test_path.relative_to(root).as_posix()
        try:
            tree = ast.parse(test_path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            parse_failures.append(rel)
            continue
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        if any(module == target or module.startswith(target + ".")
               or target.startswith(module + ".")
               for module in imported for target in target_modules):
            selected.add(test_path)
            continue
        stem = test_path.stem.lower()
        if any(stem in {"test_" + name, name + "_test"} for name in target_stems):
            selected.add(test_path)

    # A test that cannot be parsed is not safe to omit from a targeted set.
    selected.update(root / rel for rel in parse_failures)
    all_tests = [p.relative_to(root).as_posix() for p in test_files]
    chosen = sorted(p.relative_to(root).as_posix() for p in selected)
    if selected and not discovery_truncated and len(chosen) <= max_tests:
        return {
            "kind": "cortex-regression-test-selection",
            "mode": "targeted_static_recommendation",
            "targets": chosen,
            "candidate_test_count": len(all_tests),
            "selected_test_count": len(chosen),
            "truncated": False,
            "parse_failures": parse_failures,
            "fallback_to_full_suite": False,
            "limitations": [
                "Static import/name matching can miss dynamic imports and indirect runtime dependencies.",
                "Unparseable test files are included conservatively in the selected set.",
                "Run the full suite before release unless a separately qualified coverage policy permits otherwise.",
            ],
        }
    return {
        "kind": "cortex-regression-test-selection",
        "mode": "full_suite_fallback",
        "targets": all_tests if len(all_tests) <= max_tests and not discovery_truncated else [],
        "full_suite_command": ["python", "-m", "pytest", "-q"],
        "candidate_test_count": len(all_tests),
        "selected_test_count": len(all_tests) if len(all_tests) <= max_tests else 0,
        "truncated": discovery_truncated or len(all_tests) > max_tests,
        "parse_failures": parse_failures,
        "fallback_to_full_suite": True,
        "limitations": [
            "No complete, confident bounded test selection was possible.",
            "Execute the full pytest suite; an empty targets list means do not pass target files to pytest.",
            "Static import/name matching is not proof of complete test coverage.",
        ],
    }



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
