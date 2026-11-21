"""Engineering pipeline: analyze -> plan -> (authorized edits) -> test ->
security scan -> verification record.

The pipeline never marks code as verified unless it actually executed the
test command in the sandbox and it passed. `patcher` is the code-change
mechanism injected by the caller (the model's edit step in production); with
no patcher, the pipeline still validates the current repository state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from cortex.monitoring.audit import AuditLog
from cortex.planning.engine import analyze_repository, plan
from cortex.security.analyzer import analyze_tree, summarize
from cortex.testing.engine import run_pytest

Patcher = Callable[[Any], dict[str, str]]


def run_engineering_pipeline(objective: str, repo_path: str,
                             patcher: Optional[Patcher] = None,
                             authorize_destructive: bool = False,
                             audit: AuditLog | None = None,
                             test_command_top: list[str] | None = None) -> dict[str, Any]:
    audit = audit or AuditLog.default()

    repo_map = analyze_repository(repo_path)
    the_plan = plan(objective, repo_path, repo_map)
    audit.record("pipeline.plan", {"objective": objective,
                                   "steps": [s.id for s in the_plan.steps]})

    destructive_needed = any(s.requires_authorization for s in the_plan.steps)
    if destructive_needed and not authorize_destructive:
        report: dict[str, Any] = {
            "objective": objective,
            "repo": repo_path,
            "status": "blocked",
            "reason": "plan includes destructive operations; explicit authorization required",
            "plan": the_plan.to_dict(),
        }
        audit.record("pipeline.blocked", {"reason": report["reason"]})
        return report

    files_changed: list[str] = []
    if patcher is not None:
        edits = patcher(the_plan) or {}
        for rel, content in edits.items():
            target = Path(repo_path) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            files_changed.append(rel)
        audit.record("pipeline.apply_edits", {"files": files_changed})

    if test_command_top:
        from cortex.testing.engine import run_test_command
        test_report = run_test_command(test_command_top, repo_path, audit=audit.record)
    else:
        test_report = run_pytest(repo_path, timeout_s=180.0, audit=audit.record)

    findings = analyze_tree(repo_path)
    sec_summary = summarize(findings)

    tests_ok = bool(test_report and test_report.get("ok"))
    verified = tests_ok and sec_summary["critical"] == 0

    report = {
        "objective": objective,
        "repo": repo_path,
        "status": "verified" if verified else ("tests-failed" if test_report
                                               else "insecure-or-untested"),
        "plan": the_plan.to_dict(),
        "files_changed": files_changed,
        "tests": test_report,
        "security": {"summary": sec_summary,
                     "findings": [f.__dict__ for f in findings][:50]},
        "verified": verified,
    }
    audit.record("pipeline.result", {"objective": objective,
                                     "status": report["status"],
                                     "verified": verified})
    return report
