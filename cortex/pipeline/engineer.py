"""Engineering pipeline: analyze -> plan -> (authorized edits) -> test ->
security scan -> verification record.

The pipeline never marks code as verified unless it actually executed the
test command in the sandbox and it passed. `patcher` is the code-change
mechanism injected by the caller (the model's edit step in production); with
no patcher, the pipeline still validates the current repository state.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from cortex.monitoring.audit import AuditLog
from cortex.planning.engine import analyze_repository, plan
from cortex.security.analyzer import analyze_tree, summarize
from cortex.testing.engine import run_pytest

Patcher = Callable[[Any], dict[str, str]]
ObjectiveVerifier = Callable[[Any, dict[str, Any]], bool]


def _apply_patch_set(repo_path: str, edits: dict[str, str]) -> list[str]:
    """Validate and stage an entire patch set before publishing any file.

    Replacements are atomic per file. If a later replacement fails, already
    replaced files are restored from their original bytes (or removed if new).
    This is rollback-capable, not a filesystem-wide atomic transaction.
    """
    root = Path(repo_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository root must be an existing directory")

    prepared: list[tuple[str, Path, str, bytes | None, int | None]] = []
    seen: set[Path] = set()
    for rel, content in edits.items():
        if not isinstance(rel, str) or not rel:
            raise ValueError("patch path must be a non-empty string")
        rel_path = Path(rel)
        if (rel_path.is_absolute() or ".." in rel_path.parts
                or any(part in {".git", ".cortex-state"} for part in rel_path.parts)):
            raise ValueError(f"patcher returned a protected path: {rel}")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 1_000_000:
            raise ValueError(f"patch content invalid or exceeds 1MB: {rel}")

        target = root / rel_path
        # Reject symlink components rather than allowing aliases or races
        # through an existing repository symlink.
        cursor = root
        for part in rel_path.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"patcher path traverses a symlink: {rel}")
        resolved = target.resolve()
        if resolved == root or root not in resolved.parents:
            raise ValueError(f"patcher path escapes repository: {rel}")
        if resolved in seen:
            raise ValueError(f"duplicate patch target: {rel}")
        seen.add(resolved)
        if target.exists() and not target.is_file():
            raise ValueError(f"patch target is not a regular file: {rel}")
        original = target.read_bytes() if target.exists() else None
        mode = target.stat().st_mode & 0o777 if target.exists() else None
        prepared.append((rel_path.as_posix(), target, content, original, mode))

    staged: list[tuple[str, Path, Path, bytes | None, int | None]] = []
    published: list[tuple[Path, bytes | None, int | None]] = []
    try:
        # Stage every replacement in the target directory so os.replace stays
        # on the same filesystem and is atomic for each individual file.
        for rel, target, content, original, mode in prepared:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=".cortex-stage-", dir=target.parent)
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                if mode is not None:
                    os.chmod(temp_path, mode)
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
            staged.append((rel, target, temp_path, original, mode))

        for _rel, target, temp_path, original, mode in staged:
            os.replace(temp_path, target)
            published.append((target, original, mode))
    except BaseException:
        rollback_errors: list[str] = []
        for target, original, mode in reversed(published):
            try:
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    fd, restore_name = tempfile.mkstemp(
                        prefix=".cortex-restore-", dir=target.parent)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(original)
                        stream.flush()
                        os.fsync(stream.fileno())
                    restore_path = Path(restore_name)
                    if mode is not None:
                        os.chmod(restore_path, mode)
                    os.replace(restore_path, target)
            except OSError as exc:
                rollback_errors.append(f"{target}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "patch publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors))
        raise
    finally:
        for _rel, _target, temp_path, _original, _mode in staged:
            temp_path.unlink(missing_ok=True)

    return [rel for rel, _target, _temp, _original, _mode in staged]


def run_engineering_pipeline(objective: str, repo_path: str,
                             patcher: Optional[Patcher] = None,
                             authorize_destructive: bool = False,
                             audit: AuditLog | None = None,
                             test_command_top: list[str] | None = None,
                             objective_verifier: ObjectiveVerifier | None = None
                             ) -> dict[str, Any]:
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
        if not isinstance(edits, dict):
            raise ValueError("patcher must return a dictionary of path-to-content edits")
        files_changed = _apply_patch_set(repo_path, edits)
        audit.record("pipeline.apply_edits", {"files": files_changed})

    if test_command_top:
        from cortex.testing.engine import run_test_command
        test_report = run_test_command(test_command_top, repo_path, audit=audit.record)
    else:
        test_report = run_pytest(repo_path, timeout_s=180.0, audit=audit.record)

    findings = analyze_tree(repo_path)
    sec_summary = summarize(findings)

    tests_ok = bool(test_report and test_report.get("ok"))
    implementation_performed = bool(files_changed)
    validation_passed = tests_ok and sec_summary["critical"] == 0
    # Passing the existing repository test suite is baseline validation, not
    # evidence that the requested objective was implemented.
    objective_verified = bool(
        objective_verifier and objective_verifier(the_plan, test_report or {}))
    verified = (validation_passed and implementation_performed
                and objective_verified)
    status = ("verified" if verified else
              "baseline-validated-no-implementation"
              if validation_passed and not implementation_performed else
              "changes-tested-not-objective-verified"
              if validation_passed and implementation_performed else
              "tests-failed" if test_report else "insecure-or-untested")

    report = {
        "objective": objective,
        "repo": repo_path,
        "status": status,
        "plan": the_plan.to_dict(),
        "files_changed": files_changed,
        "tests": test_report,
        "security": {"summary": sec_summary,
                     "findings": [f.__dict__ for f in findings][:50]},
        "verified": verified,
        "implementation_performed": implementation_performed,
        "validation_passed": validation_passed,
        "objective_verified": objective_verified,
    }
    audit.record("pipeline.result", {"objective": objective,
                                     "status": report["status"],
                                     "verified": verified,
                                     "implementation_performed": implementation_performed,
                                     "validation_passed": validation_passed,
                                     "objective_verified": objective_verified})
    return report
