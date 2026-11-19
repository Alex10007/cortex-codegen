"""Planning engine.

The repository analysis half is fully deterministic and real: it maps files,
languages, imports, entry points, tests, manifests and CI before any change is
planned. Plan construction emits an ordered, auditable step list; destructive
steps are flagged and require explicit authorization.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cortex.interfaces import Plan, PlanStep

CODE_EXTS = {".py": "python", ".ts": "typescript", ".tsx": "typescript",
             ".js": "javascript", ".jsx": "javascript", ".go": "go",
             ".rs": "rust", ".java": "java", ".rb": "ruby", ".php": "php",
             ".cs": "csharp", ".html": "html", ".css": "css", ".sql": "sql"}
MANIFESTS = ["requirements.txt", "pyproject.toml", "package.json", "Cargo.toml",
             "go.mod", "pom.xml", "composer.json", "Gemfile"]
CI_FILES = [".github/workflows"]


def analyze_repository(repo_path: str, max_files: int = 20000) -> dict[str, Any]:
    root = Path(repo_path).resolve()
    files: list[dict[str, Any]] = []
    lang_counter: Counter[str] = Counter()
    imports: dict[str, set[str]] = {}
    entry_points: list[str] = []
    test_files: list[str] = []

    candidates = [p for p in root.rglob("*")
                  if p.is_file() and not any(part.startswith(".git") for part in p.parts)]
    for p in candidates[:max_files]:
        rel = p.relative_to(root).as_posix()
        lang = CODE_EXTS.get(p.suffix)
        if lang:
            lang_counter[lang] += 1
        record: dict[str, Any] = {"path": rel, "bytes": p.stat().st_size,
                                  "language": lang}
        if lang == "python":
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                record["parses"] = False
            else:
                record["parses"] = True
                mods = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        mods.update(a.name.split(".")[0] for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        mods.add(node.module.split(".")[0])
                    elif isinstance(node, ast.If):
                        src_ok = (isinstance(node.test, ast.Compare) and
                                  isinstance(node.test.left, ast.Name) and
                                  node.test.left.id == "__name__")
                        if src_ok and any("__main__" in ast.dump(c) for c in node.test.comparators):
                            entry_points.append(rel)
                imports[rel] = mods
        if "test" in Path(rel).stem.lower() or "/tests/" in f"/{rel}/":
            test_files.append(rel)
        files.append(record)

    manifests = [m for m in MANIFESTS if (root / m).exists()]
    ci = [c for c in CI_FILES if (root / c).exists()]

    return {
        "root": str(root),
        "file_count": len(files),
        "languages": dict(lang_counter),
        "python_imports": {k: sorted(v) for k, v in imports.items()},
        "entry_points": sorted(set(entry_points)),
        "test_files": test_files[:200],
        "manifests": manifests,
        "ci": ci,
        "files": files[:1000],
    }


def plan(objective: str, repo_path: str, repo_map: dict[str, Any]) -> Plan:
    steps = [
        PlanStep(id="s1", action="analyze", target="repository",
                 details={"status": "complete", "file_count": repo_map["file_count"],
                          "languages": repo_map["languages"]}),
        PlanStep(id="s2", action="implement", target=repo_map["root"],
                 details={"objective": objective,
                          "constraints": ["preserve existing architecture",
                                          "no duplicate subsystems"]}),
        PlanStep(id="s3", action="test-command", target=repo_map["root"],
                 details={"suggested": _suggest_test_command(repo_map)}),
        PlanStep(id="s4", action="security-scan", target=repo_map["root"], details={}),
        PlanStep(id="s5", action="validate", target=objective, details={}),
    ]
    if any(k in objective.lower() for k in ("delete", "drop", "deploy", "publish",
                                            "rewrite architecture")):
        steps.insert(2, PlanStep(id="s2b", action="authorized-destructive-op",
                                 target="host/remote", details={"requested_by": objective},
                                 requires_authorization=True))
    return Plan(objective=objective, repo_path=repo_path, steps=steps,
                repository_map=repo_map)


def _suggest_test_command(repo_map: dict[str, Any]) -> list[str]:
    if "pyproject.toml" in repo_map["manifests"] or repo_map["test_files"]:
        return ["python3", "-m", "pytest", "-q"]
    if "requirements.txt" in repo_map["manifests"]:
        return ["python3", "-m", "pytest", "-q"]
    if "package.json" in repo_map["manifests"]:
        return ["npm", "test", "--silent"]
    return []


def plan_to_json(plan: Plan) -> str:
    return json.dumps(plan.to_dict(), indent=2)
