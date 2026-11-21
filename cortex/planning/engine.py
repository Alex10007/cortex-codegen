"""Planning engine with deterministic metacognitive task assessment.

Repository inspection remains deterministic. The assessment policy is an
explicit heuristic control layer: it estimates ambiguity, impact and required
verification effort. It is not a learned confidence model and never claims
that heuristic scores represent calibrated model confidence.
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
_DESTRUCTIVE = ("delete", "drop", "destroy", "wipe", "remove all", "reset database")
_HIGH_IMPACT = ("production", "prod", "deploy", "publish", "release", "migration",
                "migrate", "database", "schema", "credentials", "secret", "auth",
                "permission", "security", "payment", "billing", "customer data")
_AMBIGUOUS = ("fix it", "make it better", "improve everything", "optimize it",
              "clean it up", "make it work", "build something")


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


def assess_objective(objective: str, repo_map: dict[str, Any]) -> dict[str, Any]:
    """Produce a deterministic task-risk and verification assessment.

    Scores are policy heuristics, not learned probabilities or calibrated
    confidence. The output controls planning/verification depth only.
    """
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be a non-empty string")
    text = " ".join(objective.lower().split())
    words = text.split()
    risk_flags = sorted({
        flag for terms, flag in (
            (_DESTRUCTIVE, "destructive_operation"),
            (_HIGH_IMPACT, "high_impact_surface"),
        ) if any(term in text for term in terms)
    })
    ambiguous_hits = [term for term in _AMBIGUOUS if term in text]
    conjunctions = sum(text.count(term) for term in (" and ", " then ", " plus "))
    has_target = any(token in text for token in ("/", ".", "module", "file",
                                                  "function", "class", "api",
                                                  "endpoint", "test", "repo"))
    ambiguity = 0.15
    if len(words) <= 3:
        ambiguity += 0.25
    if ambiguous_hits:
        ambiguity += min(0.35, 0.15 * len(ambiguous_hits))
    if not has_target:
        ambiguity += 0.15
    ambiguity = min(1.0, round(ambiguity, 3))

    complexity = min(1.0, round(
        0.10 + min(len(words), 40) * 0.012 + conjunctions * 0.12
        + (0.15 if any(term in text for term in
                       ("architecture", "refactor", "multi-file", "distributed",
                        "concurrent", "end-to-end", "whole repository")) else 0.0),
        3))
    uncertainty = min(1.0, round(
        0.55 * ambiguity
        + (0.20 if not repo_map.get("file_count") else 0.0)
        + (0.15 if not repo_map.get("manifests") else 0.0)
        + (0.10 if repo_map.get("file_count", 0) >= 20000 else 0.0), 3))

    destructive = "destructive_operation" in risk_flags
    high_impact = "high_impact_surface" in risk_flags
    if destructive and high_impact:
        risk_level = "critical"
    elif destructive or high_impact:
        risk_level = "high"
    elif complexity >= 0.65 or ambiguity >= 0.65:
        risk_level = "medium"
    else:
        risk_level = "low"

    score = complexity + uncertainty + (0.35 if risk_level in ("high", "critical") else 0.0)
    reasoning_depth = "deep" if score >= 1.35 else "expanded" if score >= 0.8 else "focused"
    verification = "critical" if risk_level == "critical" else (
        "deep" if risk_level == "high" or reasoning_depth == "deep" else
        "standard" if reasoning_depth == "expanded" else "focused")
    required_evidence = ["objective_acceptance_criteria", "targeted_tests",
                         "regression_tests", "diff_review"]
    if risk_level in ("high", "critical"):
        required_evidence.extend(["security_review", "rollback_plan"])
    if destructive:
        required_evidence.append("explicit_human_authorization")
    if any(term in text for term in ("performance", "latency", "throughput", "memory")):
        required_evidence.append("measured_performance_baseline")

    return {
        "policy_version": "cortex-task-assessment-v1",
        "assessment_mode": "deterministic_heuristic_not_model_confidence",
        "objective_word_count": len(words),
        "ambiguity_score": ambiguity,
        "complexity_score": complexity,
        "uncertainty_score": uncertainty,
        "risk_level": risk_level,
        "risk_flags": risk_flags,
        "reasoning_depth": reasoning_depth,
        "verification_profile": verification,
        "required_evidence": required_evidence,
        "requires_human_authorization": destructive,
        "clarification_recommended": ambiguity >= 0.75,
        "recommended_hypotheses": 3 if reasoning_depth == "deep" else (
            2 if reasoning_depth == "expanded" else 1),
        "repository_scale": (
            "large" if repo_map.get("file_count", 0) >= 5000 else
            "medium" if repo_map.get("file_count", 0) >= 500 else "small"),
        "limitations": [
            "Scores are deterministic policy heuristics, not calibrated probabilities.",
            "This assessment does not replace model reasoning, security review, tests, or human approval.",
        ],
    }


def plan(objective: str, repo_path: str, repo_map: dict[str, Any]) -> Plan:
    assessment = assess_objective(objective, repo_map)
    steps = [
        PlanStep(id="s1", action="analyze", target="repository",
                 details={"status": "complete", "file_count": repo_map["file_count"],
                          "languages": repo_map["languages"],
                          "assessment_policy": assessment["policy_version"]}),
        PlanStep(id="s2", action="implement", target=repo_map["root"],
                 details={"objective": objective,
                          "constraints": ["preserve existing architecture",
                                          "no duplicate subsystems"],
                          "reasoning_depth": assessment["reasoning_depth"],
                          "risk_level": assessment["risk_level"]}),
        PlanStep(id="s3", action="test-command", target=repo_map["root"],
                 details={"suggested": _suggest_test_command(repo_map),
                          "verification_profile": assessment["verification_profile"]}),
        PlanStep(id="s4", action="security-scan", target=repo_map["root"], details={}),
        PlanStep(id="s5", action="validate", target=objective,
                 details={"required_evidence": assessment["required_evidence"]}),
    ]
    if assessment["requires_human_authorization"]:
        steps.insert(2, PlanStep(id="s2b", action="authorized-destructive-op",
                                 target="host/remote",
                                 details={"requested_by": objective},
                                 requires_authorization=True))
    return Plan(objective=objective, repo_path=repo_path, steps=steps,
                repository_map=repo_map, assessment=assessment)


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
