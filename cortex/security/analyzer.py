"""Security engine: AST-based static analysis plus secret scanning."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from cortex.interfaces import SecurityFinding

_CALL_BLOCKLIST: dict[str, tuple[str, str, str]] = {
    "eval": ("CORTEX-SEC-001", "critical", "Use of eval() executes arbitrary code"),
    "exec": ("CORTEX-SEC-002", "critical", "Use of exec() executes arbitrary code"),
    "compile": ("CORTEX-SEC-003", "high", "Dynamic compile can hide packed logic"),
}

_MEMBER_CALL_BLOCKLIST: dict[tuple[str, str], tuple[str, str, str]] = {
    ("os", "system"): ("CORTEX-SEC-004", "high", "os.system shells out"),
    ("pickle", "loads"): ("CORTEX-SEC-005", "high", "Unsafe deserialization of untrusted data"),
    ("dill", "loads"): ("CORTEX-SEC-005", "high", "Unsafe deserialization of untrusted data"),
    ("marshal", "loads"): ("CORTEX-SEC-005", "high", "Unsafe deserialization of untrusted data"),
    ("pty", "spawn"): ("CORTEX-SEC-009", "high", "Direct pseudo-terminal spawn"),
}

SUBPROCESS_FUNCS = {"run", "call", "Popen", "check_call", "check_output"}

SECRET_PATTERNS = [
    (re.compile(r"\b(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b"), "CORTEX-SEC-020",
     "API key material in source"),
    (re.compile(r"(?i)(password|passwd|secret|api_key|apikey)\s*=\s*['\"][^'\"]{8,}['\"]"),
     "CORTEX-SEC-021", "Hardcoded credential-looking literal"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "CORTEX-SEC-022",
     "Private key material embedded in source"),
]


def analyze_source(path: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        findings.append(SecurityFinding("CORTEX-SEC-099", "low", path, 1,
                                        "file does not parse as Python"))
        return _scan_secrets(path, src, findings)
    findings.extend(_walk(path, tree))
    return _scan_secrets(path, src, findings)


def _walk(path: str, tree: ast.AST) -> Iterable[SecurityFinding]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        plain_name = getattr(func, "id", None)
        attr_name = getattr(func, "attr", None)
        base_name = getattr(getattr(func, "value", None), "id", None)

        if plain_name in _CALL_BLOCKLIST:
            rid, sev, msg = _CALL_BLOCKLIST[plain_name]
            yield SecurityFinding(rid, sev, path, node.lineno, msg)

        if base_name is not None and attr_name is not None:
            key = (base_name, attr_name)
            if key in _MEMBER_CALL_BLOCKLIST:
                rid, sev, msg = _MEMBER_CALL_BLOCKLIST[key]
                yield SecurityFinding(rid, sev, path, node.lineno, msg)

            if base_name == "yaml" and attr_name == "load" and \
                    not any(kw.arg == "Loader" for kw in node.keywords):
                yield SecurityFinding("CORTEX-SEC-007", "high", path, node.lineno,
                                      "yaml.load without explicit SafeLoader")

            if base_name == "subprocess" and attr_name in SUBPROCESS_FUNCS and \
                    any(kw.arg == "shell" and getattr(kw.value, "value", None) is True
                        for kw in node.keywords):
                yield SecurityFinding("CORTEX-SEC-010", "high", path, node.lineno,
                                      "subprocess call with shell=True")


def _scan_secrets(path: str, src: str,
                  findings: list[SecurityFinding]) -> list[SecurityFinding]:
    for lineno, line in enumerate(src.splitlines(), start=1):
        for rx, rid, msg in SECRET_PATTERNS:
            if "EXAMPLE" in line or "example" in line:
                continue
            if rx.search(line):
                findings.append(SecurityFinding(rid, "critical", path, lineno, msg))
    return findings


def analyze_tree(root: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for p in Path(root).rglob("*.py"):
        if ".git" in p.parts or "node_modules" in p.parts:
            continue
        findings.extend(analyze_source(str(p)))
    return findings


def summarize(findings: list[SecurityFinding]) -> dict[str, int]:
    summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        summary[f.severity] = summary.get(f.severity, 0) + 1
    return summary
