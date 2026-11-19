"""Harvest instruction->code pairs that are verified by ACTUAL execution.

The whole module source is executed in the sandbox once, then each candidate
function's doctests run against that real module namespace. Only functions
whose examples all pass are emitted; a missing or failing doctest never
becomes a training pair.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

from cortex.execution.sandbox import run

VERIFY_SCRIPT = r'''
import doctest, json, sys

path = sys.argv[1]
source = open("module_src.py", encoding="utf-8").read()
targets = json.load(open("targets.json", encoding="utf-8"))
out = {}
ns = {}
try:
    code = compile(source, path, "exec")
    exec(code, ns)
except Exception as exc:
    print(json.dumps({k: {"ok": False, "error": "module-import-failed: " + repr(exc),
                          "attempted": 0} for k in targets}))
    sys.exit(0)
for fname in targets:
    fn = ns.get(fname)
    if fn is None:
        out[fname] = {"ok": False, "error": "symbol-not-found", "attempted": 0}
        continue
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    found_examples = False
    try:
        for t in finder.find(fn, fname, globs=ns):
            if t.examples:
                found_examples = True
                runner.run(t)
    except Exception as exc:
        out[fname] = {"ok": False, "error": "doctest-run-failed: " + repr(exc),
                      "attempted": 0}
        continue
    res = runner.summarize(verbose=False)
    out[fname] = {
        "ok": found_examples and res.failed == 0 and res.attempted > 0,
        "attempted": res.attempted,
        "failed": res.failed,
    }
print(json.dumps(out))
'''


def extract_candidates(path: Path, max_per_file: int = 25) -> dict[str, str | None]:
    """Module-level functions with docstrings containing doctest examples."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError):
        return {}
    candidates: dict[str, str | None] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or ">>>" not in doc:
            continue
        segment = ast.get_source_segment(src, node)
        if not segment:
            continue
        candidates[node.name] = segment.strip()
        if len(candidates) >= max_per_file:
            break
    return candidates


def verify_file(path: Path, max_per_file: int = 25) -> dict[str, dict]:
    candidates = extract_candidates(path, max_per_file)
    if not candidates:
        return {}
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {k: {"ok": False, "error": "unreadable"} for k in candidates}
    with tempfile.TemporaryDirectory(prefix="cortex-harvest-") as ws:
        (Path(ws) / "module_src.py").write_text(source, encoding="utf-8")
        (Path(ws) / "targets.json").write_text(
            json.dumps(list(candidates)), encoding="utf-8")
        result = run(argv=["python3", "-c", VERIFY_SCRIPT, str(path)],
                     workspace=ws, timeout_s=30.0, mem_mb=512)
    try:
        line = [l for l in result.stdout.splitlines() if l.startswith("{")][-1]
        return json.loads(line)
    except (IndexError, json.JSONDecodeError):
        return {k: {"ok": False, "error": "verifier-unavailable"}
                for k in candidates}


def script_instruction(name: str, module: str, source: str, docstring: str) -> str:
    signature = source.splitlines()[0]
    return (
        f"Implement the Python function `{name}` (originally part of the "
        f"CPython standard library module `{module}`).\n\n"
        f"Documentation:\n{docstring.strip()}\n\n"
        f"Executable form to match:\n{signature}"
    )


def harvest(roots: list[str], max_files: int = 0,
            max_per_file: int = 25) -> list[dict]:
    """Harvest verified pairs from PSF-licensed stdlib source trees.

    max_files=0 scans the whole tree; sandbox verification is only invoked for
    files that actually contain doctest-bearing functions.
    """
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(Path(root).rglob("*.py")))
    if max_files:
        files = files[:max_files]

    pairs: list[dict] = []
    for path in files:
        try:
            if path.stat().st_size > 300_000:
                continue
        except OSError:
            continue
        candidates = extract_candidates(path, max_per_file)
        if not candidates:
            continue
        verdicts = verify_file(path, max_per_file)
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        for fname, verdict in verdicts.items():
            if not verdict.get("ok"):
                continue
            segment = candidates.get(fname)
            if not segment:
                continue
            doc = ast.get_docstring(ast.parse(segment)) or ""
            mod = path.stem if path.name != "__init__.py" else path.parent.name
            pairs.append({
                "name": fname,
                "module": mod,
                "instruction": script_instruction(fname, mod, segment, doc),
                "code": segment,
                "doctests_attempted": verdict.get("attempted", 0),
                "provenance": {
                    "source_uri": f"file://{path}",
                    "license": "Python-2.0 / PSF",
                    "version": f"cpython-{sys.version.split()[0]}",
                },
            })
    return pairs
