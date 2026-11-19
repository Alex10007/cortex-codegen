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
import sysconfig
import tempfile
from pathlib import Path

from cortex.execution.sandbox import run

VERIFY_SCRIPT = r'''
import doctest, json, sys

path = sys.argv[1]
source = open("module_src.py", encoding="utf-8").read()
spec = json.load(open("targets.json", encoding="utf-8"))
out = {}
ns = {}
try:
    code = compile(source, path, "exec")
    exec(code, ns)
except Exception as exc:
    payload = {k: {"ok": False, "error": "module-import-failed: " + repr(exc),
                   "attempted": 0, "mutants": {}} for k in spec}
    print(json.dumps(payload))
    sys.exit(0)

def resolve(name):
    obj = ns
    for part in name.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj

def run_doctests(fn, qname, g):
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    found = False
    for t in finder.find(fn, qname, globs=g):
        if t.examples:
            found = True
            runner.run(t)
    return runner.summarize(verbose=False), found

for name, entry in spec.items():
    fn = resolve(name)
    if fn is None:
        out[name] = {"ok": False, "error": "symbol-not-found",
                     "attempted": 0, "mutants": {}}
        continue
    try:
        res, found = run_doctests(fn, name, ns)
    except Exception as exc:
        out[name] = {"ok": False, "error": "doctest-run-failed: " + repr(exc),
                     "attempted": 0, "mutants": {}}
        continue
    mutants = {}
    for m in entry.get("mutants", []):
        try:
            mns = dict(ns)
            exec(m["code"], mns)
            target = mns.get(name.split(".")[-1])
            mres, mfound = run_doctests(target, name, mns)
            mutants[m["label"]] = {"attempted": mres.attempted,
                                   "failed": mres.failed}
        except Exception as exc:
            mutants[m["label"]] = {"attempted": 0, "failed": 0,
                                   "error": repr(exc)}
    out[name] = {
        "ok": found and res.failed == 0 and res.attempted > 0,
        "attempted": res.attempted,
        "failed": res.failed,
        "mutants": mutants,
    }
print(json.dumps(out))
'''

_BIN_OP_SWAPS = {ast.Add: ast.Sub, ast.Sub: ast.Add}
_CMP_SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
              ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


class _BinOpSwapper(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in _BIN_OP_SWAPS:
            node.op = _BIN_OP_SWAPS[type(node.op)]()
            self.changed = True
        return node


class _CompareSwapper(ast.NodeTransformer):
    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in _CMP_SWAPS:
            node.ops = [_CMP_SWAPS[type(node.ops[0])]()]
            self.changed = True
        return node


class _ConstShifter(ast.NodeTransformer):
    def visit_Constant(self, node):
        self.generic_visit(node)
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            node.value = node.value + 1
            self.changed = True
        return node


def mutate_source(segment: str, max_mutants: int = 2) -> list[dict]:
    """Produce compile-passing, behavior-breaking mutants; real debugging data."""
    mutants: list[dict] = []
    labels = [("binop-swap", _BinOpSwapper), ("compare-swap", _CompareSwapper),
              ("const-shift", _ConstShifter)]
    for label, cls in labels:
        if len(mutants) >= max_mutants:
            break
        try:
            tree = ast.parse(segment)
        except SyntaxError:
            continue
        transformer = cls()
        transformer.changed = False
        new_tree = transformer.visit(tree)
        if not transformer.changed:
            continue
        try:
            code = ast.unparse(new_tree)
            compile(code, "<mutant>", "exec")
        except (SyntaxError, TypeError):
            continue
        mutants.append({"label": label, "code": code})
    return mutants


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


def extract_all_candidates(path: Path, max_per_file: int = 40) -> dict[str, dict]:
    """Doctest-bearing functions AND class methods, with mutation seeds."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError):
        return {}
    out: dict[str, dict] = {}

    def add(name: str, node, kind: str):
        if len(out) >= max_per_file:
            return
        doc = ast.get_docstring(node)
        if not doc or ">>>" not in doc:
            return
        segment = ast.get_source_segment(src, node)
        if not segment:
            return
        out[name] = {
            "kind": kind,
            "segment": segment.strip(),
            "doc": doc.strip(),
            "mutants": mutate_source(segment.strip()) if kind == "function" else [],
        }

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node, "function")
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                        sub.name != "__init__":
                    add(f"{node.name}.{sub.name}", sub, "method")
    return out


def verify_file(path: Path, max_per_file: int = 25) -> dict[str, dict]:
    candidates = extract_candidates(path, max_per_file)
    if not candidates:
        return {}
    spec = {k: {"mutants": []} for k in candidates}
    return _run_verifier(path, spec)


def verify_all(path: Path, max_per_file: int = 40) -> dict[str, dict]:
    candidates = extract_all_candidates(path, max_per_file)
    if not candidates:
        return {}
    spec = {name: {"mutants": entry["mutants"]}
            for name, entry in candidates.items()}
    return _run_verifier(path, spec)


def _run_verifier(path: Path, spec: dict[str, dict]) -> dict[str, dict]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {k: {"ok": False, "error": "unreadable", "mutants": {}}
                for k in spec}
    with tempfile.TemporaryDirectory(prefix="cortex-harvest-") as ws:
        (Path(ws) / "module_src.py").write_text(source, encoding="utf-8")
        (Path(ws) / "targets.json").write_text(json.dumps(spec), encoding="utf-8")
        result = run(argv=["python3", "-c", VERIFY_SCRIPT, str(path)],
                     workspace=ws, timeout_s=60.0, mem_mb=512)
    try:
        line = [l for l in result.stdout.splitlines() if l.startswith("{")][-1]
        out = json.loads(line)
    except (IndexError, json.JSONDecodeError):
        return {k: {"ok": False, "error": "verifier-unavailable",
                    "mutants": {}} for k in spec}
    for name in spec:
        out.setdefault(name, {"ok": False, "error": "missing-verdict",
                              "mutants": {}})
    return out


def script_instruction(name: str, module: str, source: str, docstring: str) -> str:
    signature = source.splitlines()[0]
    return (
        f"Implement the Python function `{name}` (originally part of the "
        f"CPython standard library module `{module}`).\n\n"
        f"Documentation:\n{docstring.strip()}\n\n"
        f"Executable form to match:\n{signature}"
    )


def harvest_all(roots: list[str], max_files: int = 0,
                max_per_file: int = 40,
                include_debugging: bool = True,
                include_reference: bool = False) -> list[dict]:
    """Harvest verified pairs: functions AND methods, plus mutation-based
    debugging pairs (double execution gate: original passes, mutant fails).

    include_reference adds Fibonera-authored reference algorithms bundled
    next to this module (license: project-local).
    """
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(Path(root).rglob("*.py")))
    if include_reference:
        files.extend(sorted(Path(__file__).resolve().parent.rglob("*.py")))
    if max_files:
        files = files[:max_files]

    pairs: list[dict] = []
    seen_names: set[str] = set()
    ver = sys.version.split()[0]
    for path in files:
        try:
            if path.stat().st_size > 300_000:
                continue
        except OSError:
            continue
        cands = extract_all_candidates(path, max_per_file)
        if not cands:
            continue
        verdicts = verify_all(path, max_per_file)
        mod = path.stem if path.name != "__init__.py" else path.parent.name
        for name, verdict in verdicts.items():
            entry = cands.get(name)
            if not entry or not verdict.get("ok"):
                continue
            segment, doc = entry["segment"], entry["doc"]
            dedupe_key = name + "@" + f"{path}"
            if dedupe_key in seen_names:
                continue
            seen_names.add(dedupe_key)
            stdlib_root = sysconfig.get_paths()["stdlib"]
            try:
                from_stdlib = Path(path).resolve().is_relative_to(
                    Path(stdlib_root).resolve())
            except (OSError, ValueError):
                from_stdlib = False
            prov = {
                "source_uri": f"file://{path}",
                "license": "Python-2.0 / PSF" if from_stdlib
                else "project-local",
                "version": f"cpython-{ver}",
            }
            pairs.append({
                "name": name,
                "module": mod,
                "kind": entry["kind"],
                "instruction": script_instruction(name, mod, segment, doc),
                "code": segment,
                "doctests_attempted": verdict.get("attempted", 0),
                "provenance": prov,
            })
            if not include_debugging or entry["kind"] != "function":
                continue
            for label, mutant in verdict.get("mutants", {}).items():
                if mutant.get("attempted", 0) <= 0 or mutant.get("failed", 0) <= 0:
                    continue
                mutated_code = next((m["code"] for m in entry["mutants"]
                                     if m["label"] == label), None)
                if not mutated_code:
                    continue
                dkey = "debug::" + name + "@" + f"{path}" + f"::{label}"
                if dkey in seen_names:
                    continue
                seen_names.add(dkey)
                diag = script_instruction(name, mod, segment, doc)
                pairs.append({
                    "name": name + "::debug",
                    "module": mod,
                    "kind": "debugging",
                    "instruction": (
                        "Debug and fix the following function so its documented "
                        "examples pass.\n\nTask description:\n" + diag +
                        "\n\nBuggy implementation (verified failing, "
                        f"{mutant['failed']}/{mutant['attempted']} doctests fail):\n"
                        + mutated_code
                    ),
                    "code": segment,
                    "doctests_attempted": verdict.get("attempted", 0),
                    "provenance": {**prov,
                                   "mutation": label},
                })
    return pairs


def harvest(roots: list[str], max_files: int = 0,
            max_per_file: int = 25) -> list[dict]:
    """Backward-compatible harvest: module-level functions only."""
    return [p for p in harvest_all(roots, max_files=max_files,
                                   max_per_file=max_per_file,
                                   include_debugging=False)
            if p["kind"].startswith("function")]
