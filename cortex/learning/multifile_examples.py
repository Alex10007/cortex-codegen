"""Multi-file verified repo examples (Fibonera-authored, project-local).

Small packages with real cross-module imports; every package is verified by
executing cross-file doctests in the sandbox before it can become a training
record. Records use the same fenced-per-file format graded by
cortex.evaluation.runner_v2.parse_multifile.
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from cortex.execution.sandbox import run

PACKAGES: list[dict[str, Any]] = [
    {
        "name": "cortexpkg_statutils",
        "files": {
            "cortexpkg_statutils/__init__.py": "",
            "cortexpkg_statutils/valid.py": (
                "def is_sorted(items):\n"
                "    \"\"\"Return True when items is non-descending.\n"
                "\n"
                "    >>> is_sorted([1, 2, 2, 3])\n"
                "    True\n"
                "    >>> is_sorted([2, 1])\n"
                "    False\n"
                "    \"\"\"\n"
                "    return all(items[i] <= items[i + 1]\n"
                "                for i in range(len(items) - 1))\n"
            ),
            "cortexpkg_statutils/rank.py": (
                "from .valid import is_sorted\n"
                "\n"
                "\n"
                "def percentile_sorted(sorted_items, pct):\n"
                "    \"\"\"Nearest percentile of a SORTED list (interpolated).\n"
                "\n"
                "    >>> percentile_sorted([10, 20, 30, 40], 50)\n"
                "    30\n"
                "    >>> percentile_sorted([10, 20], 0)\n"
                "    10\n"
                "    \"\"\"\n"
                "    if not is_sorted(sorted_items):\n"
                "        raise ValueError(\"input must be sorted\")\n"
                "    idx = int(round(pct / 100 * (len(sorted_items) - 1)))\n"
                "    return sorted_items[idx]\n"
            ),
        },
        "targets": ["cortexpkg_statutils.valid.is_sorted",
                    "cortexpkg_statutils.rank.percentile_sorted"],
        "instruction": (
            "Create a Python package `cortexpkg_statutils` with two modules: "
            "`valid.py` exposing `is_sorted(items)` that returns True when the "
            "list is non-descending, and `rank.py` exposing "
            "`percentile_sorted(sorted_items, pct)` that imports `is_sorted` "
            "from the sibling module, raises ValueError on unsorted input, and "
            "otherwise returns the nearest percentile element (interpolated "
            "index = round(pct/100*(n-1))). Follow the documented examples in "
            "each module. Output one fenced code block per file, labeled with "
            "the file path."
        ),
    },
    {
        "name": "cortexpkg_textools",
        "files": {
            "cortexpkg_textools/__init__.py": "",
            "cortexpkg_textools/clean.py": (
                "def normalize_ws(text):\n"
                "    \"\"\"Collapse every whitespace run to one space and strip ends.\n"
                "\n"
                "    >>> normalize_ws('a  b   c '  )\n"
                "    'a b c'\n"
                "    >>> normalize_ws('  hi ')\n"
                "    'hi'\n"
                "    \"\"\"\n"
                "    return \" \".join(text.split())\n"
            ),
            "cortexpkg_textools/count.py": (
                "from .clean import normalize_ws\n"
                "\n"
                "\n"
                "def word_frequency(text):\n"
                "    \"\"\"Word frequency over whitespace-normalized text.\n"
                "\n"
                "    >>> word_frequency('hi hi yo') == {'hi': 2, 'yo': 1}\n"
                "    True\n"
                "    >>> word_frequency('  ')\n"
                "    {}\n"
                "    \"\"\"\n"
                "    words = normalize_ws(text).split(\" \")\n"
                "    freq = {}\n"
                "    for w in words:\n"
                "        if w:\n"
                "            freq[w] = freq.get(w, 0) + 1\n"
                "    return freq\n"
            ),
        },
        "targets": ["cortexpkg_textools.clean.normalize_ws",
                    "cortexpkg_textools.count.word_frequency"],
        "instruction": (
            "Create a Python package `cortexpkg_textools` with two modules: "
            "`clean.py` exposing `normalize_ws(text)` that collapses whitespace "
            "runs to single spaces and strips the ends, and `count.py` exposing "
            "`word_frequency(text)` that imports `normalize_ws` from the sibling "
            "module and counts words over the normalized text. Output one "
            "fenced code block per file, labeled with the file path."
        ),
    },
    {
        "name": "cortexpkg_shapes",
        "files": {
            "cortexpkg_shapes/__init__.py": "",
            "cortexpkg_shapes/geom.py": (
                "def hypotenuse(a, b):\n"
                "    \"\"\"Length of the hypotenuse of a right triangle.\n"
                "\n"
                "    >>> round(hypotenuse(3.0, 4.0), 6)\n"
                "    5.0\n"
                "    \"\"\"\n"
                "    return (a * a + b * b) ** 0.5\n"
            ),
            "cortexpkg_shapes/units.py": (
                "from .geom import hypotenuse\n"
                "\n"
                "\n"
                "def diagonal_width(w, h):\n"
                "    \"\"\"Diagonal of a w-by-h rectangle, rounded to 2 decimals.\n"
                "\n"
                "    >>> diagonal_width(6, 8)\n"
                "    10.0\n"
                "    >>> diagonal_width(1, 1)\n"
                "    1.41\n"
                "    \"\"\"\n"
                "    return round(hypotenuse(float(w), float(h)), 2)\n"
            ),
        },
        "targets": ["cortexpkg_shapes.geom.hypotenuse",
                    "cortexpkg_shapes.units.diagonal_width"],
        "instruction": (
            "Create a Python package `cortexpkg_shapes` with two modules: "
            "`geom.py` exposing `hypotenuse(a, b)`, and `units.py` exposing "
            "`diagonal_width(w, h)` that imports `hypotenuse` from the sibling "
            "module and returns the rectangle diagonal rounded to 2 decimals. "
            "Output one fenced code block per file, labeled with the file path."
        ),
    },
    {
        "name": "cortexpkg_queueops",
        "files": {
            "cortexpkg_queueops/__init__.py": "",
            "cortexpkg_queueops/priority.py": (
                "def prioritize(tasks):\n"
                "    \"\"\"Stable sort (priority, label) pairs by priority asc.\n"
                "\n"
                "    >>> prioritize([(2, 'b'), (1, 'a'), (2, 'a')])\n"
                "    [(1, 'a'), (2, 'b'), (2, 'a')]\n"
                "    \"\"\"\n"
                "    return sorted(tasks, key=lambda t: t[0])\n"
            ),
            "cortexpkg_queueops/runner.py": (
                "from .priority import prioritize\n"
                "\n"
                "\n"
                "def execution_order(tasks):\n"
                "    \"\"\"Labels only, in execution order.\n"
                "\n"
                "    >>> execution_order([(2, 'b'), (1, 'a')])\n"
                "    ['a', 'b']\n"
                "    >>> execution_order([])\n"
                "    []\n"
                "    \"\"\"\n"
                "    return [label for _, label in prioritize(tasks)]\n"
            ),
        },
        "targets": ["cortexpkg_queueops.priority.prioritize",
                    "cortexpkg_queueops.runner.execution_order"],
        "instruction": (
            "Create a Python package `cortexpkg_queueops` with two modules: "
            "`priority.py` exposing `prioritize(tasks)` that stable-sorts "
            "(priority, label) pairs by priority ascending, and `runner.py` "
            "exposing `execution_order(tasks)` that imports `prioritize` from "
            "the sibling module and returns just the labels in order. Output "
            "one fenced code block per file, labeled with the file path."
        ),
    },
]


def format_examples(examples: list[dict[str, Any]]) -> str:
    """Fenced per-file record body identical to parse_multifile input format."""
    blocks = []
    for path, content in sorted(examples["files"].items()):
        body = textwrap.dedent(content).rstrip("\n")
        blocks.append(f"```{path}\n{body}\n```")
    return "\n".join(blocks)


def verify_package(pkg: dict[str, Any]) -> dict[str, Any]:
    """Execute the package's cross-module doctests in the sandbox."""
    import json as _json
    files_json = _json.dumps({p: c for p, c in pkg["files"].items()
                              if p != "__init__.py"}, sort_keys=True)
    targets_json = _json.dumps({"targets": pkg["targets"]}, sort_keys=True)
    workdir = tempfile.mkdtemp(prefix="cortex-pkgverify-")
    script = textwrap.dedent('''
        import doctest
        import importlib
        import json
        from pathlib import Path

        files = json.loads(open("files.json", encoding="utf-8").read())
        spec = json.loads(open("targets.json", encoding="utf-8").read())
        for rel, content in files.items():
            p = Path(rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        out = {}
        for dotted in spec["targets"]:
            mod_name, attr = dotted.rsplit(".", 1)
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, attr)
                if fn is None:
                    raise AttributeError(attr)
            except Exception as exc:
                out[dotted] = {"ok": False, "error": repr(exc),
                               "attempted": 0, "failed": 0}
                continue
            try:
                finder = doctest.DocTestFinder()
                runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
                for t in finder.find(fn, dotted, globs=vars(mod)):
                    if t.examples:
                        runner.run(t)
                s = runner.summarize(verbose=False)
                out[dotted] = {"ok": s.attempted > 0 and s.failed == 0,
                               "attempted": s.attempted, "failed": s.failed}
            except Exception as exc:
                out[dotted] = {"ok": False, "error": repr(exc),
                               "attempted": 0, "failed": 0}
        print(json.dumps(out))
    ''')
    (Path(workdir) / "verify_pkg.py").write_text(script, encoding="utf-8")
    (Path(workdir) / "files.json").write_text(files_json, encoding="utf-8")
    (Path(workdir) / "targets.json").write_text(targets_json, encoding="utf-8")
    result = run(argv=["python3", "verify_pkg.py"], workspace=workdir,
                 timeout_s=60)
    try:
        last = (result.stdout or "").strip().splitlines()[-1]
        return json.loads(last)
    except (ValueError, IndexError):
        return {"error": (result.stderr or result.stdout or "")[-400:]}


def build_multifile_pairs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verified multi-file repo examples; only verified ones are emitted."""
    pairs: list[dict[str, Any]] = []
    verify_log: dict[str, Any] = {}
    for pkg in PACKAGES:
        verdict = verify_package(pkg)
        verify_log[pkg["name"]] = verdict
        if any(not v.get("ok") for v in verdict.values()):
            continue
        pairs.append({
            "name": "repo::" + pkg["name"],
            "module": pkg["name"],
            "kind": "multi-file",
            "instruction": pkg["instruction"],
            "code": format_examples(pkg),
            "doctests_attempted": sum(v.get("attempted", 0)
                                      for v in verdict.values()),
            "provenance": {"source_uri": "cortex://multifile_examples/"
                                         + pkg["name"],
                           "license": "project-local",
                           "version": "cortex-stage4-1"},
        })
    return pairs, verify_log
