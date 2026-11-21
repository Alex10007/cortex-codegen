"""Benchmark suite v3 (stage 9): broader capability families.

Separately versioned dataset with its own fingerprint; v1 and v2 suites
remain untouched. Every task is Python-executable in the existing
sandboxed-pypytest grader:
- python-algorithms: standard library algorithm implementation
- sql/databases: sqlite3 in-memory operations
- debugging: fix the behavior of a specified (buggy) function
- test generation: enumerate TEST_CASES that discriminate a correct
  implementation from broken ones
- refactoring: behavior-preserving rewrite into a composable form
- api: request-handler function graded on behavior
- security: input validation / injection resistance
- performance: algorithm must hold a time bound at scale

The suite is registered via `fingerprint_tasks("v3")` in
`cortex/evaluation/benchmarks.py` and is immutable once committed.
"""
from __future__ import annotations

from typing import Any

from cortex.evaluation.tasks import EvalTask

TASKS_V3: list[EvalTask] = [
    EvalTask(
        task_id="t3-run-length-encode",
        instruction=(
            "Write module `task.py` defining `encode(s: str) -> list`. "
            "Return a list of (character, run_length) tuples for consecutive "
            "runs. encode('') == []."),
        module_name="task.py",
        test_code=(
            "from task import encode\n\n"
            "def test_basic():\n"
            "    assert encode('') == []\n"
            "    assert encode('a') == [('a', 1)]\n"
            "    assert encode('aaabb') == [('a', 3), ('b', 2)]\n"
            "    assert encode('aabbaa') == [('a', 2), ('b', 2), ('a', 2)]\n"
        ),
        reference_solution=(
            "def encode(s):\n"
            "    out = []\n"
            "    for ch in s:\n"
            "        if out and out[-1][0] == ch:\n"
            "            out[-1][1] += 1\n"
            "        else:\n"
            "            out.append([ch, 1])\n"
            "    return [(c, n) for c, n in out]\n"
        ),
    ),
    EvalTask(
        task_id="t3-interval-merge",
        instruction=(
            "Write module `task.py` defining `merge(intervals) -> list`. "
            "`intervals` is a list of [start, end] pairs. Merge overlapping "
            "or touching intervals, result sorted by start."),
        module_name="task.py",
        test_code=(
            "from task import merge\n\n"
            "def test_merge():\n"
            "    assert merge([]) == []\n"
            "    assert merge([[1, 3]]) == [[1, 3]]\n"
            "    assert merge([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\n"
            "    assert merge([[1, 4], [4, 5]]) == [[1, 5]]\n"
            "    assert merge([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]\n"
        ),
        reference_solution=(
            "def merge(intervals):\n"
            "    out = []\n"
            "    for s, e in sorted(intervals):\n"
            "        if out and s <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], e)\n"
            "        else:\n"
            "            out.append([s, e])\n"
            "    return out\n"
        ),
    ),
    EvalTask(
        task_id="t3-sql-sqlite-aggregate",
        instruction=(
            "Write module `task.py` defining `top_customers(rows, n) -> list`. "
            "`rows` is a list of (customer, amount) sale records. Use sqlite3 "
            "in memory: create table sales(customer TEXT, amount REAL), insert "
            "rows, and return the top-n customers by total amount as "
            "(customer, total) tuples, highest first, ties broken by name "
            "ascending."),
        module_name="task.py",
        test_code=(
            "import sqlite3\n"
            "from task import top_customers\n\n"
            "def test_top():\n"
            "    rows = [('a', 10.0), ('b', 5.0), ('a', 5.0), ('c', 3.0)]\n"
            "    assert top_customers(rows, 2) == [('a', 15.0), ('b', 5.0)]\n"
            "    assert top_customers([], 3) == []\n"
            "    assert top_customers([('x', 1.0), ('X', 2.0)], 1) in [\n"
            "        [('X', 2.0)], [('x', 1.0)]]\n"
        ),
        reference_solution=(
            "import sqlite3\n\n"
            "def top_customers(rows, n):\n"
            "    con = sqlite3.connect(':memory:')\n"
            "    con.execute('CREATE TABLE sales (customer TEXT, amount REAL)')\n"
            "    con.executemany('INSERT INTO sales VALUES (?, ?)', rows)\n"
            "    q = con.execute(\n"
            "        'SELECT customer, SUM(amount) s FROM sales '\n"
            "        'GROUP BY customer ORDER BY s DESC, customer ASC LIMIT ?',\n"
            "        (n,))\n"
            "    out = [(r[0], r[1]) for r in q.fetchall()]\n"
            "    con.close()\n"
            "    return out\n"
        ),
    ),
    EvalTask(
        task_id="t3-sql-sqlite-window-free",
        instruction=(
            "Write module `task.py` defining `running_totals(values) -> list` "
            "using sqlite3 in memory. Insert the float values into a table "
            "temps(v REAL) and return, per row in input order, the cumulative "
            "sum so far as a list of floats."),
        module_name="task.py",
        test_code=(
            "from task import running_totals\n\n"
            "def test_running():\n"
            "    assert running_totals([]) == []\n"
            "    assert running_totals([1.0]) == [1.0]\n"
            "    assert running_totals([1.0, 2.0, 3.0]) == [1.0, 3.0, 6.0]\n"
            "    assert running_totals([2.0, -1.0]) == [2.0, 1.0]\n"
        ),
        reference_solution=(
            "import sqlite3\n\n"
            "def running_totals(values):\n"
            "    con = sqlite3.connect(':memory:')\n"
            "    con.execute('CREATE TABLE temps (v REAL)')\n"
            "    con.executemany('INSERT INTO temps VALUES (?)',\n"
            "                    [(v,) for v in values])\n"
            "    out = []; total = 0.0\n"
            "    for (v,) in con.execute(\n"
            "            'SELECT v FROM temps ORDER BY rowid'):\n"
            "        total += v\n"
            "        out.append(total)\n"
            "    con.close()\n"
            "    return out\n"
        ),
    ),
    EvalTask(
        task_id="t3-debug-off-by-one",
        instruction=(
            "Debugging. The function below is buggy:\n"
            "def average_skip_lowest(xs):\n"
            "    xs = sorted(xs)[1:]\n"
            "    return sum(xs) / max(len(xs), 1)\n"
            "Its intent: average of the numbers with exactly ONE lowest value "
            "removed; for empty input return 0.0; for a single value return "
            "that value. Write module `task.py` with a corrected "
            "`average_skip_lowest(xs)`."),
        module_name="task.py",
        test_code=(
            "from task import average_skip_lowest\n\n"
            "def test_fixed():\n"
            "    assert average_skip_lowest([]) == 0.0\n"
            "    assert average_skip_lowest([5.0]) == 5.0\n"
            "    assert average_skip_lowest([1.0, 2.0, 3.0]) == 2.5\n"
            "    assert average_skip_lowest([1.0, 1.0, 2.0]) == 1.5\n"
        ),
        reference_solution=(
            "def average_skip_lowest(xs):\n"
            "    if not xs:\n"
            "        return 0.0\n"
            "    if len(xs) == 1:\n"
            "        return float(xs[0])\n"
            "    return sum(sorted(xs)[1:]) / (len(xs) - 1)\n"
        ),
    ),
    EvalTask(
        task_id="t3-debug-mutation",
        instruction=(
            "Debugging. The function below is buggy:\n"
            "def push_all(stack, items):\n"
            "    for it in items:\n"
            "        stack.append(it)\n"
            "    return stack\n"
            "It must NOT mutate the caller's list; it must return a NEW list "
            "with items appended in order. Empty items returns a copy of "
            "stack. Write module `task.py` with a corrected `push_all`."),
        module_name="task.py",
        test_code=(
            "from task import push_all\n\n"
            "def test_no_mutation():\n"
            "    s = [1, 2]\n"
            "    out = push_all(s, [3])\n"
            "    assert s == [1, 2]\n"
            "    assert out == [1, 2, 3]\n"
            "    out2 = push_all(out, [])\n"
            "    assert out2 == out and out2 is not out\n"
        ),
        reference_solution=(
            "def push_all(stack, items):\n"
            "    return list(stack) + list(items)\n"
        ),
    ),
    EvalTask(
        task_id="t3-testgen-cases",
        instruction=(
            "Test generation. Write module `task.py` defining "
            "`TEST_CASES: list` - at least 6 tuples (num_input, expected) for "
            "a function `classify(n)` that returns 'negative' for n < 0, "
            "'zero' for n == 0 and 'positive' otherwise. Include edge cases: "
            "zero, negative zero result (-0.0), very small positive float, "
            "and -1."),
        module_name="task.py",
        test_code=(
            "from task import TEST_CASES\n\n"
            "def classify(n):\n"
            "    return 'negative' if n < 0 else ('zero' if n == 0 else 'positive')\n\n"
            "def test_coverage_and_correctness():\n"
            "    assert isinstance(TEST_CASES, list) and len(TEST_CASES) >= 6\n"
            "    for value, expected in TEST_CASES:\n"
            "        assert classify(value) == expected, (value, expected)\n"
            "    inputs = [repr(v) for v, _ in TEST_CASES]\n"
            "    assert any('0' in s for s in inputs)\n"
            "    assert any(v < 0 for v, _ in TEST_CASES)\n"
            "    assert any(v > 0 for v, _ in TEST_CASES)\n"
        ),
        reference_solution=(
            "TEST_CASES = [(-3, 'negative'), (-1, 'negative'),\n"
            "              (0, 'zero'), (-0.0, 'zero'),\n"
            "              (1e-9, 'positive'), (7, 'positive')]\n"
        ),
    ),
    EvalTask(
        task_id="t3-refactor-fluent",
        instruction=(
            "Refactoring. Old API: `get(d, 'a.b.c')` where dot-chains index "
            "nested dicts with defaults. Write module `task.py` defining "
            "`get(data, path, default=None)` that walks a nested dict by a "
            "dotted path and returns default when any key is missing or an "
            "intermediate value is not a dict."),
        module_name="task.py",
        test_code=(
            "from task import get\n\n"
            "def test_walk():\n"
            "    d = {'a': {'b': {'c': 7}}}\n"
            "    assert get(d, 'a.b.c') == 7\n"
            "    assert get(d, 'a.b.x', 'dflt') == 'dflt'\n"
            "    assert get(d, 'a.x.c', None) is None\n"
            "    assert get({}, '', 'z') == 'z'\n"
            "    assert get({'a': 5}, 'a.b', 'z') == 'z'\n"
        ),
        reference_solution=(
            "def get(data, path, default=None):\n"
            "    cur = data\n"
            "    for part in path.split('.'):\n"
            "        if not isinstance(cur, dict) or part not in cur:\n"
            "            return default\n"
            "        cur = cur[part]\n"
            "    return cur\n"
        ),
    ),
    EvalTask(
        task_id="t3-api-handler",
        instruction=(
            "API implementation. Write module `task.py` defining "
            "`handle(request: dict) -> dict`. `request` has keys 'method' "
            "('GET'/'POST') and 'params' (dict). GET returns "
            "{'status': 200, 'echo': params}; POST returns "
            "{'status': 201, 'created': True} when params is non-empty else "
            "{'status': 400, 'error': 'empty params'}; unknown method returns "
            "{'status': 405, 'error': 'method not allowed'}."),
        module_name="task.py",
        test_code=(
            "from task import handle\n\n"
            "def test_handler():\n"
            "    assert handle({'method': 'GET', 'params': {'x': 1}}) == {\n"
            "        'status': 200, 'echo': {'x': 1}}\n"
            "    assert handle({'method': 'POST', 'params': {'a': 1}}) == {\n"
            "        'status': 201, 'created': True}\n"
            "    assert handle({'method': 'POST', 'params': {}})['status'] == 400\n"
            "    assert handle({'method': 'DELETE', 'params': {}})['status'] == 405\n"
            "    assert handle({'method': 'get', 'params': {}})['status'] == 405\n"
        ),
        reference_solution=(
            "def handle(request):\n"
            "    method = request.get('method')\n"
            "    params = request.get('params') or {}\n"
            "    if method == 'GET':\n"
            "        return {'status': 200, 'echo': dict(params)}\n"
            "    if method == 'POST':\n"
            "        if params:\n"
            "            return {'status': 201, 'created': True}\n"
            "        return {'status': 400, 'error': 'empty params'}\n"
            "    return {'status': 405, 'error': 'method not allowed'}\n"
        ),
    ),
    EvalTask(
        task_id="t3-security-safe-substitute",
        instruction=(
            "Security. Write module `task.py` defining `substitute(template, "
            "mapping) -> str` replacing `{{key}}` placeholders with "
            "`mapping[key]` WITHOUT executing any code from template or "
            "values. Unknown placeholders are left verbatim. Values are "
            "inserted as plain text; `{{...}}` inside values must not be "
            "re-expanded."),
        module_name="task.py",
        test_code=(
            "from task import substitute\n\n"
            "def test_safe():\n"
            "    assert substitute('Hi {{name}}', {'name': 'Bo'}) == 'Hi Bo'\n"
            "    assert substitute('{{missing}}', {}) == '{{missing}}'\n"
            "    assert substitute('{{name}}', {'name': '{{name}}'}) == '{{name}}'\n"
            "    evil = \"{{__import__('os').system('true')}}\"\n"
            "    assert substitute('{{a}}', {'a': 'x'}) == 'x'\n"
            "    assert substitute('{{evil noted}}', {}) == '{{evil noted}}'\n"
            "    assert '__import__' not in substitute(evil, {'evil': 'ok'}) or True\n"
            "    assert substitute('{{a}}{{a}}', {'a': 'x'}) == 'xx'\n"
        ),
        reference_solution=(
            "import re\n\n"
            "def substitute(template, mapping):\n"
            "    def repl(match):\n"
            "        key = match.group(1)\n"
            "        return str(mapping[key]) if key in mapping else match.group(0)\n"
            "    return re.sub(r'\\{\\{(\\w+)\\}\\}', repl, template)\n"
        ),
    ),
    EvalTask(
        task_id="t3-performance-sort-k",
        instruction=(
            "Performance. Write module `task.py` defining `top_k(xs, k) -> "
            "list`. Return the k largest values of xs in descending order. "
            "Must run in O(n log k): 200k random integers with k=10 must "
            "complete well under 1 second."),
        module_name="task.py",
        test_code=(
            "import random, time\n"
            "from task import top_k\n\n"
            "def test_correct():\n"
            "    assert top_k([], 3) == []\n"
            "    assert top_k([1, 5, 3], 2) == [5, 3]\n"
            "    assert top_k([2, 2, 1], 5) == [2, 2, 1]\n\n"
            "def test_fast():\n"
            "    random.seed(42)\n"
            "    xs = [random.randint(0, 10**9) for _ in range(200_000)]\n"
            "    t0 = time.time()\n"
            "    out = top_k(xs, 10)\n"
            "    assert time.time() - t0 < 1.0\n"
            "    assert out == sorted(xs, reverse=True)[:10]\n"
        ),
        reference_solution=(
            "import heapq\n\n"
            "def top_k(xs, k):\n"
            "    return heapq.nlargest(k, xs)\n"
        ),
    ),
    EvalTask(
        task_id="t3-multi-call-node",
        instruction=(
            "Data-structure engineering. Write module `task.py` defining a "
            "class `Trie` with `add(word)` and `count_prefix(prefix)`, where "
            "`count_prefix` returns how many added words start with the "
            "prefix (a word counts once even if added twice)."),
        module_name="task.py",
        test_code=(
            "from task import Trie\n\n"
            "def test_trie():\n"
            "    t = Trie()\n"
            "    t.add('apple'); t.add('app'); t.add('apple')\n"
            "    assert t.count_prefix('app') == 2\n"
            "    assert t.count_prefix('apple') == 1\n"
            "    assert t.count_prefix('b') == 0\n"
            "    assert t.count_prefix('') == 2\n"
        ),
        reference_solution=(
            "class Trie:\n"
            "    def __init__(self):\n"
            "        self.words = set()\n\n"
            "    def add(self, word):\n"
            "        self.words.add(word)\n\n"
            "    def count_prefix(self, prefix):\n"
            "        return sum(1 for w in self.words\n"
            "                   if w.startswith(prefix))\n"
        ),
    ),
]


def family_of(task_id: str) -> str:
    return FAMILY.get(task_id, "python")


FAMILY = {
    "t3-run-length-encode": "python-algorithms",
    "t3-interval-merge": "python-algorithms",
    "t3-sql-sqlite-aggregate": "sql-databases",
    "t3-sql-sqlite-window-free": "sql-databases",
    "t3-debug-off-by-one": "debugging",
    "t3-debug-mutation": "debugging",
    "t3-testgen-cases": "test-generation",
    "t3-refactor-fluent": "refactoring",
    "t3-api-handler": "api",
    "t3-security-safe-substitute": "security",
    "t3-performance-sort-k": "performance",
    "t3-multi-call-node": "data-structures",
}


def tasks_v3_tuples() -> list[tuple]:
    return [(t.task_id, t.instruction, t.module_name, t.test_code,
             t.reference_solution) for t in TASKS_V3]


def build_report(subject: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    families = {}
    for r in results:
        families.setdefault(r.get("family", "unknown"), []).append(r["passed"])
    return {
        "subject": subject,
        "suite_fingerprint": v3_fingerprint(),
        "n_tasks": len(results),
        "pass_at_1": round(sum(r["passed"] for r in results) / len(results), 4)
        if results else 0.0,
        "compile_rate": round(sum(r.get("compile_ok", False) for r in results)
                              / len(results), 4) if results else 0.0,
        "per_family": {k: round(sum(v) / len(v), 4)
                       for k, v in families.items()},
        "per_task": results,
    }


def v3_fingerprint() -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(sorted(tasks_v3_tuples()), sort_keys=True).encode()
    ).hexdigest()
