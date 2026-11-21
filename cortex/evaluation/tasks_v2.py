"""Evaluation suite v2: progressively harder, execution-graded tasks.

The v1 task set (cortex.evaluation.tasks.TASKS) is immutable; this module is
additive and separately fingerprinted. Tiers: function generation, debugging,
multi-file modification, repository-level tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskV2:
    task_id: str
    tier: str                    # function | debugging | multi-file | repo
    instruction: str
    base_files: dict[str, str]   # pre-existing files copied into the workspace
    edit_targets: list[str]      # files the candidate may (re)write
    test_code: str               # hidden pytest suite
    timeout_s: float = 30.0
    reference_edits: dict[str, str] = field(default_factory=dict)


TASKS_V2: list[TaskV2] = [
    TaskV2(
        task_id="t2-collatz",
        tier="function",
        instruction=("Write module `task.py` defining `collatz_steps(n: int) -> int`: "
                     "number of steps for n to reach 1 (n even -> n/2, odd -> 3n+1). "
                     "n=1 has 0 steps."),
        base_files={},
        edit_targets=["task.py"],
        test_code=(
            "from task import collatz_steps\n\n"
            "def test_one():\n    assert collatz_steps(1) == 0\n\n"
            "def test_two():\n    assert collatz_steps(2) == 1\n\n"
            "def test_six():\n    assert collatz_steps(6) == 8\n\n"
            "def test_twentyseven():\n    assert collatz_steps(27) == 111\n"
        ),
        reference_edits={"task.py": (
            "def collatz_steps(n):\n"
            "    steps = 0\n"
            "    while n != 1:\n"
            "        if n % 2 == 0:\n"
            "            n //= 2\n"
            "        else:\n"
            "            n = 3 * n + 1\n"
            "        steps += 1\n"
            "    return steps\n"
        )},
    ),
    TaskV2(
        task_id="t2-binary-search",
        tier="function",
        instruction=("Write module `task.py` defining `bsearch(items: list, target) -> int` "
                     "returning any index of target in a sorted list or -1."),
        base_files={},
        edit_targets=["task.py"],
        test_code=(
            "from task import bsearch\n\n"
            "def test_found():\n    assert bsearch([1,2,3,4,5], 4) in (3,)\n\n"
            "def test_missing():\n    assert bsearch([1,2,3], 9) == -1\n\n"
            "def test_empty():\n    assert bsearch([], 1) == -1\n"
        ),
        reference_edits={"task.py": (
            "def bsearch(items, target):\n"
            "    lo, hi = 0, len(items) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if items[mid] == target:\n"
            "            return mid\n"
            "        if items[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        )},
    ),
    TaskV2(
        task_id="t2-debug-off-by-one",
        tier="debugging",
        instruction=("`task.py` below has a bug: its tests fail. Rewrite ONLY "
                     "`task.py` (same API) so the tests pass."),
        base_files={"task.py": (
            "def clamp(value, low, high):\n"
            "    \"\"\"Clamp value into [low, high].\"\"\"\n"
            "    if value < low:\n"
            "        return low\n"
            "    if value > high + 1:\n"
            "        return high\n"
            "    return value\n"
        )},
        edit_targets=["task.py"],
        test_code=(
            "from task import clamp\n\n"
            "def test_low():\n    assert clamp(-5, 0, 10) == 0\n\n"
            "def test_high():\n    assert clamp(99, 0, 10) == 10\n\n"
            "def test_inside():\n    assert clamp(5, 0, 10) == 5\n\n"
            "def test_boundary():\n    assert clamp(10, 0, 10) == 10\n"
        ),
        reference_edits={"task.py": (
            "def clamp(value, low, high):\n"
            "    \"\"\"Clamp value into [low, high].\"\"\"\n"
            "    if value < low:\n"
            "        return low\n"
            "    if value > high:\n"
            "        return high\n"
            "    return value\n"
        )},
    ),
    TaskV2(
        task_id="t2-multifile",
        tier="multi-file",
        instruction=("Two modules exist: `store.py` (working) and `report.py` "
                     "(imports a missing helper). Provide BOTH files: reply with "
                     "fenced blocks labeled store.py and report.py implementing "
                     "a top-score reporter."),
        base_files={"store.py": (
            "SCORES = {'amy': 88, 'bob': 95, 'cal': 72}\n"
        )},
        edit_targets=["store.py", "report.py"],
        test_code=(
            "from report import top_report\n\n"
            "def test_top_report():\n"
            "    assert top_report() == [('bob', 95), ('amy', 88), ('cal', 72)]\n"
        ),
        reference_edits={
            "store.py": "SCORES = {'amy': 88, 'bob': 95, 'cal': 72}\n",
            "report.py": (
                "from store import SCORES\n\n\n"
                "def top_report():\n"
                "    return sorted(SCORES.items(), key=lambda kv: -kv[1])\n"
            ),
        },
    ),
    TaskV2(
        task_id="t2-repo-config",
        tier="repo",
        instruction=("Repository `pkg/` is missing its feature flag registry: "
                     "provide `pkg/flags.py` defining `feature_enabled(name)` "
                     "reading the FEATURES dict from `pkg/settings.py` (values "
                     "may be bool or the string 'on'/'off'). Unknown names are disabled."),
        base_files={
            "pkg/__init__.py": "",
            "pkg/settings.py": "FEATURES = {'dark_mode': 'on', 'beta': False}\n",
        },
        edit_targets=["pkg/flags.py"],
        test_code=(
            "from pkg.flags import feature_enabled\n\n"
            "def test_on_string():\n    assert feature_enabled('dark_mode') is True\n\n"
            "def test_false():\n    assert feature_enabled('beta') is False\n\n"
            "def test_unknown():\n    assert feature_enabled('nope') is False\n"
        ),
        reference_edits={"pkg/flags.py": (
            "from pkg.settings import FEATURES\n\n\n"
            "def feature_enabled(name):\n"
            "    value = FEATURES.get(name, False)\n"
            "    if isinstance(value, str):\n"
            "        return value == 'on'\n"
            "    return bool(value)\n"
        )},
    ),
]


def tasks_v2_tuples() -> list[tuple]:
    return [(t.task_id, t.tier, t.instruction, t.test_code,
             json_dumps_files(t.base_files), json_dumps_files(t.reference_edits))
            for t in TASKS_V2]


def json_dumps_files(files: dict[str, str]) -> str:
    import json

    return json.dumps(files, sort_keys=True)
