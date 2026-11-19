"""Evaluation tasks with hidden execution-graded tests.

Tasks are authored inside this repository and never derived from the training
corpus, to avoid train/test contamination. Each task is graded by executing
pytest inside the sandbox — a pass is a real execution fact, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    instruction: str
    module_name: str
    test_code: str
    timeout_s: float = 30.0
    reference_solution: str = ""


TASKS: list[EvalTask] = [
    EvalTask(
        task_id="t-palindrome",
        instruction=("Write module `task.py` defining `is_palindrome(s: str) -> bool`. "
                     "Ignore case and non-alphanumeric characters."),
        module_name="task.py",
        test_code=(
            "from task import is_palindrome\n\n"
            "def test_basic():\n"
            "    assert is_palindrome('racecar') is True\n\n"
            "def test_with_punct():\n"
            "    assert is_palindrome('A man, a plan, a canal: Panama') is True\n\n"
            "def test_negative():\n"
            "    assert is_palindrome('hello') is False\n"
        ),
        reference_solution=(
            "def is_palindrome(s):\n"
            "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
    ),
    EvalTask(
        task_id="t-fizzbuzz",
        instruction=("Write module `task.py` defining `fizzbuzz(n: int) -> list[str]` "
                     "for values 1..n with the classic rules as strings."),
        module_name="task.py",
        test_code=(
            "from task import fizzbuzz\n\n"
            "def test_fifteen():\n"
            "    got = fizzbuzz(15)\n"
            "    assert got[0] == '1'\n"
            "    assert got[2] == 'Fizz'\n"
            "    assert got[4] == 'Buzz'\n"
            "    assert got[14] == 'FizzBuzz'\n"
            "    assert len(got) == 15\n\n"
            "def test_one():\n"
            "    assert fizzbuzz(1) == ['1']\n"
        ),
        reference_solution=(
            "def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        s = ''\n"
            "        if i % 3 == 0:\n"
            "            s += 'Fizz'\n"
            "        if i % 5 == 0:\n"
            "            s += 'Buzz'\n"
            "        out.append(s or str(i))\n"
            "    return out\n"
        ),
    ),
    EvalTask(
        task_id="t-wordcount",
        instruction=("Write module `task.py` defining `word_frequencies(text: str) -> dict` "
                     "mapping lowercase words to counts; split on whitespace."),
        module_name="task.py",
        test_code=(
            "from task import word_frequencies\n\n"
            "def test_counts():\n"
            "    got = word_frequencies('a b A  b!')\n"
            "    assert got.get('a') == 2 or got == {'a': 2, 'b': 2, 'b!': 1}\n"
            "    assert list(got.values()).count(1) >= 1\n\n"
            "def test_empty():\n"
            "    assert word_frequencies('') == {}\n"
        ),
        reference_solution=(
            "def word_frequencies(text):\n"
            "    out = {}\n"
            "    for w in text.split():\n"
            "        key = w.lower()\n"
            "        out[key] = out.get(key, 0) + 1\n"
            "    return out\n"
        ),
    ),
]
