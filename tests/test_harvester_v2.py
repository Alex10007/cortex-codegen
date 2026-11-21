from pathlib import Path

from cortex.learning.doctest_harvester import (
    extract_all_candidates,
    harvest_all,
    mutate_source,
    verify_all,
)


GOOD = '''def add_two(a, b):
    """Add two numbers.

    >>> add_two(2, 3)
    5
    """
    return a + b
'''

METHOD = '''class Counter:
    def __init__(self, start):
        self.value = start

    def next(self):
        """Return the next value.

        >>> c = Counter(3)
        >>> c.next()
        4
        """
        self.value += 1
        return self.value
'''


def test_mutate_source_produces_compiling_mutants():
    mutants = mutate_source("def add_two(a, b):\n    return a + b\n")
    assert mutants
    for m in mutants:
        compile(m["code"], "<mutant>", "exec")
    assert any(m["code"] != "def add_two(a, b):\n    return a + b\n"
               for m in mutants)


def test_extract_all_candidates_emits_methods(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(GOOD + METHOD)
    cands = extract_all_candidates(p)
    names = set(cands)
    assert "add_two" in names
    assert any(name.startswith("Counter.") for name in names)
    assert cands["add_two"]["kind"] == "function"
    for name in names:
        if name.startswith("Counter."):
            assert cands[name]["kind"] == "method"


def test_verify_all_returns_mutant_verdicts(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(GOOD)
    cands = extract_all_candidates(p)
    assert cands["add_two"]["mutants"]
    verdicts = verify_all(p)
    v = verdicts["add_two"]
    assert v["ok"] is True
    assert v["attempted"] == 1
    assert v["mutants"]
    assert all(m["attempted"] == 1 for m in v["mutants"].values())


def test_harvest_all_emits_function_and_debugging_pairs(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(GOOD)
    pairs = harvest_all([str(tmp_path)])
    kinds = {pr["kind"] for pr in pairs}
    assert "function" in kinds
    assert "debugging" in kinds
    fn_pair = next(pr for pr in pairs if pr["kind"] == "function")
    dbg_pair = next(pr for pr in pairs if pr["kind"] == "debugging")
    assert fn_pair["code"] == dbg_pair["code"]
    assert "Buggy implementation" in dbg_pair["instruction"]
    licenses = {pr["provenance"]["license"] for pr in pairs}
    assert licenses == {"project-local"}
