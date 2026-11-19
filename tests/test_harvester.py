from pathlib import Path

from cortex.learning.doctest_harvester import extract_candidates, harvest, verify_file


GOOD = '''def add_two(a, b):
    """Add two numbers.

    >>> add_two(2, 3)
    5
    >>> add_two(-1, 1)
    0
    """
    return a + b
'''

FAILING = '''def broken_add(a, b):
    """Add numbers.

    >>> broken_add(1, 1)
    3
    """
    return a + b
'''


def test_extract_candidates(tmp_path):
    p = tmp_path / "good.py"
    p.write_text(GOOD)
    cands = extract_candidates(p)
    assert set(cands) == {"add_two"}


def test_verify_file_passing_doctest(tmp_path):
    p = tmp_path / "good.py"
    p.write_text(GOOD)
    verdicts = verify_file(p)
    assert verdicts["add_two"]["ok"] is True
    assert verdicts["add_two"]["attempted"] == 2
    assert verdicts["add_two"]["failed"] == 0


def test_verify_file_failing_doctest_is_not_verified(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text(FAILING)
    verdicts = verify_file(p)
    assert verdicts["broken_add"]["ok"] is False
    assert verdicts["broken_add"]["failed"] == 1


def test_harvest_only_emits_verified_pairs(tmp_path):
    (tmp_path / "good.py").write_text(GOOD)
    (tmp_path / "broken.py").write_text(FAILING)
    (tmp_path / "nodoc.py").write_text("def quiet():\n    return 0\n")
    pairs = harvest([str(tmp_path)])
    names = {p["name"] for p in pairs}
    assert "add_two" in names
    assert "broken_add" not in names
    assert "quiet" not in names
    for p in pairs:
        assert p["provenance"]["license"] == "Python-2.0 / PSF"
        assert p["instruction"].startswith("Implement the Python function")
