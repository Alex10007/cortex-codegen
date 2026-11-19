import json
from pathlib import Path

from cortex.planning.engine import analyze_repository, plan
from cortex.pipeline.engineer import run_engineering_pipeline


def _make_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_calc.py").write_text(
        "from pkg.calc import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n")
    (root / "requirements.txt").write_text("pytest\n")


def test_repository_analysis(tmp_path):
    _make_repo(tmp_path)
    repo_map = analyze_repository(str(tmp_path))
    assert repo_map["file_count"] >= 4
    assert repo_map["languages"]["python"] >= 3
    assert any("test_calc.py" in t for t in repo_map["test_files"])
    assert "requirements.txt" in repo_map["manifests"]
    assert repo_map["python_imports"]["tests/test_calc.py"] == ["pkg"]


def test_plan_structure(tmp_path):
    _make_repo(tmp_path)
    repo_map = analyze_repository(str(tmp_path))
    p = plan("add a multiply helper", str(tmp_path), repo_map)
    d = p.to_dict()
    assert d["objective"] == "add a multiply helper"
    actions = [s["action"] for s in d["steps"]]
    assert actions == ["analyze", "implement", "test-command", "security-scan",
                       "validate"]
    assert not any(s["requires_authorization"] for s in d["steps"])


def test_plan_flags_destructive_objectives(tmp_path):
    _make_repo(tmp_path)
    repo_map = analyze_repository(str(tmp_path))
    p = plan("deploy to production", str(tmp_path), repo_map)
    assert any(s.requires_authorization for s in p.steps)


def test_pipeline_blocks_destructive_without_authorization(tmp_path):
    _make_repo(tmp_path)
    report = run_engineering_pipeline("delete prod data", str(tmp_path))
    assert report["status"] == "blocked"


def test_pipeline_verifies_passing_repo(tmp_path):
    _make_repo(tmp_path)
    report = run_engineering_pipeline("add multiply helper", str(tmp_path))
    assert report["status"] == "verified"
    assert report["tests"]["ok"] is True
    assert report["tests"]["summary"]["passed"] >= 1
    assert report["verified"] is True


def test_pipeline_reports_failure(tmp_path):
    _make_repo(tmp_path)
    (tmp_path / "tests" / "test_bad.py").write_text(
        "import pkg\n\ndef test_broken():\n    assert pkg.calc.add(1, 1) == 3\n")
    report = run_engineering_pipeline("fix failing test", str(tmp_path))
    assert report["status"] == "tests-failed"
    assert report["verified"] is False


def test_pipeline_applies_patcher(tmp_path):
    _make_repo(tmp_path)
    report = run_engineering_pipeline(
        "add multiply", str(tmp_path),
        patcher=lambda p: {"pkg/calc.py":
                           "def add(a, b):\n    return a + b\n\n"
                           "def multiply(a, b):\n    return a * b\n"},
        test_command_top=["python3", "-m", "pytest", "-q"])
    assert "pkg/calc.py" in report["files_changed"]
    assert report["verified"] is True
