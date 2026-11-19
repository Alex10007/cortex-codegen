from cortex.evaluation.runner_v2 import evaluate_v2, grade_v2, parse_multifile
from cortex.evaluation.tasks_v2 import TASKS_V2


def test_reference_solutions_pass_all_v2_tasks():
    report = evaluate_v2(reference=True)
    assert report["pass_at_1"] == 1.0
    assert report["n_tasks"] == len(TASKS_V2)
    assert set(report["per_tier"]) == {"function", "debugging",
                                       "multi-file", "repo"}


def test_stage_metrics_present():
    report = evaluate_v2(reference=True)
    for rec in report["per_task"]:
        assert rec["compile_ok"] is True
        assert rec["executed"] is True
        assert rec["passed"] is True
        assert rec["missing_files"] == []


def test_parse_multifile_labels_and_fallback():
    text = "intro\n```store.py\nA = 1\n```\n```report.py\nB = 2\n```\n"
    edits = parse_multifile(text, ["store.py", "report.py"])
    assert edits == {"store.py": "A = 1\n", "report.py": "B = 2\n"}
    fallback = parse_multifile("plain code", ["task.py"])
    assert fallback == {"task.py": "plain code\n"}


def test_wrong_candidate_fails():
    task = TASKS_V2[0]
    rec = grade_v2(task, {"task.py": "def collatz_steps(n):\n    return 99\n"})
    assert rec["passed"] is False


def test_missing_file_reported():
    task = TASKS_V2[4]
    rec = grade_v2(task, {})
    assert rec["passed"] is False
    assert "pkg/flags.py" in rec["missing_files"]
