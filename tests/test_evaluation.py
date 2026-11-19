import pytest

from cortex.evaluation.runner import evaluate, grade_single
from cortex.evaluation.tasks import TASKS


def test_harness_sanity_reference_solutions_pass():
    report = evaluate(reference=True)
    assert report["n_tasks"] == len(TASKS)
    assert report["pass_at_1"] == 1.0, report["per_task"]


def test_grade_single_detects_wrong_code():
    task = TASKS[0]
    rec = grade_single(task, "def is_palindrome(s):\n    return False\n")
    assert rec["passed"] is False


def test_grade_single_detects_correct_code():
    task = TASKS[0]
    correct = task.reference_solution
    rec = grade_single(task, correct)
    assert rec["passed"] is True


def test_evaluate_requires_candidate():
    with pytest.raises(ValueError):
        evaluate(candidate=None, reference=False)
