"""Stage-10 deep failure diagnosis for code generation.

Stage 9 measured pass@1 = 0.0 with its 8-category execution taxonomy
(passed/syntax/import/runtime/assertion/timeout/empty/unknown). This module
goes one level deeper per the Stage 10 directive (workstream C1): it
diagnoses WHY a candidate failed by jointly examining the generated text and
the execution evidence, mapping failures to the directive's diagnosis
categories:

- degenerate_generation   (excessive repetition / no code content)
- truncation              (output cut mid-construction, no fence closure)
- extraction_failure      (code present but no fenced block)
- signature_mismatch      (function name/arity differs from the required one)
- syntax_error            (python failed to parse the candidate)
- undefined_dependency    (invalid imports or API usage at import/run time)
- incorrect_logic         (compiles and runs but fails verification)
- incomplete_implementation (candidate ends prematurely / missing body)
- non_execution           (grader never got to run tests, unknown cause)
- passed

Every diagnosis carries the evidence used to assign it. Nothing here reads
hidden test content: diagnosis uses only the candidate text, the grader's
execution record (exit code / stderr tail) and the task's PUBLIC reference
signature (task_id / module_name / instruction), which candidates are
already shown.
"""
from __future__ import annotations

import ast
from typing import Any

from cortex.evaluation.runner import grade_single
from cortex.evaluation.inference import extract_code

DIAGNOSES = (
    "passed", "degenerate_generation", "truncation", "extraction_failure",
    "signature_mismatch", "syntax_error", "undefined_dependency",
    "incorrect_logic", "incomplete_implementation", "non_execution")


def _extract_function_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _repeated_fraction(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    return max(counts.values()) / len(lines)


DIAGNOSIS_LEVELS = {
    "degenerate_generation": 3,  # worst early exits order first
    "truncation": 3,
    "extraction_failure": 3,
    "syntax_error": 3,
    "undefined_dependency": 2,
    "signature_mismatch": 2,
    "incomplete_implementation": 2,
    "non_execution": 1,
    "incorrect_logic": 1,
    "passed": 0,
}


def expected_callable(instruction: str) -> str | None:
    """Best-effort required function name from the INSTRUCTION text the
    candidate also sees (public task signature, not hidden tests)."""
    import re

    m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(", instruction)
    return m.group(1) if m else None


def _required_fences(text: str) -> bool:
    return text.count("```") >= 2


def diagnose_candidate(candidate: str, grade: dict[str, Any],
                       instruction: str = "",
                       max_tokens_hint: int | None = None) -> dict[str, Any]:
    """Diagnose one graded candidate from its own evidence."""
    if grade.get("passed"):
        return {"diagnosis": "passed", "evidence": "grader passed"}

    body = (candidate or "").strip()
    if not body:
        return {"diagnosis": "incomplete_implementation",
                "evidence": "empty candidate (nothing generated)"}

    rep = _repeated_fraction(body)
    if rep >= 0.6 and len(body.splitlines()) >= 4:
        return {"diagnosis": "degenerate_generation",
                "evidence": f"top line repeated in {rep:.0%} of lines"}

    if max_tokens_hint is not None and "```" not in body \
            and len(body.splitlines()) >= max_tokens_hint // 10:
        return {"diagnosis": "truncation",
                "evidence": "output ran to the token limit without closure"}

    if "```" in body and not _required_fences(body):
        return {"diagnosis": "truncation",
                "evidence": "fenced block opened but never closed"}

    code = extract_code(body)
    if code is body and "```" in body:
        return {"diagnosis": "extraction_failure",
                "evidence": "no fenced code block found inside markdown"}

    try:
        ast.parse(code)
    except SyntaxError as e:
        # distinguish mid-generation truncation from real syntax errors
        tail = code.splitlines()[-1] if code.strip() else ""
        if e.msg in ("unexpected EOF while parsing",
                     "invalid syntax") and (tail.endswith((":", "(", ","))
                                            or code.count("(")
                                            > code.count(")")):
            return {"diagnosis": "truncation",
                    "evidence": f"syntax EOF with unbalanced delimiters: {e}"}
        return {"diagnosis": "syntax_error", "evidence": str(e)}

    stderr = grade.get("stderr_tail", "") or ""
    err = grade.get("error", "") or ""
    blob = stderr + "\n" + err
    if grade.get("compile_ok") is False:
        return {"diagnosis": "syntax_error", "evidence": "grader compile fail"}

    ib = blob.lower()
    if "nameerror" in ib or "modulenotfounderror" in ib \
            or "attributeerror" in ib and "module" in ib:
        return {"diagnosis": "undefined_dependency",
                "evidence": blob.strip().splitlines()[-1][:160]
                if blob.strip() else "name/module error in stderr"}

    need = expected_callable(instruction)
    got = _extract_function_names(code)
    if need and need not in got:
        return {"diagnosis": "signature_mismatch",
                "evidence": f"defines {got[:3]} but instruction requires "
                            f"`{need}(...)`"}

    cat = grade.get("failure_category")
    if cat == "runtime" and "syntaxerror" in ib:
        return {"diagnosis": "syntax_error",
                "evidence": blob.strip().splitlines()[-1][:160]}
    if cat in ("runtime", "assertion", "timeout"):
        kind = {"runtime": "ran but terminated with an error",
                "assertion": "ran but failed verification",
                "timeout": "exceeded time limit"}.get(cat, cat)
        return {"diagnosis": "incorrect_logic" if cat == "assertion"
                else "undefined_dependency" if "error" in ib and
                ("import" in ib or "name" in ib) else "non_execution",
                "evidence": (blob.strip().splitlines()[-1][:160]
                             if blob.strip() else kind)}

    if "indent" in ib or "eof" in ib:
        return {"diagnosis": "incomplete_implementation",
                "evidence": blob.strip().splitlines()[-1][:160]}

    return {"diagnosis": "non_execution",
            "evidence": (blob.strip().splitlines()[-1][:160]
                         if blob.strip() else
                         f"ungraded failure path (category={cat})")}


def diagnose_tasks(tasks, candidates_by_task: dict[str, str],
                   max_tokens_hint: int | None = None) -> dict[str, Any]:
    """Grade + diagnose a full task set from generated raw outputs."""
    per_task = []
    breakdown: dict[str, int] = {}
    for task in tasks:
        candidate = candidates_by_task.get(task.task_id, "")
        grade = grade_single(task, extract_code(candidate))
        d = diagnose_candidate(candidate, grade, task.instruction,
                               max_tokens_hint)
        breakdown[d["diagnosis"]] = breakdown.get(d["diagnosis"], 0) + 1
        per_task.append({
            "task_id": task.task_id,
            "passed": bool(grade.get("passed")),
            "failure_category": grade.get("failure_category"),
            "diagnosis": d["diagnosis"],
            "evidence": d["evidence"],
            "candidate_head": candidate[:300],
        })
    return {"per_task": per_task, "diagnosis_breakdown": breakdown}


def root_cause_rank(breakdown: dict[str, int]) -> list[tuple[str, int]]:
    """Order diagnosis categories by DIAGNOSIS_LEVELS severity, then count."""
    return sorted(
        ((k, v) for k, v in breakdown.items() if k != "passed"),
        key=lambda kv: (-DIAGNOSIS_LEVELS.get(kv[0], 5), -kv[1], kv[0]))
