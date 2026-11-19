"""Divergent reasoning: parallel hypotheses, contradiction detection,
execution-based verification (stage 7, stage F).

For each coding task the model generates N hypotheses at different
temperatures (independent sampling paths), grades every candidate against
the task's hidden pytest suite in the sandbox, detects contradictions
(some hypotheses pass, others fail on the same task), measures
convergence, and selects only execution-verified candidates.

Quality is measured by verified pass rates on real tasks - never by the
number of hypotheses produced.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch

from cortex.evaluation.runner import grade_single
from cortex.evaluation.inference import extract_code
from cortex.evaluation.tasks import TASKS


def _generate_candidates(agent, task, temperatures, max_new_tokens: int):
    prompt = agent.tokenizer.EOT + "\n" + "# Task: " + task.instruction + "\n"
    ids = agent.tokenizer.encode(prompt)
    x = torch.tensor([ids[:200]], dtype=torch.long)
    hyps = []
    for temperature in temperatures:
        out = agent.model.generate(x, max_new_tokens=max_new_tokens,
                                   temperature=temperature,
                                   eot_id=agent.tokenizer.eot_id)
        text = agent.tokenizer.decode(out[0].tolist()[len(ids[:200]) - 1:])
        candidate = extract_code(text)
        grade = grade_single(task, candidate)
        hyps.append({
            "temperature": temperature,
            "candidate_head": candidate[:400],
            "nonempty": bool(candidate.strip()),
            "passed": bool(grade["passed"]),
            "exit_code": grade["exit_code"],
        })
    return hyps


def divergent_eval(agent, tasks=None, n_hypotheses: int = 3,
                   temperatures: tuple[float, ...] | None = None,
                   max_new_tokens: int = 180) -> dict[str, Any]:
    if tasks is None:
        tasks = TASKS
    if temperatures is None:
        temperatures = (0.2, 0.7, 1.0)
    if len(temperatures) != n_hypotheses:
        temperatures = temperatures[:n_hypotheses]

    n = max(1, len(temperatures))
    per_task = []
    for task in tasks:
        hyps = _generate_candidates(agent, task, temperatures,
                                    max_new_tokens)
        passes = [h["passed"] for h in hyps]
        n_pass = sum(passes)
        contradiction = 0 < n_pass < len(passes)
        verdicts = [tuple(sorted((h["passed"], h["exit_code"])))
                    for h in hyps]
        distinct_outcomes = len(set(verdicts))
        # convergence: fraction of hypotheses agreeing with the majority
        # verdict on task success (1.0 = all hypotheses agree)
        majority = max(n_pass, len(passes) - n_pass)
        convergence = round(majority / len(passes), 4) if passes else 0.0
        verified = next((h for h in hyps if h["passed"]), None)
        per_task.append({
            "task_id": task.task_id,
            "hypotheses": hyps,
            "n_hypotheses": len(passes),
            "n_passed": n_pass,
            "contradiction": contradiction,
            "distinct_outcomes": distinct_outcomes,
            "convergence": convergence,
            "verified_candidate_found": verified is not None,
        })

    total = len(per_task)
    agg = {
        "kind": "cortex-divergent-reasoning-eval",
        "n_tasks": total,
        "n_hypotheses_per_task": len(temperatures),
        "temperatures": list(temperatures),
        "pass_at_1_index0": round(
            sum(1 for t in per_task if t["hypotheses"][0]["passed"]) / total, 4),
        "pass_at_n_any": round(
            sum(1 for t in per_task if t["n_passed"] > 0) / total, 4),
        "contradiction_rate": round(
            sum(1 for t in per_task if t["contradiction"]) / total, 4),
        "mean_convergence": round(
            sum(t["convergence"] for t in per_task) / total, 4),
        "tasks_with_majority_correct": sum(
            1 for t in per_task if t["n_passed"] * 2 > t["n_hypotheses"]),
        "grading": "execution of hidden pytest suite inside sandbox per hypothesis",
    }
    return {"aggregate": agg, "per_task": per_task}
