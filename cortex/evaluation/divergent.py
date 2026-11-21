"""Divergent reasoning: parallel hypotheses, contradiction detection,
execution-based verification, and execution-feedback repair (stage 8
extension of the stage-7 mechanism).

Round 1 (divergence): N independently sampled candidates per task at
different temperatures, each graded against the task's hidden pytest suite
in the sandbox. Grad-backed selection picks only execution-verified
candidates.

Round 2 (repair): for tasks with no passing hypothesis, a new generation
is conditioned on the actual failure output (executed pytest stderr),
producing repair hypotheses graded by the same independent grader.

Quality is measured by verified pass rates on real tasks - never by the
number of hypotheses produced.
"""
from __future__ import annotations

import hashlib

import torch

from cortex.evaluation.runner import grade_single
from cortex.evaluation.inference import extract_code
from cortex.evaluation.failure_tax import (classify, failure_breakdown,
                                            repair_hint)
from cortex.evaluation.tasks import TASKS

REPAIR_TEMPERATURES = (0.2,)


def _sample_and_grade(agent, task, temperatures, max_new_tokens: int,
                      prefix: str, hyps_tag: str) -> list[dict]:
    prompt = (agent.tokenizer.EOT + "\n" + "# Task: " + task.instruction
              + "\n" + prefix)
    ids = agent.tokenizer.encode(prompt)
    x = torch.tensor([ids[:max(1, 200)]], dtype=torch.long)
    out = []
    for temperature in temperatures:
        gen = agent.model.generate(x, max_new_tokens=max_new_tokens,
                                   temperature=temperature,
                                   eot_id=agent.tokenizer.eot_id)
        text = agent.tokenizer.decode(gen[0].tolist()[len(ids[:200]) - 1:])
        candidate = extract_code(text)
        grade = grade_single(task, candidate)
        out.append({
            "round": hyps_tag,
            "temperature": temperature,
            "candidate_head": candidate[:300],
            "sha1": hashlib.sha1(
                candidate.encode("utf-8", "replace")).hexdigest()[:16],
            "nonempty": bool(candidate.strip()),
            "passed": bool(grade["passed"]),
            "failure_category": classify(grade),
            "grade": {"exit_code": grade["exit_code"],
                      "stderr_tail": (grade["stderr_tail"] or "")[-300:]},
        })
    return out


def divergent_eval(agent, tasks=None, n_hypotheses: int = 3,
                   temperatures: tuple[float, ...] | None = None,
                   max_new_tokens: int = 180,
                   repair_round: bool = True,
                   adaptive_budget: bool = False) -> dict:
    """Full divergent + repair evaluation. Every candidate is graded by
    real sandbox execution; no score is derived from generation alone."""
    if tasks is None:
        tasks = TASKS
    if temperatures is None:
        temperatures = (0.2, 0.7, 1.0)
    if len(temperatures) != n_hypotheses:
        temperatures = temperatures[:n_hypotheses]

    per_task = []
    import time
    for task in tasks:
        t0 = time.time()
        assessment = None
        task_temperatures = temperatures
        if adaptive_budget:
            from cortex.planning.engine import assess_objective
            assessment = assess_objective(task.instruction, {})
            task_temperatures = temperatures[:assessment["recommended_hypotheses"]]
            if not task_temperatures:
                task_temperatures = temperatures[:1]
        hyps = _sample_and_grade(agent, task, task_temperatures,
                                 max_new_tokens, "", "divergent")
        repair = []
        need_repair = repair_round and not any(h["passed"] for h in hyps)
        if need_repair:
            # targeted repair: classify the DOMINANT observed failure mode
            # and condition generation on the actual failure output
            breakdown = failure_breakdown(hyps)
            category = max((c for c in breakdown
                            if c not in ("passed", "unknown")),
                           key=lambda c: breakdown[c], default="unknown")
            failure = next(
                (h["grade"]["stderr_tail"] for h in hyps
                 if h["grade"]["stderr_tail"]), "")
            prefix = ""
            hint = repair_hint(category)
            if hint:
                prefix += "# " + hint + "\n"
            if failure:
                prefix += ("# Previous attempt failed:\n# " +
                           failure.replace("\n", "\n# ")[:600] + "\n")
            repair = _sample_and_grade(agent, task, REPAIR_TEMPERATURES,
                                       max_new_tokens, prefix, "repair")
            for h in repair:
                h["repair_category_targeted"] = category

        all_hyps = hyps + repair
        distinct_candidates = {h["sha1"] for h in all_hyps}
        n_pass = sum(1 for h in all_hyps if h["passed"])
        verdicts = {(h["round"], h["passed"]) for h in all_hyps}
        contradiction = (0 < n_pass < len(all_hyps)
                         and len({h["passed"] for h in hyps}) > 1)
        majority = n_pass if n_pass * 2 >= len(all_hyps) else len(all_hyps) - n_pass
        convergence = round(majority / len(all_hyps), 4) if all_hyps else 0.0
        verified = [h for h in all_hyps if h["passed"]]
        repair_recovered = (bool(repair)
                            and any(h["passed"] for h in repair)
                            and not any(h["passed"] for h in hyps))
        per_task.append({
            "task_id": task.task_id,
            "hypotheses_round1": hyps,
            "repair_round": repair,
            "distinct_candidates": len(distinct_candidates),
            "n_passed_total": n_pass,
            "contradiction": contradiction,
            "convergence": convergence,
            "repair_recovered": repair_recovered,
            "verified_selection": verified[0]["sha1"] if verified else None,
            "verified_candidate_found": bool(verified),
            "failure_breakdown_round1": failure_breakdown(
                hyps) if repair_round else None,
            "seconds_elapsed": round(time.time() - t0, 2),
            "tokens_used": len(task_temperatures) * max_new_tokens +
            len(repair) * max_new_tokens,
            "n_hypotheses": len(task_temperatures),
            "adaptive_assessment": assessment,
        })

    total = len(per_task)
    import collections

    cat_counts: dict[str, int] = collections.Counter()
    for t in per_task:
        for c, n in (t.get("failure_breakdown_round1") or {}).items():
            cat_counts[c] += n
    agg = {
        "kind": "cortex-divergent-reasoning-eval",
        "n_tasks": total,
        "n_hypotheses_per_task": len(temperatures),
        "adaptive_budget": adaptive_budget,
        "hypotheses_dispatched_total": sum(t["n_hypotheses"] for t in per_task),
        "mean_hypotheses_per_task": round(
            sum(t["n_hypotheses"] for t in per_task) / max(1, total), 4),
        "temperatures": list(temperatures),
        "repair_round": repair_round,
        "pass_at_1_index0": round(
            sum(1 for t in per_task if t["hypotheses_round1"][0]["passed"])
            / total, 4),
        "pass_at_n": round(sum(1 for t in per_task if t["n_passed_total"] > 0)
                           / total, 4),
        "pass_via_repair": round(
            sum(1 for t in per_task if t["repair_recovered"]) / total, 4),
        "contradiction_rate": round(
            sum(1 for t in per_task if t["contradiction"]) / total, 4),
        "mean_convergence": round(
            sum(t["convergence"] for t in per_task) / total, 4),
        "distinct_candidate_rate": round(
            sum(t["distinct_candidates"] for t in per_task)
            / max(1, sum(len(t["hypotheses_round1"]) + len(t["repair_round"])
                         for t in per_task)), 4),
        "failure_categories_round1": dict(cat_counts),
        "tokens_consumed": sum(t["tokens_used"] for t in per_task),
        "seconds_elapsed": round(
            sum(t["seconds_elapsed"] for t in per_task), 2),
        "grading": "each hypothesis graded by sandboxed pytest execution",
    }
    return {"aggregate": agg, "per_task": per_task}
