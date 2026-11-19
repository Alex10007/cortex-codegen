#!/usr/bin/env python3
"""Measure the demo model's real code-generation capability.

Greedy-generates a task.py from the instruction prompt, then grades it by
executing the hidden tests in the sandbox. The stored pass@1 is the measured
truth for this checkpoint scale, whatever it is.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cortex.core import CortexAgent  # noqa: E402
from cortex.evaluation.runner import grade_single  # noqa: E402
from cortex.evaluation.tasks import TASKS  # noqa: E402

ART = ROOT / "artifacts"


def extract_code(text: str) -> str:
    fences = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
    if fences:
        return fences[0]
    return text


def main() -> None:
    agent = CortexAgent(
        state_dir=str(ART / "cortex-state"),
        tokenizer_path=str(ART / "tokenizer" / "tokenizer.json"),
        checkpoint_path=str(ART / "runs" / "cortex-demo" / "final.pt"),
    )
    per_task = []
    for task in TASKS:
        prompt = agent.tokenizer.EOT + "\n" + "# Task: " + task.instruction + "\n"
        ids = agent.tokenizer.encode(prompt)
        x = torch.tensor([ids[:200]], dtype=torch.long)
        out = agent.model.generate(x, max_new_tokens=180, temperature=0.2)
        text = agent.tokenizer.decode(out[0].tolist()[len(ids[:200]) - 1:])
        candidate = extract_code(text)
        rec = grade_single(task, candidate)
        rec["generated_head"] = candidate[:400]
        per_task.append(rec)
        print(f"[eval] {task.task_id}: passed={rec['passed']}")

    passed = sum(1 for r in per_task if r["passed"])
    report = {
        "model_config": "cortex-demo",
        "checkpoint": str(ART / "runs" / "cortex-demo" / "final.pt"),
        "pass_at_1": round(passed / len(per_task), 4),
        "n_tasks": len(per_task),
        "per_task": per_task,
        "grading": "execution of hidden pytest suite inside sandbox",
        "note": ("measured on the demo-scale checkpoint after 600 CPU training "
                 "steps; this is NOT the 2B-class target config"),
    }
    (ART / "eval_model.json").write_text(json.dumps(report, indent=2))
    print("model pass@1:", report["pass_at_1"], "-> artifacts/eval_model.json")


if __name__ == "__main__":
    main()
