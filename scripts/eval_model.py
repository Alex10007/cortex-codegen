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


def extract_code(text: str) -> str:
    fences = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
    if fences:
        return fences[0]
    return text


def evaluate_checkpoint(art: Path, checkpoint_path: Path | str,
                        temperature: float = 0.2,
                        max_new_tokens: int = 180) -> dict:
    agent = CortexAgent(
        state_dir=str(art / "cortex-state"),
        tokenizer_path=str(art / "tokenizer" / "tokenizer.json"),
        checkpoint_path=str(checkpoint_path),
    )
    per_task = []
    for task in TASKS:
        prompt = agent.tokenizer.EOT + "\n" + "# Task: " + task.instruction + "\n"
        ids = agent.tokenizer.encode(prompt)
        x = torch.tensor([ids[:200]], dtype=torch.long)
        out = agent.model.generate(x, max_new_tokens=max_new_tokens,
                                   temperature=temperature)
        text = agent.tokenizer.decode(out[0].tolist()[len(ids[:200]) - 1:])
        candidate = extract_code(text)
        rec = grade_single(task, candidate)
        rec["generated_head"] = candidate[:400]
        per_task.append(rec)
        print(f"[eval] {task.task_id}: passed={rec['passed']}")

    passed = sum(1 for r in per_task if r["passed"])
    return {
        "model_config": "cortex-demo",
        "checkpoint": str(checkpoint_path),
        "pass_at_1": round(passed / len(per_task), 4),
        "n_tasks": len(per_task),
        "per_task": per_task,
        "grading": "execution of hidden pytest suite inside sandbox",
        "note": ("measured on the demo-scale checkpoint trained on CPU; "
                 "this is NOT the 2B-class target config"),
    }


def main() -> None:
    art = ROOT / "artifacts"
    checkpoint = art / "runs" / "cortex-demo" / "final.pt"
    if len(sys.argv) > 1:
        checkpoint = Path(sys.argv[1])
    report = evaluate_checkpoint(art, checkpoint)
    (art / "eval_model.json").write_text(json.dumps(report, indent=2))
    print("model pass@1:", report["pass_at_1"], "-> artifacts/eval_model.json")


if __name__ == "__main__":
    main()
