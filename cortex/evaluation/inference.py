"""Inference diagnostics: separates model capacity limits from inference,
decoding and evaluation-compatibility limits.

Measured instruments (all real):
- teacher-forced next-token accuracy on held-out pairs (how well the model
  predicts verified code when reading it — capacity signal);
- prompt-format A/B: SFT-trained prompt style vs eval task prompt style,
  pass@1 measured both ways for the same checkpoint (inference/eval
  compatibility signal);
- decode ablations: temperature/top-k variants (decoding signal).
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from cortex.evaluation.runner import grade_single
from cortex.evaluation.tasks import TASKS

FENCE_RE = re.compile(r"```(?:(?:python|py)?\n)(.*?)```", flags=re.S)


def _first_code_block(text: str) -> str:
    blocks = FENCE_RE.findall(text)
    return blocks[0] if blocks else text


def teacher_forced_token_accuracy(model, tok, recs: list[dict[str, Any]],
                                  seq_len: int, batch_size: int = 8,
                                  max_batches: int = 6) -> Optional[float]:
    import torch

    from cortex.training.sft import make_batch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, min(len(recs), batch_size * max_batches), batch_size):
            x, y = make_batch(recs[i:i + batch_size], tok, seq_len)
            if x is None:
                continue
            logits, _ = model(x)
            mask = y != -100
            pred = logits.argmax(dim=-1)
            hit = (pred == y) & mask
            correct += int(hit.sum())
            total += int(mask.sum())
    model.train()
    return round(correct / total, 4) if total else None


def _grade_task(task, candidate_code: str) -> dict[str, Any]:
    """Grade a task with the right runner: v2 tasks (multiple edit targets)
    go through the workspace-level v2 grader, v1 tasks through grade_single."""
    if hasattr(task, "edit_targets"):
        from cortex.evaluation.runner_v2 import grade_v2, parse_multifile

        rec = grade_v2(task, parse_multifile(candidate_code, task.edit_targets))
        return {"passed": rec["passed"], "compile_ok": rec["compile_ok"],
                "executed": rec["executed"]}
    return grade_single(task, candidate_code)


def prompt_format_ab(agent) -> dict[str, Any]:
    """Evaluate the SFT checkpoint under two prompt styles on the v1 tasks."""
    task = TASKS[0]
    styles = {
        "eval_style": "# Task: {instruction}\n",
        "sft_style": "{instruction}\n",
    }
    out: dict[str, Any] = {}
    import torch

    ids_cache: dict[str, list[int]] = {}
    for style, template in styles.items():
        prompt = template.format(instruction=task.instruction)
        ids = agent.tokenizer.encode(agent.tokenizer.EOT + "\n" + prompt)
        x = torch.tensor([ids[-200:]], dtype=torch.long)
        gen = agent.model.generate(x, max_new_tokens=120, temperature=0.2,
                                   top_k=20)[0].tolist()[len(x[0]) - 1:]
        text = agent.tokenizer.decode(gen)
        candidate = _first_code_block(text)
        rec = _grade_task(task, candidate)
        ids_cache[style] = len(ids)
        out[style] = {"passed": rec["passed"],
                      "generated_head": candidate[:240]}
    return out


def decode_ablation(agent, task) -> dict[str, Any]:
    import torch

    results = {}
    for label, (temp, top_k) in {
        "greedy": (0.01, 1), "temp02-k50": (0.2, 50), "temp07-k50": (0.7, 50),
    }.items():
        prompt = agent.tokenizer.EOT + "\n" + task.instruction + "\n"
        ids = agent.tokenizer.encode(prompt)
        x = torch.tensor([ids[-200:]], dtype=torch.long)
        gen = agent.model.generate(x, max_new_tokens=120, temperature=temp,
                                   top_k=top_k)[0].tolist()[len(x[0]) - 1:]
        text = agent.tokenizer.decode(gen)
        candidate = _first_code_block(text)
        rec = _grade_task(task, candidate)
        results[label] = {"passed": rec["passed"],
                          "generated_head": candidate[:240]}
    return results


def compile_diagnosis(agent, task) -> dict[str, Any]:
    import torch

    prompt = agent.tokenizer.EOT + "\n" + task.instruction + "\n"
    ids = agent.tokenizer.encode(prompt)
    x = torch.tensor([ids[-200:]], dtype=torch.long)
    gen = agent.model.generate(x, max_new_tokens=150, temperature=0.2,
                               top_k=20)[0].tolist()[len(x[0]) - 1:]
    text = agent.tokenizer.decode(gen)
    candidate = _first_code_block(text)
    try:
        ast.parse(candidate)
        stage = "parsed"
    except SyntaxError:
        stage = "syntax-error"
    return {"stage": stage, "generated_head": candidate[:400],
            "text_head": text[:400]}
