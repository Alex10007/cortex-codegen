"""Memorization-versus-generalization investigation (stage 12 workstream C).

All metrics are computed from the persisted dataset and executed model
probes. Reference solutions of perturbed tasks are validated by execution
before use; the immutable benchmarks are never modified.
"""
from __future__ import annotations

import ast
import dataclasses
import random
import re
from typing import Any

from cortex.evaluation.structure_audit import body_fingerprint
from cortex.training.sft import contamination_overlap


def _norm_wordset(text: str) -> set[str]:
    out: set[str] = set()
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text):
        out.add(word)
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def dataset_overlap(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Duplicate and near-duplicate accounting within and across splits."""
    by_split: dict[str, list] = {"train": [], "val": [], "heldout": []}
    for rec in recs:
        split = rec.get("split", "train")
        by_split.setdefault(split, []).append(rec)
    exact_dups = 0
    for split, rows in by_split.items():
        hashes = [r.get("content_sha256") for r in rows]
        exact_dups += len(hashes) - len(set(hashes))

    fingerprints = []
    for rec in recs:
        for key, fp in body_fingerprint(rec["code"]).items():
            fingerprints.append((rec.get("split"), key, fp))
    fp_by_body: dict[str, list[str]] = {}
    for split, key, fp in fingerprints:
        fp_by_body.setdefault(fp, []).append(f"{split}/{key}")
    repeated_bodies = {fp: v for fp, v in fp_by_body.items() if len(v) > 1}

    # near-duplicate pairs (word-set Jaccard >= 0.8) between train and
    # other splits - the memorization surface
    train_ws = [(_norm_wordset(r["instruction"] + "\n" + r["code"]), r)
                for r in by_split.get("train", [])]
    cross_near = 0
    for split in ("val", "heldout"):
        for r in by_split.get(split, []):
            ws = _norm_wordset(r["instruction"] + "\n" + r["code"])
            for tw, _ in train_ws:
                if jaccard(ws, tw) >= 0.8:
                    cross_near += 1
                    break
    return {
        "n_records": len(recs),
        "exact_duplicate_records": exact_dups,
        "repeated_function_bodies": len(repeated_bodies),
        "repeated_function_body_examples": sorted(
            {k for v in repeated_bodies.values() for k in v})[:20],
        "train_val_heldout_near_dup_pairs_ge_0.8": cross_near,
        "splits": {k: len(v) for k, v in by_split.items()},
    }


def benchmark_contamination(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Max contamination overlap against the immutable benchmark reference
    solutions per record (re-executes the referenced builder's filter)."""
    worst = 0.0
    worst_task = None
    worst_name = None
    for rec in recs:
        scores = contamination_overlap(rec)
        for task_id, v in scores.items():
            if v > worst:
                worst, worst_task, worst_name = v, task_id, rec["name"]
    return {"max_contamination_overlap": round(worst, 4),
            "worst_task": worst_task, "worst_record": worst_name}


_GEN_TOKENS = ("evaluate", "signal", "wrapper", "handler", "buffer",
               "record", "stream", "filter", "measure", "cortex_probe")


def perturb_task(task, seed: int) -> Any:
    """Build an identifier-renamed variant of a benchmark task (a
    generalization PROBE, not a benchmark): every occurrence of every
    top-level def/class name is renamed consistently in instruction, tests
    and reference. The variant is executed against its reference solution
    at build time; tasks whose perturbed reference fails are rejected."""
    names = re.findall(r"(^def\s+|^class\s+)([A-Za-z_]\w*)",
                       task.reference_solution, flags=re.M)
    flat = [b for _, b in names]
    if not flat:
        return None
    rng = random.Random(seed)
    mapping = {}
    for i, name in enumerate(sorted(set(flat))):
        mapping[name] = f"cortex_probe_{_GEN_TOKENS[i % len(_GEN_TOKENS)]}" \
                        f"_{rng.randrange(1000, 9999)}"

    def rename(text: str) -> str:
        for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text

    return dataclasses.replace(
        task, task_id=task.task_id + "-perturbed",
        instruction=rename(task.instruction),
        test_code=rename(task.test_code),
        reference_solution=rename(task.reference_solution))


def perturbation_suite_valid(tasks, seed_base: int = 20261200,
                             max_tasks: int = 15) -> dict[str, Any]:
    """Build + execution-validate perturbed probes; only probes whose
    renamed reference solution passes its own tests are used."""
    from cortex.evaluation.runner import grade_single

    built, rejected = [], []
    for i, task in enumerate(tasks[:max_tasks]):
        pt = perturb_task(task, seed_base + i)
        if pt is None:
            rejected.append({"task_id": task.task_id,
                             "reason": "no def/class found"})
            continue
        rec = grade_single(pt, pt.reference_solution)
        if rec["passed"]:
            built.append(pt)
        else:
            rejected.append({"task_id": task.task_id,
                             "reason": "perturbed reference failed "
                                       "execution",
                             "stderr_tail": rec["stderr_tail"][-160:]})
    return {"probes": built, "rejected": rejected,
            "n_valid_probes": len(built)}


def parse_similarity_to_reference(generated_code: str,
                                  reference_code: str) -> dict[str, Any]:
    """How close is a free-running generation to the (reference) program:
    token-set similarity + identifier overlap - used as a memorization
    signal (does the model copy familiar code) alongside parse state."""
    gen_words = _norm_wordset(generated_code)
    ref_words = _norm_wordset(reference_code)
    try:
        ast.parse(generated_code)
        parses = True
    except (SyntaxError, ValueError):
        parses = False
    return {
        "parses": parses,
        "word_jaccard_vs_reference": round(jaccard(gen_words, ref_words), 4),
        "n_words_generated": len(gen_words),
    }
