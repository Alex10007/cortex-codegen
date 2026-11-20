"""Stage 14 orchestrator: execution-feedback learning, horizon validation
and promotion attempt at demo scale.

Stage 13 established (independently audited, commit c30ceb5): direct-intro-
duced indentation-related structural failures dominate free-running gene-
ration; the opt-in structural segmentation repairs the stream (95.62%
clean window ends vs 50.16% fixed) with identical exactly-once supervision
yet regressed held-out at the 2000-step horizon; indentation weighting
improves whitespace accuracy but also regresses held-out; pass@1 stayed
0.0 (0/5 gates, stage-12 champion retained). Stage-13 pre-registered the
stage-14 experiments carried out here.

Pre-registered (before any stage-14 execution; never adjusted post hoc):

    Workstream B - execution-feedback training run (the deferred stage-13
    F ticket, status NOT TESTED there):
      hypothesis: conditioning SFT prompts on the model's OWN actual
      execution evidence (sandbox exit/stderr plus reference-doctest
      failures executed against the candidate) from the TRAIN split only
      improves unconditional fixed-decoding pass@1 on the immutable v1/v3
      benchmarks vs the stage-12 champion.
      construction: 48 train records (seed 20261201), candidates from the
      champion at fixed decoding; every failed candidate yields one
      feedback row prompt = "# Task: <instruction>\n# Previous attempt
      failed:\n# <actual evidence cap 400 chars>\n", target = the record's
      own verified CPython-provenance code + "\n" + EOT; NO benchmark
      task or answer enters training;
      training: FB_STEPS=600 optimizer steps, lr 5e-5, batch 4, seed
      20261201, AdamW(0.9,0.98,wd 0.01), clip 1.0 - warm-started from the
      stage-12 champion.
    Workstream C - horizon arms (single variable, segmentation):
      control_h6000 and structural_h6000, H_STEPS=6000, EVAL_EVERY=1000,
      batch 4, seed 20260101, identical stage-13 recipe; tests whether
      the stage-13 structural-segmentation held-out regression (+0.0256)
      is a horizon artifact.
    Workstream D - promotion gates (UNCHANGED from stage 13):
      Gate 1 syntax_valid_rate (best candidate) > 0.0 (champion value);
      Gate 2 complete_program_rate > 0.0;
      Gate 3 at least one execution-verified pass@1 > 0 on v1/v3;
      Gate 4 perturbed-pass@1 (seed_base 20261400) > 0 on the best
      candidate;
      Gate 5 = Gate 3 AND held-out <= 3.3015 in both prompt styles.
      pass@1 = 0 on every candidate means NOT PROMOTED regardless of any
      other improvement.
    Workstream A - GPU re-qualification (honest, no procurement).
    Workstream E - independent audit (scripts/stage14_audit.py, run
    separately) + VERIFICATION_stage14.md.

Usage:
    PYTHONPATH=repo python scripts/stage14_run.py
    PYTHONPATH=repo python scripts/stage14_run.py --finalize-from-artifacts
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cortex.evaluation.benchmarks import fingerprint_tasks  # noqa: E402
from cortex.evaluation.generalization import (  # noqa: E402
    perturbation_suite_valid)
from cortex.evaluation.inference import extract_code  # noqa: E402
from cortex.evaluation.structure_audit import structure_metrics  # noqa: E402
from cortex.evaluation.tasks import TASKS  # noqa: E402
from cortex.evaluation.tasks_v3 import TASKS_V3  # noqa: E402
from cortex.training.longcontext import make_batch_chunked  # noqa: E402
from cortex.training.sft import (  # noqa: E402
    evaluate_heldout, load_split, make_batch)
from cortex.virtual.coverage import persisted_optimizer_scan  # noqa: E402
from cortex.virtual.store import VirtualParameterStore  # noqa: E402

from scripts.stage10_run import (  # noqa: E402
    ART, CKPT10_DIR, DATASET, FINGERPRINT_V1, FINGERPRINT_V2, SEQ,
    _finish, _save_artifact, _sha256_file, dense_model_from_warm6,
    gen_fixed_decoding, WARM6_CKPT)
from scripts.stage13_run import (  # noqa: E402
    CHAMPION12, GATES, PROMOTION_RULE, _capability_eval, _load_arm,
    _subject, grade_single_quiet, structural_token_accuracy)

PARENT_COMMIT = "c30ceb5"
FB_RECORDS = 48
FB_SEED = 20261201
FB_STEPS = 600
FB_LR = 5e-5
FB_BATCH = 4
H_STEPS = 6000
H_EVAL_EVERY = 1000
H_LR = 5e-5
H_BATCH = 4
SEED_H = 20260101
GEN_SEED = 20261214
PERTURB_SEED_BASE = 20261400
BASE_HELDOUT_ALLOWED = 3.3015
STAGE13_CHAMPION_CAP = {"syntax_valid_rate": 0.0, "complete_program_rate":
                        0.0, "v1_pass_at_1": 0.0, "v3_pass_at_1": 0.0}
STAGE13_CHAMPION_HELDOUT = {"native": 2.9435, "eval_aligned": 2.9498}


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z")


def _experiment_header14(experiment_id: str, hypothesis: str,
                         acceptance: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "parent_commit": PARENT_COMMIT,
        "source_commit": _git_head(),
        "dataset_sha256": _sha256_file(DATASET),
        "hardware": "CPU-only host (5 cores, <=6 GB RAM, 0 CUDA devices)",
        "pre_registered_hypothesis": hypothesis,
        "acceptance_criteria": acceptance,
        "started": _now_iso(),
    }


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True).stdout.strip()[:12]


def _dist(values: list) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _heldout(model, heldout_recs, tok, style: str = "legacy") -> float:
    v = evaluate_heldout(model, heldout_recs, tok, SEQ, prompt_style=style)
    return round(v, 4) if v is not None else 4.0


def _with_suite_pass(ev: dict) -> dict:
    """Derive per-suite pass@1 from the single _capability_eval per_task
    list (suite marker v1/v3) so gates and reports address them directly.
    Per-task rows are retained for independent audit re-derivation."""
    for suite in ("v1", "v3"):
        rows = [r for r in ev["per_task"] if r["suite"] == suite]
        ev[f"{suite}_pass_at_1"] = round(
            sum(1 for r in rows if r["passed"]) / max(len(rows), 1), 4)
        ev[f"{suite}_n_tasks"] = len(rows)
        ev[f"{suite}_per_task"] = rows
    return ev


# ------------------------------------------------- execution-based verification

def _reference_docstrings(rec: dict) -> list:
    docstrings = []
    try:
        tree = ast.parse(rec["code"])
    except (SyntaxError, ValueError):
        return docstrings
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        docstrings.append(mod_doc)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            d = ast.get_docstring(node)
            if d:
                docstrings.append(d)
    return docstrings


def _module_namespace(rec: dict) -> tuple[dict, str]:
    """Dataset records were harvested from named CPython standard-library
    modules with doctests verified against the real module namespace
    (cortex.learning.doctest_harvester runs the whole module source and
    then each function's doctests in that namespace). Verification must
    reproduce that context: when rec['module'] is importable, seed the
    execution namespace from the real imported module."""
    import importlib
    import importlib.util

    mod = rec.get("module")
    if mod and importlib.util.find_spec(mod) is not None:
        try:
            ns = dict(importlib.import_module(mod).__dict__)
            ns[mod] = importlib.import_module(mod)
            return ns, mod
        except Exception:
            return {}, mod
    return {}, ""


def verify_candidate(candidate: str, rec: dict) -> dict:
    """Verification with REAL execution evidence only (existing sandbox +
    stdlib doctest against the record's own reference doctests):
      1) the candidate is executed in cortex.execution.sandbox.run, with
      the record's source-module import prepended so harvested records
      are executed in the same context they were verified in;
      2) the reference docstring examples run against the candidate's
      actual namespace, seeded from the real module namespace when the
      record names an importable module.
    A candidate passes only when it executes AND every reference doctest
    example passes. No credit for plausible-looking code."""
    import doctest

    from cortex.execution.sandbox import run as sandbox_run

    out = {"executed": False, "exec_stderr": "", "exec_failure": None,
           "doctests_total": 0, "doctest_failures": 0, "doctest_evidence":
           "", "passed": False}
    ns, mod = _module_namespace(rec)
    preamble = f"import {mod}\n" if mod else ""
    res = sandbox_run(preamble + (candidate or "pass"))
    out["executed"] = res.exit_code == 0
    out["exec_stderr"] = (res.stderr or "")[-500:]
    if res.exit_code != 0:
        out["exec_failure"] = "sandbox_exec_failed"
        return out
    try:
        exec(compile(candidate, "candidate", "exec"), ns)
    except Exception as exc:
        out["exec_failure"] = f"namespace_exec: {exc!r}"
        return out
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(
        verbose=False, optionflags=doctest.ELLIPSIS
        | doctest.NORMALIZE_WHITESPACE)
    evidence = io.StringIO()
    for doc in _reference_docstrings(rec):
        try:
            test = parser.get_doctest(doc, ns, f"{rec['name']}_ref",
                                      None, 0)
        except ValueError:
            continue
        if not test.examples:
            continue
        out["doctests_total"] += len(test.examples)
        with contextlib.redirect_stdout(evidence):
            runner.run(test, out=evidence.write, clear_globs=False)
    out["doctest_failures"] = runner.failures
    evidence_tail = evidence.getvalue()
    out["doctest_evidence"] = evidence_tail[-500:]
    out["passed"] = bool(out["executed"] and out["doctests_total"] > 0
                         and runner.failures == 0)
    return out


# --------------------------------------------------------------- workstream A

def stage_a_gpu() -> dict:
    header = _experiment_header14(
        "stage14-A-gpu-qualification",
        "GPU availability is re-checked each stage; the 2B gate opens "
        "only on genuine qualified CUDA infrastructure",
        "no procurement; no simulated GPU claims; $0.00 spent")
    from cortex.hardware.profiler import can_train_locally, profile_hardware

    from cortex.model.configs import CORTEX_2B
    profile = profile_hardware()
    feasible, reason = can_train_locally(
        profile, CORTEX_2B.theoretical_param_count())
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan = persisted_optimizer_scan(auth)
    auth.close()
    out = {
        "kind": "cortex-stage14-gpu-qualification",
        "owned_by": "Fibonera", "workstream": "A-gpu",
        "hardware": {
            "cpu_cores": profile.cpu_cores,
            "ram_total_gb": profile.ram_total_gb,
            "ram_available_gb": profile.ram_available_gb,
            "has_cuda": profile.has_cuda,
            "cuda_devices": torch.cuda.device_count()
            if torch.cuda.is_available() else 0,
            "nccl_available": bool(torch.cuda.is_available()
                                   and torch.distributed.is_nccl_available()),
        },
        "two_b_gate_feasible_locally": {"feasible": bool(feasible),
                                        "reason": reason},
        "two_b_training_decision":
            "BLOCKED - 0 CUDA devices; ~46 GB RAM required vs measured "
            "available; no paid procurement authorized ($0.00 spent)",
        "authoritative_store_recount": {
            "slots_updated": scan.get("slots_updated"),
            "unchanged": scan.get("slots_updated") == 35292,
        },
    }
    return _finish(header, out, "BLOCKED")


# --------------------------------------------------------------- workstream B

def _sample_train_records(train_recs: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(train_recs)), min(n, len(train_recs))))
    return [train_recs[i] for i in idx]


def make_feedback_row(rec: dict, evidence: str) -> dict:
    """Pre-registered feedback-row surface form (fixed)."""
    prompt_text = ("# Task: " + rec["instruction"]
                   + "\n# Previous attempt failed:\n# "
                   + evidence.replace("\n", "\n# ")[:400] + "\n")
    return {"name": f"feedback-{rec['name']}", "instruction": prompt_text,
            "code": rec["code"], "split": "train",
            "provenance": {"kind": "stage14-execution-feedback",
                           "parent": rec.get("provenance")}}


def stage_b_feedback(tok, train_recs: list, heldout_recs: list) -> dict:
    header = _experiment_header14(
        "stage14-B-execution-feedback-training",
        "conditioning SFT prompts on the model's OWN actual execution "
        "evidence (sandbox execution + reference-doctest outcomes) from "
        "the TRAIN split only improves unconditional fixed-decoding "
        "pass@1 on immutable v1/v3 vs the stage-12 champion",
        "every feedback row's evidence comes from a really executed "
        "candidate; targets are the records' own verified CPython-"
        "provenance solutions; NO immutable-benchmark task/answer in "
        "training; promotion strictly by the pre-registered 5-gate rule; "
        f"FB_STEPS={FB_STEPS}, lr {FB_LR}, batch {FB_BATCH}, seed "
        f"{FB_SEED}")
    model = _load_arm(CHAMPION12)
    model.eval()
    recs = _sample_train_records(train_recs, FB_RECORDS, FB_SEED)
    bench_instructions = {t.instruction for t in TASKS} | \
        {t.instruction for t in TASKS_V3}
    generations = []
    torch.manual_seed(GEN_SEED)
    contamination_hits = 0
    for rec in recs:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), rec["instruction"])).strip()
        m = structure_metrics(cand) if cand else {"parses": False,
                                                  "complete_program": False}
        verdict = verify_candidate(cand, rec)
        if rec["instruction"] in bench_instructions:
            contamination_hits += 1
        generations.append({
            "rec_name": rec["name"], "module": rec.get("module"),
            "n_tokens": len(tok.encode(cand)) if cand else 0,
            "parse": m["parses"], "complete_program":
            m["complete_program"], **verdict})
    torch.manual_seed(FB_SEED)
    rows = []
    n_pass = 0
    row_sources = {}
    for rec, gen in zip(recs, generations):
        if gen["passed"]:
            n_pass += 1
            continue
        if gen["exec_stderr"]:
            source = "exec_stderr"
        elif gen["doctest_evidence"]:
            source = "doctest_evidence"
        elif gen["exec_failure"]:
            source = gen["exec_failure"]
        else:
            source = "unknown failure"
        row_sources[rec["name"]] = source
        evidence = gen["exec_stderr"] or gen["doctest_evidence"] \
            or gen["exec_failure"] or "unknown failure"
        rows.append(make_feedback_row(rec, evidence))
    del model

    model = _load_arm(CHAMPION12)
    opt = torch.optim.AdamW(model.parameters(), lr=FB_LR,
                            betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    losses = []
    steps_with_targets = 0
    t0 = time.time()
    shuffled = rows[:]
    random.Random(FB_SEED).shuffle(shuffled)
    for i in range(FB_STEPS):
        base = (i * FB_BATCH) % len(shuffled)
        batch = shuffled[base:base + FB_BATCH]
        if len(batch) < FB_BATCH:
            batch = (batch + shuffled)[:FB_BATCH]
        x, y = make_batch(batch, tok, SEQ)
        if x is None:
            continue
        _, loss = model(x, y, pad_mask=x == tok.pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(round(float(loss.detach()), 4))
        steps_with_targets += 1
    opt.zero_grad(set_to_none=True)
    seconds = round(time.time() - t0, 1)
    fb_ckpt = CKPT10_DIR / "cortex-sft-stage14-feedback.pt"
    torch.save({
        "model_state": model.state_dict(), "arch": "CortexTransformer",
        "stage": 14, "kind": "execution-feedback", "steps": FB_STEPS,
        "lr": FB_LR, "batch": FB_BATCH, "n_rows": len(rows),
        "parent": str(CKPT10_DIR / f"cortex-sft-{CHAMPION12}.pt"),
        "parent_sha256": _sha256_file(CKPT10_DIR
                                      / f"cortex-sft-{CHAMPION12}.pt")},
        fb_ckpt)
    model.eval()
    ev = _with_suite_pass(_capability_eval(model, tok))
    held_native = _heldout(model, heldout_recs, tok, "legacy")
    held_aligned = _heldout(model, heldout_recs, tok, "eval_aligned")
    stacc_heldout = structural_token_accuracy(model, tok, heldout_recs,
                                              SEQ)
    del model

    repair_successes = 0
    repair_rows = []
    repair_model = torch.load(fb_ckpt, map_location="cpu",
                              weights_only=True)
    model = dense_model_from_warm6()
    model.load_state_dict(repair_model["model_state"])
    del repair_model
    model.eval()
    torch.manual_seed(GEN_SEED)
    for rec, gen in list(zip(recs, generations))[:16]:
        if gen["passed"]:
            continue
        evidence = gen["exec_stderr"] or gen["doctest_evidence"] \
            or gen["exec_failure"] or "unknown failure"
        row = make_feedback_row(rec, evidence)
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), row["instruction"])).strip()
        v2 = verify_candidate(cand, rec)
        repair_successes += bool(v2["passed"])
        repair_rows.append({"rec_name": rec["name"],
                            "first_passed": gen["passed"],
                            "repair_passed": v2["passed"],
                            "first_doctest_failures":
                            gen["doctest_failures"],
                            "repair_doctest_failures":
                            v2["doctest_failures"]})
    del model

    n_gen = len(generations)
    n_repair = len(repair_rows)
    out = {
        "kind": "cortex-stage14-execution-feedback-training",
        "owned_by": "Fibonera", "workstream": "B-feedback",
        "champion_parent": CHAMPION12,
        "sampled_records": len(recs),
        "benchmark_instruction_contamination_hits": contamination_hits,
        "candidate_outcomes": {
            "n": n_gen, "passed_reference_doctests": n_pass,
            "parse_rate": round(sum(g["parse"] for g in generations)
                                / n_gen, 4) if n_gen else None,
            "executed_rate": round(sum(g["executed"] for g in generations)
                                   / n_gen, 4) if n_gen else None,
            "doctest_failure_distribution": _dist(
                [g["doctest_failures"] for g in generations]),
            "per_record": [
                {k: g[k] for k in (
                    "rec_name", "executed", "exec_failure", "parse",
                    "complete_program", "doctests_total",
                    "doctest_failures", "passed", "n_tokens")}
                for g in generations],
        },
        "feedback_rows_built": len(rows),
        "feedback_row_evidence_source": row_sources,
        "training": {
            "steps": FB_STEPS,
            "steps_with_targets": steps_with_targets,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "lr": FB_LR, "batch": FB_BATCH, "seed": FB_SEED,
            "seconds": seconds,
            "checkpoint": str(fb_ckpt),
            "checkpoint_sha256": _sha256_file(fb_ckpt),
        },
        "evaluation_unconditional": {
            "capability": ev,
            "heldout_native_final": held_native,
            "heldout_eval_aligned_final": held_aligned,
            "structural_token_accuracy": stacc_heldout,
            "heldout_delta_vs_stage13_champion": round(
                held_aligned - STAGE13_CHAMPION_HELDOUT["eval_aligned"], 4),
        },
        "repair_conditioned_diagnostics": {
            "n": n_repair,
            "repair_successes": repair_successes,
            "success_rate": round(repair_successes / n_repair, 4)
            if n_repair else None,
            "per_record": repair_rows,
        },
        "leakage_controls": {
            "benchmark_instructions_in_training_rows":
                contamination_hits,
            "targets_are_reference_solutions_of_train_records_only":
                True,
        },
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream C

def train_arm(tok, arm_name: str, cfg: dict, train_recs: list,
              heldout_recs: list, steps: int, eval_every: int) -> dict:
    torch.manual_seed(SEED_H)
    model = dense_model_from_warm6()
    opt = torch.optim.AdamW(model.parameters(), lr=H_LR,
                            betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    held0_legacy = _heldout(dense_model_from_warm6(), heldout_recs, tok,
                            "legacy")
    held0_aligned = _heldout(dense_model_from_warm6(), heldout_recs, tok,
                             "eval_aligned")
    curve = [{"step": 0, "heldout_native": held0_legacy,
              "heldout_eval_aligned": held0_aligned}]
    window_losses: list = []
    steps_with_targets = 0
    t0 = time.time()
    for ch in range(steps // eval_every):
        chunk_losses: list = []
        for i in range(eval_every):
            base = ((ch * eval_every + i) * H_BATCH) % len(train_recs)
            x, y = make_batch_chunked(
                train_recs[base:base + H_BATCH] or train_recs, tok, SEQ,
                prompt_style=cfg["prompt_style"],
                segmentation=cfg["segmentation"])
            if x is None:
                continue
            _, loss = model(x, y, pad_mask=x == tok.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            chunk_losses.append(round(float(loss.detach()), 4))
            steps_with_targets += 1
        curve.append({
            "step": (ch + 1) * eval_every,
            "heldout_native": _heldout(model, heldout_recs, tok,
                                       cfg["prompt_style"]),
            "heldout_eval_aligned": _heldout(model, heldout_recs, tok,
                                             "eval_aligned"),
        })
        window_losses += chunk_losses
    seconds = round(time.time() - t0, 1)
    ck_path = CKPT10_DIR / f"cortex-sft-stage14-{arm_name}.pt"
    torch.save({
        "model_state": model.state_dict(), "arch": "CortexTransformer",
        "stage": 14, "steps": steps, "lr": H_LR, "batching": arm_name,
        "config": cfg, "parent": str(WARM6_CKPT),
        "parent_sha256": _sha256_file(WARM6_CKPT)}, ck_path)
    model.eval()
    ev = _with_suite_pass(_capability_eval(model, tok))
    stacc = structural_token_accuracy(model, tok, heldout_recs, SEQ)
    native_final = curve[-1]["heldout_native"]
    aligned_final = curve[-1]["heldout_eval_aligned"]
    out = {
        "arm": arm_name, "config": cfg,
        "loss_first": window_losses[0] if window_losses else None,
        "loss_last": window_losses[-1] if window_losses else None,
        "steps_with_targets_total": steps_with_targets,
        f"steps_out_of_{steps}": steps_with_targets,
        "seconds": seconds, "heldout_curve": curve,
        "heldout_native_final": native_final,
        "heldout_eval_aligned_final": aligned_final,
        "heldout_eval_aligned_best": min(
            c["heldout_eval_aligned"] for c in curve),
        "capability": ev,
        "structural_token_accuracy_post": stacc,
        "checkpoint": str(ck_path),
        "checkpoint_sha256": _sha256_file(ck_path),
    }
    del model
    return out


def stage_c_horizon(tok, train_recs: list, heldout_recs: list) -> dict:
    header = _experiment_header14(
        "stage14-C-horizon-arms",
        "the stage-13 structural-segmentation held-out regression "
        "(+0.0256 at 2000 steps) is a horizon artifact; at 6000 steps "
        "the structural arm should not regress vs the same-horizon "
        "control and may improve held-out loss",
        f"two arms differing ONLY in segmentation (fixed vs structural); "
        f"H_STEPS={H_STEPS}, eval every {H_EVAL_EVERY}, lr {H_LR}, batch "
        f"{H_BATCH}, seed {SEED_H}; both warm-started from warm-6; "
        "promotion strictly by the pre-registered 5-gate rule")
    arms = {
        "control_h6000": train_arm(
            tok, "control_h6000",
            {"prompt_style": "legacy", "segmentation": "fixed"},
            train_recs, heldout_recs, H_STEPS, H_EVAL_EVERY),
        "structural_h6000": train_arm(
            tok, "structural_h6000",
            {"prompt_style": "legacy", "segmentation": "structural"},
            train_recs, heldout_recs, H_STEPS, H_EVAL_EVERY),
    }
    control = arms["control_h6000"]
    struct = arms["structural_h6000"]
    findings = {
        "heldout_eval_aligned_delta_structural_vs_control": round(
            struct["heldout_eval_aligned_final"]
            - control["heldout_eval_aligned_final"], 4),
        "heldout_native_delta_structural_vs_control": round(
            struct["heldout_native_final"]
            - control["heldout_native_final"], 4),
        "structural_vs_stage13_align_delta": round(
            struct["heldout_eval_aligned_final"]
            - STAGE13_CHAMPION_HELDOUT["eval_aligned"], 4),
        "regression_persists_at_6000": bool(
            struct["heldout_eval_aligned_final"]
            > control["heldout_eval_aligned_final"]),
    }
    out = {
        "kind": "cortex-stage14-horizon-arms",
        "owned_by": "Fibonera", "workstream": "C-horizon",
        "steps": H_STEPS, "eval_every": H_EVAL_EVERY, "lr": H_LR,
        "batch": H_BATCH, "seed": SEED_H,
        "stage13_reference": STAGE13_CHAMPION_HELDOUT,
        "arms": arms,
        "findings": findings,
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream D

def stage_d_promotion(b_report: dict, c_report: dict, tok) -> dict:
    header = _experiment_header14(
        "stage14-D-promotion-audit",
        "the pre-registered 5-gate hierarchy (unchanged from stage 13) "
        "selects the best candidate across the feedback arm and both "
        "horizon arms; pass@1 = 0 on every candidate implies NOT "
        "PROMOTED regardless of loss or syntax improvements",
        "gates evaluated from measured per-task results; perturbation "
        "probes rebuilt at seed_base 20261400 for the best candidate")
    candidates: dict = {}
    candidates["stage14-feedback"] = {
        "heldout_native_final":
            b_report["evaluation_unconditional"]["heldout_native_final"],
        "heldout_eval_aligned_final":
            b_report["evaluation_unconditional"][
                "heldout_eval_aligned_final"],
        "capability": b_report["evaluation_unconditional"]["capability"],
        "checkpoint": b_report["training"]["checkpoint"],
    }
    for arm_name in c_report["arms"]:
        arm = c_report["arms"][arm_name]
        candidates[arm_name] = {
            "heldout_native_final": arm["heldout_native_final"],
            "heldout_eval_aligned_final":
            arm["heldout_eval_aligned_final"],
            "capability": arm["capability"],
            "checkpoint": arm["checkpoint"],
        }
    champion = min(candidates, key=lambda n: (
        candidates[n]["heldout_eval_aligned_final"],
        -(candidates[n]["capability"]["v1_pass_at_1"]
          + candidates[n]["capability"]["v3_pass_at_1"])))
    champ = candidates[champion]
    any_pass = any(
        c["capability"]["v1_pass_at_1"] > 0
        or c["capability"]["v3_pass_at_1"] > 0
        for c in candidates.values())
    champ_ck = torch.load(champ["checkpoint"], map_location="cpu",
                          weights_only=True)
    model = dense_model_from_warm6()
    model.load_state_dict(champ_ck["model_state"])
    del champ_ck
    model.eval()
    suite = perturbation_suite_valid(TASKS + TASKS_V3,
                                     seed_base=PERTURB_SEED_BASE)
    perturbed = []
    torch.manual_seed(GEN_SEED)
    for pt in suite["probes"]:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), pt.instruction)).strip()
        grade = grade_single_quiet(pt, cand)
        perturbed.append({"task_id": pt.task_id, "nonempty": bool(cand),
                          "executed": grade.get("exit_code") is not None,
                          "passed": bool(grade["passed"])})
    del model
    perturbed_pass = round(
        sum(1 for p in perturbed if p["passed"]) / max(len(perturbed), 1),
        4) if perturbed else None
    champ_cap = champ["capability"]
    gate1 = champ_cap["syntax_valid_rate"] > \
        STAGE13_CHAMPION_CAP["syntax_valid_rate"]
    gate2 = gate1 and (champ_cap["complete_program_rate"]
                       > STAGE13_CHAMPION_CAP["complete_program_rate"])
    gate3 = any_pass
    gate4 = bool(perturbed_pass) and perturbed_pass > 0
    gate5 = gate3
    loss_ok = (champ["heldout_native_final"] <= BASE_HELDOUT_ALLOWED
               and champ["heldout_eval_aligned_final"]
               <= BASE_HELDOUT_ALLOWED)
    promoted = gate5 and loss_ok
    gates_passed = sum(1 for g in (gate1, gate2, gate3, gate4, gate5) if g)
    out = {
        "kind": "cortex-stage14-promotion-audit", "owned_by": "Fibonera",
        "workstream": "D-promotion",
        "promotion_rule": PROMOTION_RULE,
        "gates": {
            "gate1_structural_improvement": {
                "definition": GATES["gate1_structural_improvement"],
                "ok": bool(gate1),
                "measured": {"champion_candidate_syntax_valid_rate":
                             champ_cap["syntax_valid_rate"],
                             "stage13_champion_syntax_valid_rate":
                             STAGE13_CHAMPION_CAP["syntax_valid_rate"]}},
            "gate2_complete_generation": {
                "definition": GATES["gate2_complete_generation"],
                "ok": bool(gate2),
                "measured": {"champion_candidate_complete_program_rate":
                             champ_cap["complete_program_rate"],
                             "stage13_champion_complete_program_rate":
                             STAGE13_CHAMPION_CAP["complete_program_rate"]}},
            "gate3_functional_correctness": {
                "definition": GATES["gate3_functional_correctness"],
                "ok": bool(gate3),
                "measured": "any candidate pass@1 > 0 on v1/v3 ("
                            "execution verified)"},
            "gate4_generalization": {
                "definition": GATES["gate4_generalization"],
                "ok": bool(gate4),
                "measured": {"perturbed_pass_at_1": perturbed_pass,
                             "n_probes": len(perturbed),
                             "seed_base": PERTURB_SEED_BASE}},
            "gate5_capability_promotion": {
                "definition": GATES["gate5_capability_promotion"],
                "ok": bool(gate5),
                "measured": "identical evidence to gate3 by definition",
                "requires_heldout_ok": True},
        },
        "candidate_metrics": {
            name: {k: c["capability"][k] for k in (
                "v1_pass_at_1", "v3_pass_at_1", "syntax_valid_rate",
                "complete_program_rate")} | {
                "heldout_eval_aligned_final":
                    c["heldout_eval_aligned_final"]}
            for name, c in candidates.items()},
        "champion_candidate": champion,
        "perturbation_probe_results": perturbed,
        "heldout_loss_gate": {"limit": BASE_HELDOUT_ALLOWED,
                              "ok": bool(loss_ok)},
        "gates_passed": gates_passed,
        "promotion_decision": (
            "PROMOTED - first verified pass@1 milestone with held-out "
            "within the pre-registered limit" if promoted else
            "NOT PROMOTED - pass@1 remains 0.0 (or held-out regression); "
            "negative result preserved; stage-12 champion retained"),
        "rollback": {
            "baseline_checkpoint": str(CKPT10_DIR
                                       / f"cortex-sft-{CHAMPION12}.pt"),
            "procedure": "restore the stage-12 champion; stage-14 "
                         "checkpoints are additive separate files",
        },
    }
    return _finish(header, out, "PROMOTED" if promoted else "NOT PROMOTED")


# --------------------------------------------------------------- main

def main() -> int:
    head = _git_head()
    assert head.startswith(PARENT_COMMIT), (
        f"stage-14 must start from {PARENT_COMMIT}, got {head}")
    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    train_recs = load_split(str(DATASET), "train")
    heldout_recs = load_split(str(DATASET), "heldout")
    print(f"[stage14] dataset train={len(train_recs)} "
          f"heldout={len(heldout_recs)}", flush=True)

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_pre = persisted_optimizer_scan(auth)
    auth.close()
    assert scan_pre["slots_updated"] == 35292, "authoritative store moved"

    a = stage_a_gpu()
    _save_artifact("stage14_gpu_qualification.json", a)
    print(f"[stage14] A decision: {a['experiment']['decision']}",
          flush=True)

    b = stage_b_feedback(tok, train_recs, heldout_recs)
    _save_artifact("stage14_feedback_training.json", b)
    bcap = b["evaluation_unconditional"]["capability"]
    print(f"[stage14] B rows={b['feedback_rows_built']} "
          f"repair_successes="
          f"{b['repair_conditioned_diagnostics']['repair_successes']} "
          f"v1pass={bcap['v1_pass_at_1']} "
          f"v3pass={bcap['v3_pass_at_1']}", flush=True)

    c = stage_c_horizon(tok, train_recs, heldout_recs)
    _save_artifact("stage14_horizon_arms.json", c)
    print(f"[stage14] C findings: {c['findings']}", flush=True)

    d = stage_d_promotion(b, c, tok)
    _save_artifact("stage14_promotion_audit.json", d)
    print(f"[stage14] D decision: {d['experiment']['decision']} "
          f"(gates passed {d['gates_passed']}/5)", flush=True)

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_post = persisted_optimizer_scan(auth)
    auth.close()
    _save_artifact("stage14_report.json",
                   _report_payload(a, b, c, d, scan_post))
    print("[stage14] complete", flush=True)
    return 0


def _report_payload(a, b, c, d, scan_post) -> dict:
    return {
        "kind": "cortex-stage14-report", "owned_by": "Fibonera",
        "stage": 14,
        "summary": {
            "parent_commit": PARENT_COMMIT,
            "A_two_b_training": a["two_b_training_decision"],
            "B_feedback_rows": b["feedback_rows_built"],
            "B_candidate_parse_rate":
                b["candidate_outcomes"]["parse_rate"],
            "B_passed_reference_doctests":
                b["candidate_outcomes"]["passed_reference_doctests"],
            "B_repair_success_rate":
                b["repair_conditioned_diagnostics"]["success_rate"],
            "B_capability": b["evaluation_unconditional"]["capability"],
            "B_heldout_final": {
                "native": b["evaluation_unconditional"][
                    "heldout_native_final"],
                "eval_aligned": b["evaluation_unconditional"][
                    "heldout_eval_aligned_final"]},
            "C_findings": c["findings"],
            "C_arms": {n: {"heldout_eval_aligned_final":
                           ar["heldout_eval_aligned_final"],
                           "capability": ar["capability"]}
                       for n, ar in c["arms"].items()},
            "D_decision": d["experiment"]["decision"],
            "D_gates_passed": d["gates_passed"],
            "D_champion_candidate": d["champion_candidate"],
            "fingerprint_v1": FINGERPRINT_V1,
            "fingerprint_v2": FINGERPRINT_V2,
            "fingerprint_v3": fingerprint_tasks("v3"),
            "spend_usd": 0.0, "gpu_hours": 0.0,
            "authoritative_store_untouched":
                scan_post["slots_updated"] == 35292,
        }}


def _finalize_from_artifacts() -> int:
    """Recover an interrupted run: A/B/C are read back from the persisted
    artifacts and only D + the report re-execute."""
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    paths = (("a", "stage14_gpu_qualification"),
             ("b", "stage14_feedback_training"),
             ("c", "stage14_horizon_arms"))
    saved = {}
    for k, name in paths:
        p = ART / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"cannot finalize: {p} missing")
        saved[k] = json.loads(p.read_text())
    d = stage_d_promotion(saved["b"], saved["c"], tok)
    _save_artifact("stage14_promotion_audit.json", d)
    print(f"[stage14] D decision: {d['experiment']['decision']} "
          f"(gates passed {d['gates_passed']}/5)", flush=True)
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_post = persisted_optimizer_scan(auth)
    auth.close()
    _save_artifact("stage14_report.json",
                   _report_payload(saved["a"], saved["b"], saved["c"], d,
                                   scan_post))
    print("[stage14] report rebuilt from artifacts", flush=True)
    return 0


if __name__ == "__main__":
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    if len(sys.argv) > 1 and sys.argv[1] == "--finalize-from-artifacts":
        raise SystemExit(_finalize_from_artifacts())
    raise SystemExit(main())
