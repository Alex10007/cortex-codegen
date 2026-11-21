"""Stage 16 orchestrator: autoregressive structural learning,
complete-program generation, execution-grounded correction and
generalizable coding capability.

All workstreams run against the real demo model, real tokenizer, real
training coordinator and real execution sandbox. Nothing is simulated.
$0.00 is spent and the authoritative 500B virtual store is never moved.

PHASE 2 PRE-REGISTERED EXPOSURE DESIGN (fixed BEFORE execution):
  Exposure rows are built ONCE from the shared arms init (warm-6): with
  fixed low-temperature decoding (0.2) the model generates up to
  EXPOSURE_PREFIX_TOKENS tokens conditioned on the record's legacy
  prompt. A row is usable for aligned exposure ONLY if the generated
  prefix is an EXACT token-prefix of the reference completion; rows
  whose prefix diverges are (a) skipped for aligned-exposure arms with
  provenance and (b) TRUNCATED AT THE FIRST DIVERGENCE for the
  error-conditioned arm, whose context is prompt + the model's own
  correct-so-far tokens and whose supervision is the verified reference
  continuation from that point. Context positions (prompt + prefix) are
  masked -100; ONLY verified continuation tokens receive gradients.
  Prefix provenance recorded per record.

PHASE 11 PRE-REGISTERED PROMOTION GATES (thresholds fixed BEFORE any
stage-16 execution; never changed after seeing results):
  Gate A (reproducibility): exposure_sched improves the 8-probe
    free-running parse rate over control (>= +GATE_B_PARSE_DELTA) at the
    primary seed AND at the independent rep2 seed.
  Gate B (free-running structure): primary-arm parse rate on the 8
    structural probes > 0 AND >= control + GATE_B_PARSE_DELTA.
  Gate C (complete programs): primary capability complete_program_rate
    (immutable suites) > 0.
  Gate D (functional correctness): primary v1/v3/t15-unseen pass@1 > 0
    (execution-verified).
  Gate E (generalization): t16 pass@1 > 0 AND perturbed pass@1 > 0.
  Gate F (self-correction): at least 1 repair success whose corrected
    program passes the original tests.
  Gate G (regression control): primary held-out native and eval_aligned
    <= G_HELDOUT_LIMIT AND no material regression vs the stage-15
    structural reference (held-out <= reference + 0.05).
  Gate H (independent audit): scripts/stage16_audit.py reports all
    checks PASS.
  PROMOTED iff A..H all PASS. Otherwise NOT PROMOTED and the stage-12
  champion is retained with rollback recorded.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cortex.evaluation.autoregressive_diagnostics import (  # noqa: E402
    classify_structural_failure, structure_scan, trajectory_generate)
from cortex.evaluation.benchmarks import fingerprint_tasks  # noqa: E402
from cortex.evaluation.generalization import (  # noqa: E402
    perturbation_suite_valid)
from cortex.evaluation.inference import extract_code  # noqa: E402
from cortex.evaluation.moe_probe import probe as moe_probe  # noqa: E402
from cortex.evaluation.structure_audit import structure_metrics  # noqa: E402
from cortex.evaluation.tasks import TASKS  # noqa: E402
from cortex.evaluation.tasks_v3 import TASKS_V3  # noqa: E402
from cortex.model.configs import CORTEX_S  # noqa: E402
from cortex.model.transformer import CortexTransformer  # noqa: E402
from cortex.training.longcontext import (  # noqa: E402
    make_batch_chunked, record_ids, structural_cut_summary)
from cortex.training.sft import load_split, make_batch  # noqa: E402
from cortex.virtual.coverage import persisted_optimizer_scan  # noqa: E402
from cortex.virtual.store import VirtualParameterStore  # noqa: E402

from scripts.stage10_run import (  # noqa: E402
    ART, CKPT10_DIR, DATASET, FINGERPRINT_V1, FINGERPRINT_V2, SEQ,
    _finish, _save_artifact, _sha256_file, dense_model_from_warm6,
    gen_fixed_decoding)
from scripts.stage13_run import (  # noqa: E402
    grade_single_quiet, _weighted_ce)
from scripts.stage14_run import (  # noqa: E402
    _capability_eval, _heldout, _load_arm, _sample_train_records,
    _subject, _with_suite_pass, structural_token_accuracy, verify_candidate)
from scripts.stage15_run import (  # noqa: E402
    STAGE15_REFERENCE_ARM, STRUCTURAL_PROMPTS15, UNSEEN_TASKS_STAGE15,
    classify_execution_failure, compact_evidence, feedback_coverage,
    fingerprint_unseen, indent_transition_stats, make_feedback_row15)
from scripts.stage9_run import build_moe_subject  # noqa: E402

PARENT_COMMIT = "799a217"
DECODING16 = {"temperature": 0.2, "top_k": 50, "max_new_tokens": 180}
PREFIX_DECODING16 = {"temperature": 0.2, "top_k": 50, "max_new_tokens": 48}
E_STEPS = 3000
E_EVAL_EVERY = 500
E_LR = 5e-5
E_BATCH = 4
EXPOSURE_PREFIX_TOKENS = 48
EXPOSURE_ATTEMPT_RECORDS = 96
EXPOSURE_GEN_SEED = 20261810
ARMS_SEEDS = {
    "control": 20261811,
    "exposure_const": 20261812,
    "exposure_sched": 20261813,
    "exposure_sched_rep2": 20261891,
    "exposure_early": 20261814,
    "exposure_weight": 20261815,
    "complete_curriculum": 20261816,
}
FB16_RECORDS = 12
FB16_STEPS = 600
FB16_SEED = 20261821
FB16_GEN_SEED = 20261820
GEN_SEED16 = 20261830
PERTURB_SEED_BASE16 = 20261900
G_HELDOUT_LIMIT = 3.3015
GATE_B_PARSE_DELTA = 0.25
REPAIR_ROUNDS = 2
S15_FEEDBACK_HELDOUT_REF = 3.5957
CHAMPION_STAGE12 = "stage12-chunked_control"
STAGE16_PRIMARY_ARM = "exposure_sched"
CKPT_PREFIX16 = "cortex-sft-stage16"
FB16_ARM_NAME = "feedback_grounded"


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z")


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True).stdout.strip()[:12]


def _experiment_header16(experiment_id: str, hypothesis: str,
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


# --------------------------------------------------------------- phase 1

BASELINE_CHECKPOINTS = {
    "stage15_structural_champion": STAGE15_REFERENCE_ARM,
    "stage15_control": "stage15-control_rep2",
    "stage12_champion": CHAMPION_STAGE12,
}


def _max_training_prefix_overlap(gen_ids: list, train_recs: list, tok) -> dict:
    """Does the free-running generation reproduce a training-record
    completion fragment? Longest common token prefix against every train
    record completion (real token ids, no simulation)."""
    best_ratio = 0.0
    best_name = None
    gen_n = max(len(gen_ids), 1)
    for rec in train_recs:
        comp = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
        m = 0
        for a, b in zip(gen_ids, comp):
            if a != b:
                break
            m += 1
        ratio = m / gen_n
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = rec["name"]
    return {"max_prefix_overlap_ratio": round(best_ratio, 4),
            "best_matching_record": best_name}


def stage_1_baseline(tok, train_recs: list, heldout_recs: list) -> dict:
    header = _experiment_header16(
        "stage16-1-baseline-and-failure-reproduction",
        "the stage-15 reference arm's held-out, structural and free-running "
        "measurements reproduce under the identical stage-16 protocol, and "
        "its 8 free-running structural probes still fail at the FIRST "
        "indentation transition (stage-15 B: 0/8 parse, "
        "unexpected_indentation_transition)",
        "3 historical checkpoints re-hashed, re-loaded and re-evaluated with "
        "identical held-out/structural/capability recipes; instrumented "
        "trajectories (trajectory_generate + classify_structural_failure) "
        "record first-invalid-token evidence, EOT stats and training-fragment "
        "overlap; no repair, no grammar constraints, decoding never changed")
    checkpoints = {}
    for role, name in sorted(BASELINE_CHECKPOINTS.items()):
        p = CKPT10_DIR / f"cortex-sft-{name}.pt"
        entry = {"checkpoint": str(p), "exists": p.exists(),
                 "sha256": _sha256_file(p) if p.exists() else None}
        if not p.exists():
            checkpoints[role] = entry
            continue
        ck = torch.load(p, map_location="cpu", weights_only=True)
        model = _load_arm(name)
        model.eval()
        stacc = structural_token_accuracy(model, tok, heldout_recs, SEQ)
        ind = indent_transition_stats(model, tok, heldout_recs)
        ev = _with_suite_pass(_capability_eval(model, tok))
        entry.update({
            "arch": ck.get("arch"), "stage": ck.get("stage"),
            "training_steps": ck.get("steps"), "parent": ck.get("parent"),
            "config": ck.get("config"),
            "heldout_native": _heldout(model, heldout_recs, tok),
            "heldout_eval_aligned": _heldout(model, heldout_recs, tok,
                                             "eval_aligned"),
            "structural_token_accuracy": stacc,
            "indent_transition_accuracy": ind["accuracy"],
            "capability": {k: ev[k] for k in (
                "n_tasks", "pass_at_1", "v1_pass_at_1", "v3_pass_at_1",
                "syntax_valid_rate", "complete_program_rate",
                "execution_attempted_rate")},
            "capability_per_task": ev["per_task"],
        })
        checkpoints[role] = entry
        del model, ck

    reference_role = "stage15_structural_champion"
    model = _load_arm(BASELINE_CHECKPOINTS[reference_role])
    model.eval()
    torch.manual_seed(GEN_SEED16)
    probes = []
    for label, prompt in STRUCTURAL_PROMPTS15:
        traj = trajectory_generate(model, tok, prompt, DECODING16)
        text = traj["text"]
        m = structure_metrics(extract_code(text)) if text.strip() \
            else {"parses": False, "complete_program": False}
        cls = classify_structural_failure(traj)
        scan = structure_scan(text or "")
        probes.append({
            "probe": label, "parse": m["parses"],
            "complete_program": m["complete_program"],
            "taxonomy": cls.get("taxonomy"), "evidence":
            cls.get("evidence"),
            "stopped": traj.get("stopped"), "eot_step":
            traj.get("eot_step"),
            "first_invalid": traj.get("first_invalid"),
            "n_generated_tokens": traj.get("completion_length_tokens"),
            "lex_state_end": {
                "paren_depth": scan.get("paren_depth"),
                "open_quote": scan.get("open_quote"),
                "unbalanced": scan.get("unbalanced")},
            "training_fragment": _max_training_prefix_overlap(
                traj["generated_token_ids"], train_recs, tok),
        })
    del model
    n = len(probes)
    reproduction = {
        "n_probes": n,
        "parse_rate": round(sum(p["parse"] for p in probes) / n, 4),
        "complete_program_rate": round(
            sum(p["complete_program"] for p in probes) / n, 4),
        "eot_stop_count": sum(p["stopped"] == "eot" for p in probes),
        "failures_at_first_indent_transition": sum(
            p["taxonomy"] == "unexpected_indentation_transition"
            for p in probes),
        "training_fragment_continuations": sum(
            p["training_fragment"]["max_prefix_overlap_ratio"] >= 0.5
            for p in probes),
        "per_probe": probes,
    }
    reproduces = (reproduction["parse_rate"] == 0.0
                  and reproduction["failures_at_first_indent_transition"]
                  >= 6)
    out = {
        "kind": "cortex-stage16-baseline", "owned_by": "Fibonera",
        "workstream": "1-baseline",
        "checkpoints": checkpoints,
        "free_running_failure_reproduction": reproduction,
        "stage15_reference_findings": {
            "indent_transition_accuracy_stage15": 0.9608,
            "free_running_parse_rate_stage15": 0.0,
        },
    }
    return _finish(header, out,
                   "PASS" if reproduces
                   else "PARTIAL - stage-15 free-running failure NOT "
                        "reproduced exactly; measurement preserved")


# ------------------------------------------ exposure dataset + arm trainer

def _rows_batch(rows: list, tok, seq_len: int):
    pad_id = tok.pad_id
    batch_ids: list = []
    batch_labels: list = []
    for r in rows:
        ids = list(r["ids"][:seq_len])
        labels = list(r["labels"][:seq_len])
        ids_pad = ids + [pad_id] * (seq_len - len(ids))
        lab_pad = labels + [-100] * (seq_len - len(labels))
        batch_ids.append(ids_pad[:-1])
        batch_labels.append(lab_pad[1:])
    return (torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_labels, dtype=torch.long))


def build_exposure_dataset(tok, train_recs: list) -> dict:
    prefix_model = dense_model_from_warm6()
    prefix_model.eval()
    recs = _sample_train_records(train_recs, EXPOSURE_ATTEMPT_RECORDS,
                                 EXPOSURE_GEN_SEED)
    aligned_rows: list = []
    early_rows: list = []
    provenance: list = []
    skip_reasons: dict = {}
    div_positions: list = []
    prefix_lengths: list = []

    def skip(reason: str, name: str, extra: dict | None = None):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        provenance.append({"rec_name": name, "state": "skipped",
                           "reason": reason, **(extra or {})})

    for i, rec in enumerate(recs):
        prompt, comp = record_ids(rec, tok, "legacy")
        if len(prompt) + len(comp) + 1 > SEQ:
            skip("does_not_fit_seq256", rec["name"])
            continue
        torch.manual_seed(EXPOSURE_GEN_SEED + i)
        traj = trajectory_generate(prefix_model, tok,
                                   rec["instruction"] + "\n",
                                   PREFIX_DECODING16)
        gen = list(traj["generated_token_ids"])[:EXPOSURE_PREFIX_TOKENS]
        if not gen:
            skip("empty_prefix", rec["name"])
            continue
        aligned = gen == comp[:len(gen)]
        div = None if aligned else next(
            (j for j in range(len(gen)) if gen[j] != comp[j]), None)
        if div is not None:
            div_positions.append(div)
        prefix_lengths.append(len(gen))
        taxonomy = None if aligned else classify_structural_failure(
            traj).get("taxonomy")
        if len(gen) >= len(comp):
            skip("prefix_covers_full_completion", rec["name"])
            continue
        if aligned:
            aligned_rows.append({
                "rec_name": rec["name"], "ids": prompt + comp,
                "labels": [-100] * (len(prompt) + len(gen))
                + comp[len(gen):],
                "n_prefix_tokens": len(gen),
                "provenance": {
                    "kind": "stage16-autoregressive-exposure-aligned",
                    "prefix_source": "warm6",
                    "prefix_generation_seed": EXPOSURE_GEN_SEED + i,
                    "decoding": PREFIX_DECODING16,
                    "exact_token_prefix_matched": True}})
        elif div == 0:
            skip("prefix_diverged_immediately_no_context", rec["name"], {
                "first_divergence_position": 0,
                "prefix_failure_taxonomy": taxonomy})
        else:
            skip("prefix_diverged_truncated_to_early_row", rec["name"], {
                "first_divergence_position": div,
                "prefix_failure_taxonomy": taxonomy})
            early_rows.append({
                "rec_name": rec["name"], "ids": prompt + comp,
                "labels": [-100] * (len(prompt) + div) + comp[div:],
                "n_prefix_tokens": div,
                "provenance": {
                    "kind": "stage16-autoregressive-exposure-early",
                    "prefix_source": "warm6",
                    "prefix_generation_seed": EXPOSURE_GEN_SEED + i,
                    "decoding": PREFIX_DECODING16,
                    "context_is_model_prefix_truncated_at_first_"
                    "divergence": True,
                    "prefix_failure_taxonomy": taxonomy,
                    "exact_token_prefix_matched": False}})
    del prefix_model
    return {
        "attempted": len(recs),
        "aligned_rows": aligned_rows,
        "early_rows": early_rows,
        "provenance": provenance,
        "skip_reasons": skip_reasons,
        "aligned_rows_count": len(aligned_rows),
        "early_rows_count": len(early_rows),
        "divergence_positions": {
            "n_diverged": len(div_positions),
            "mean_first_divergence": round(
                sum(div_positions) / len(div_positions), 2)
            if div_positions else None,
            "max_first_divergence": max(div_positions) if div_positions
            else None},
        "prefix_lengths_mean": round(
            sum(prefix_lengths) / len(prefix_lengths), 2)
        if prefix_lengths else None,
    }


def _exposure_rate(rate_key: str, step: int, steps: int) -> float:
    if rate_key == "zero":
        return 0.0
    if rate_key == "const":
        return 0.5
    if rate_key == "sched":
        return 0.25 + 0.5 * (step / steps)
    raise ValueError(f"unknown exposure rate: {rate_key}")


def free_running_probes(model, tok) -> dict:
    torch.manual_seed(GEN_SEED16)
    probes = []
    for label, prompt in STRUCTURAL_PROMPTS15:
        traj = trajectory_generate(model, tok, prompt, DECODING16)
        text = traj["text"]
        m = structure_metrics(extract_code(text)) if text.strip() \
            else {"parses": False, "complete_program": False}
        cls = classify_structural_failure(traj)
        probes.append({
            "probe": label, "parse": m["parses"],
            "complete_program": m["complete_program"],
            "taxonomy": cls.get("taxonomy"), "stopped":
            traj.get("stopped"), "eot_step": traj.get("eot_step"),
            "n_generated_tokens": traj.get("completion_length_tokens"),
            "first_invalid": traj.get("first_invalid")})
    n = len(probes)
    return {"n_probes": n,
            "parse_rate": round(sum(p["parse"] for p in probes) / n, 4),
            "complete_program_rate": round(
                sum(p["complete_program"] for p in probes) / n, 4),
            "per_probe": probes}


def stage16_train_arm(tok, arm_name: str, cfg: dict, train_recs: list,
                      heldout_recs: list, steps: int = E_STEPS,
                      eval_every: int = E_EVAL_EVERY,
                      seed: int | None = None,
                      exposure_rows: list | None = None,
                      fitting_recs: list | None = None,
                      pool_recs: list | None = None,
                      checkpoint_prefix: str = CKPT_PREFIX16) -> dict:
    t0 = time.time()
    if seed is None:
        seed = ARMS_SEEDS.get(arm_name, 20261899)
    torch.manual_seed(seed)
    model = dense_model_from_warm6()
    if cfg.get("init_from"):
        ck = torch.load(cfg["init_from"], map_location="cpu",
                        weights_only=True)
        model.load_state_dict(ck["model_state"])
        ck = None
    held0_native = _heldout(model, heldout_recs, tok)
    held0_aligned = _heldout(model, heldout_recs, tok, "eval_aligned")
    opt = torch.optim.AdamW(model.parameters(), lr=E_LR,
                            betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    curve = [{"step": 0, "heldout_native": held0_native,
              "heldout_eval_aligned": held0_aligned}]
    rng = random.Random(seed + 777)
    rate_key = cfg.get("rate", "zero")
    window_losses: list = []
    sup_tokens: list = []
    grad_norms: list = []
    exposure_steps = 0
    target_bearing_steps = 0
    fit_pool = fitting_recs or train_recs
    data_pool = pool_recs if pool_recs is not None else train_recs
    switch_step = (cfg.get("curriculum") or {}).get("switch_step", steps)
    for step in range(steps):
        in_fit_phase = cfg.get("curriculum") is not None \
            and step < switch_step
        if in_fit_phase:
            pool, seg = fit_pool, "fixed"
        else:
            pool = data_pool
            seg = "structural" if cfg.get("curriculum") is not None \
                else "fixed"
        rate = _exposure_rate(rate_key, step, steps)
        if rate > 0 and exposure_rows and rng.random() < rate:
            rows = rng.sample(exposure_rows,
                              min(E_BATCH, len(exposure_rows)))
            x, y = _rows_batch(rows, tok, SEQ)
            src = "exposure"
        else:
            base = (step * E_BATCH) % len(pool)
            x, y = make_batch_chunked(
                pool[base:base + E_BATCH] or pool, tok, SEQ,
                prompt_style="legacy", segmentation=seg)
            src = "dataset"
        if x is None:
            continue
        if cfg.get("weighted_loss"):
            logits, _ = model(x, pad_mask=x == tok.pad_id)
            loss = _weighted_ce(logits[:, :-1, :].contiguous(),
                                y[:, 1:].contiguous(), tok, 3.0)
        else:
            _, loss = model(x, y, pad_mask=x == tok.pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % 100 == 0:
            grads = [p.grad for p in model.parameters()
                     if p.grad is not None]
            grad_norms.append(round(float(torch.norm(torch.stack(
                [g.norm() for g in grads]))), 4))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if src == "exposure":
            exposure_steps += 1
        if bool((y != -100).any()):
            target_bearing_steps += 1
        if step % 10 == 0:
            sup_tokens.append(int((y != -100).sum()))
        window_losses.append(round(float(loss.detach()), 4))
        if (step + 1) % eval_every == 0:
            curve.append({"step": step + 1,
                          "heldout_native": _heldout(model, heldout_recs,
                                                     tok),
                          "heldout_eval_aligned": _heldout(
                              model, heldout_recs, tok, "eval_aligned")})
    seconds = round(time.time() - t0, 1)
    ck_path = CKPT10_DIR / f"{checkpoint_prefix}-{arm_name}.pt"
    torch.save({
        "model_state": model.state_dict(), "arch": "CortexTransformer",
        "stage": 16, "steps": steps, "lr": E_LR, "batching": arm_name,
        "config": cfg, "seed": seed,
        "parent": cfg.get("init_from") or "warm-6",
        "parent_sha256": _sha256_file(cfg["init_from"])
        if cfg.get("init_from") else None}, ck_path)
    model.eval()
    ev = _with_suite_pass(_capability_eval(model, tok))
    stacc = structural_token_accuracy(model, tok, heldout_recs, SEQ)
    probes = free_running_probes(model, tok)
    out = {
        "arm": arm_name, "config": cfg, "seed": seed,
        "steps": steps, "lr": E_LR, "batch": E_BATCH,
        "heldout_native_initial": held0_native,
        "heldout_eval_aligned_initial": held0_aligned,
        "curve": curve,
        "heldout_native_final": curve[-1]["heldout_native"],
        "heldout_eval_aligned_final": curve[-1]["heldout_eval_aligned"],
        "loss_first": window_losses[0] if window_losses else None,
        "loss_last": window_losses[-1] if window_losses else None,
        "target_bearing_steps": target_bearing_steps,
        "exposure_step_frac": round(exposure_steps / max(steps, 1), 4),
        "supervised_tokens_per_step_mean": round(
            sum(sup_tokens) / len(sup_tokens), 1) if sup_tokens else None,
        "grad_norm_samples": grad_norms,
        "grad_norm_mean": round(sum(grad_norms) / len(grad_norms), 4)
        if grad_norms else None,
        "capability": {k: ev[k] for k in (
            "n_tasks", "pass_at_1", "v1_pass_at_1", "v3_pass_at_1",
            "syntax_valid_rate", "complete_program_rate",
            "execution_attempted_rate")},
        "structural_token_accuracy": stacc,
        "free_running_probes": probes,
        "seconds": seconds,
        "checkpoint": str(ck_path),
    }
    return out


def stage_2_exposure(tok, train_recs: list, heldout_recs: list) -> dict:
    header = _experiment_header16(
        "stage16-2-autoregressive-exposure",
        "training with model-generated prefixes as CONTEXT (only exact "
        "token-prefixes of verified reference completions; context masked "
        "-100) makes teacher-forced structural prediction transfer to "
        "free-running generation; error-conditioned early-prefix exposure "
        "and scheduled exposure change the transfer differently; the "
        "matched control (exposure rate 0) is trained with identical "
        "compute, seed lineage and evaluation",
        "matched 3000-step arms warm-started from warm-6 with identical "
        "fixed-segmentation control stream; exposure rows verified exact "
        "token-prefixes with full provenance; divergence rows skipped "
        "(aligned) or truncated at first divergence (early) with failure "
        "classification; context masked -100, only verified continuation "
        "supervised; pre-registered seeds; promotion strictly by the "
        "phase-11 gates")
    data = build_exposure_dataset(tok, train_recs)
    print(f"[stage16] exposure dataset aligned={data['aligned_rows_count']} "
          f"early={data['early_rows_count']} "
          f"skips={data['skip_reasons']}", flush=True)
    arms = {
        "control": stage16_train_arm(
            tok, "control", {"rate": "zero",
                             "data_pool": "dataset_v3_train_fixed"},
            train_recs, heldout_recs),
        "exposure_const": stage16_train_arm(
            tok, "exposure_const",
            {"rate": "const", "data_pool": "dataset_v3_train_fixed"},
            train_recs, heldout_recs, exposure_rows=data["aligned_rows"]),
        "exposure_sched": stage16_train_arm(
            tok, "exposure_sched",
            {"rate": "sched", "data_pool": "dataset_v3_train_fixed"},
            train_recs, heldout_recs, exposure_rows=data["aligned_rows"]),
        "exposure_sched_rep2": stage16_train_arm(
            tok, "exposure_sched_rep2",
            {"rate": "sched", "data_pool": "dataset_v3_train_fixed"},
            train_recs, heldout_recs,
            seed=ARMS_SEEDS["exposure_sched_rep2"],
            exposure_rows=data["aligned_rows"]),
        "exposure_early": stage16_train_arm(
            tok, "exposure_early",
            {"rate": "const", "data_pool": "dataset_v3_train_fixed"},
            train_recs, heldout_recs, exposure_rows=data["early_rows"]),
    }
    ctrl_pr = arms["control"]["free_running_probes"]["parse_rate"]
    best = None
    for name, arm in arms.items():
        if name == "control":
            continue
        delta = round(arm["free_running_probes"]["parse_rate"]
                      - ctrl_pr, 4)
        if best is None or delta > best[1]:
            best = (name, delta)
    findings = {
        "control_parse_rate": ctrl_pr,
        "parse_rate_by_arm": {
            n: a["free_running_probes"]["parse_rate"]
            for n, a in sorted(arms.items())},
        "heldout_eval_aligned_by_arm": {
            n: a["heldout_eval_aligned_final"]
            for n, a in sorted(arms.items())},
        "best_exposure_arm": best[0] if best else None,
        "best_parse_rate_delta_over_control": best[1] if best else None,
        "phase2_acceptance": bool(
            best is not None and best[1] >= GATE_B_PARSE_DELTA),
    }
    out = {
        "kind": "cortex-stage16-autoregressive-exposure",
        "owned_by": "Fibonera", "workstream": "2-exposure",
        "exposure_dataset": data,
        "arms": arms,
        "findings": findings,
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- phase 3

def first_transition_divergence(model, tok, recs) -> dict:
    """Teacher-forced: for each held-out example, the completion-token
    index of the FIRST indentation-transition position where the model's
    prediction disagrees with the reference token (None when every
    transition matches)."""
    model.eval()
    firsts: list = []
    with torch.no_grad():
        for rec in recs:
            comp = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
            prompt_ids = tok.encode(rec["instruction"] + "\n")
            if len(prompt_ids) + len(comp) + 1 > SEQ:
                continue
            ids = prompt_ids + comp
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            logits, _ = model(x, pad_mask=x == tok.pad_id)
            preds = logits[0].argmax(-1).tolist()
            cum = [""]
            for t in comp:
                cum.append(cum[-1] + tok.decode([t]))
            code_text = rec["code"].strip() + "\n"
            lines = code_text.split("\n")
            n_lines = len(lines) - 1
            first = None
            for j in range(len(comp)):
                if not cum[j].endswith("\n"):
                    continue
                line_idx = cum[j].count("\n")
                if line_idx <= 0 or line_idx >= n_lines:
                    continue
                nxt = lines[line_idx]
                prev = lines[line_idx - 1]
                if (len(nxt) - len(nxt.lstrip(" "))
                        == len(prev) - len(prev.lstrip(" "))):
                    continue
                if not nxt.strip():
                    continue
                if preds[len(prompt_ids) + j - 1] != comp[j]:
                    first = j
                    break
            firsts.append(first)
    hit = [f for f in firsts if f is not None]
    return {
        "n_examples": len(firsts),
        "n_with_transition_error": len(hit),
        "examples_with_all_transitions_correct": len(firsts) - len(hit),
        "first_error_position_mean": round(sum(hit) / len(hit), 1)
        if hit else None,
        "first_error_position_max": max(hit) if hit else None,
    }


def stage_3_structural(tok, heldout_recs: list, exposure: dict) -> dict:
    header = _experiment_header16(
        "stage16-3-structural-transition-learning",
        "the exposure arms' free-running improvement is coupled to a "
        "measurable change in teacher-forced structural prediction "
        "(first-transition divergence and indent-transition accuracy), "
        "not only to lower loss",
        "control vs pre-registered primary arm (exposure_sched) compared "
        "on: teacher-forced indent-transition accuracy, per-class "
        "structural accuracy, FIRST transition divergence position on the "
        "real held-out split, and the free-running parse rates from the "
        "arms' instrumented probes; a structural objective is accepted "
        "only if free-running parse rate changes, never on token metrics "
        "alone; no separate grammar engine, decoder unchanged")
    ctrl = exposure["arms"]["control"]
    prim = exposure["arms"][STAGE16_PRIMARY_ARM]
    m_ctrl = _load_arm(f"stage16-{ctrl['arm']}")
    m_prim = _load_arm(f"stage16-{prim['arm']}")
    div_ctrl = first_transition_divergence(m_ctrl, tok, heldout_recs)
    div_prim = first_transition_divergence(m_prim, tok, heldout_recs)
    ind_ctrl = indent_transition_stats(m_ctrl, tok, heldout_recs)
    ind_prim = indent_transition_stats(m_prim, tok, heldout_recs)
    rep2 = exposure["arms"]["exposure_sched_rep2"]
    out = {
        "kind": "cortex-stage16-structural-learning", "owned_by":
        "Fibonera", "workstream": "3-structural",
        "primary_arm": STAGE16_PRIMARY_ARM,
        "comparison": {
            "control": {
                "indent_transition_accuracy": ind_ctrl["accuracy"],
                "structural_token_accuracy":
                ctrl["structural_token_accuracy"],
                "first_transition_divergence": div_ctrl,
                "free_running_parse_rate":
                ctrl["free_running_probes"]["parse_rate"]},
            "primary": {
                "indent_transition_accuracy": ind_prim["accuracy"],
                "structural_token_accuracy":
                prim["structural_token_accuracy"],
                "first_transition_divergence": div_prim,
                "free_running_parse_rate":
                prim["free_running_probes"]["parse_rate"]},
            "rep2_free_running_parse_rate":
            rep2["free_running_probes"]["parse_rate"]},
        "primary_free_running_probes": prim["free_running_probes"],
    }
    del m_ctrl, m_prim
    improved_teacher = (
        out["comparison"]["primary"]["indent_transition_accuracy"]
        or 0) >= (out["comparison"]["control"][
            "indent_transition_accuracy"] or 0)
    free_changed = bool(
        prim["free_running_probes"]["parse_rate"]
        > ctrl["free_running_probes"]["parse_rate"])
    out["findings"] = {
        "teacher_forced_not_regressed": bool(improved_teacher),
        "free_running_changed": free_changed,
        "note": "phase-3 acceptance requires free-running change; "
                "teacher-forced-only improvement is NOT accepted",
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- phase 4

def _record_length_stats(recs: list, tok,
                         seq_lengths=(256, 512, 1024)) -> dict:
    rows = []
    for rec in recs:
        prompt, comp = record_ids(rec, tok, "legacy")
        rows.append({"name": rec["name"], "prompt_tokens": len(prompt),
                     "completion_tokens": len(comp),
                     "total_tokens": len(prompt) + len(comp) + 1})
    totals = [r["total_tokens"] for r in rows]
    out = {"n_records": len(rows),
           "total_tokens_mean": round(
               sum(totals) / max(len(totals), 1), 1),
           "total_tokens_max": max(totals, default=0),
           "total_tokens_over_256": sum(t > 256 for t in totals),
           "retention": {}}
    for L in seq_lengths:
        out["retention"][str(L)] = round(
            sum(t <= L for t in totals) / max(len(totals), 1), 4)
    return out


def cortex_s_feasibility_probe(train_recs: list, tok) -> dict:
    """Real forward/backward on the EXISTING declared CORTEX_S config
    (41,997,312 params, max_seq_len 1024): measured memory and time per
    optimizer step. A feasibility probe only - never claimed as
    full-scale training."""
    model = CortexTransformer(CORTEX_S)
    n_params = sum(p.numel() for p in model.parameters())
    fitting = []
    for rec in train_recs:
        x, y = make_batch([rec], tok, 512)
        if x is not None:
            fitting.append((x, y))
        if len(fitting) == 2:
            break
    if not fitting:
        del model
        return {"error": "no fitting records at seq 512"}
    xs = torch.cat([x for x, _ in fitting], dim=0)
    ys = torch.cat([y for _, y in fitting], dim=0)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    times: list = []
    loss_val = None
    for _ in range(3):
        t0 = time.time()
        _, loss = model(xs, ys, pad_mask=xs == tok.pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        times.append(time.time() - t0)
        loss_val = round(float(loss.detach()), 4)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res = {
        "preset": {"name": CORTEX_S.name,
                   "theoretical_param_count":
                   CORTEX_S.theoretical_param_count(),
                   "max_seq_len": CORTEX_S.max_seq_len,
                   "dim": CORTEX_S.dim, "n_layers": CORTEX_S.n_layers,
                   "status": str(CORTEX_S.status)},
        "live_param_count": n_params,
        "param_count_matches_declared":
        n_params == CORTEX_S.theoretical_param_count(),
        "probe_seq_width": int(xs.shape[1]),
        "probe_batch": int(xs.shape[0]),
        "loss_last": loss_val,
        "secs_per_step_mean": round(sum(times) / len(times), 2),
        "secs_per_step_list": [round(t, 2) for t in times],
        "peak_rss_delta_mb": round((rss1 - rss0) / 1024, 1),
        "context_within_declared_max":
        int(xs.shape[1]) <= CORTEX_S.max_seq_len,
    }
    del model, opt, xs, ys, fitting
    gc.collect()
    return res


def stage_4_complete_program(tok, train_recs: list, heldout_recs: list,
                             exposure: dict) -> dict:
    header = _experiment_header16(
        "stage16-4-complete-program-and-context",
        "the stage-15 finding that only 9 of 32 execution-verified "
        "records fit SEQ=256 is a context-capacity artifact: a "
        "complete-record curriculum (fitting-only records first, then "
        "structure-preserving segmentation over ALL records) changes "
        "complete-program generation, and the declared CORTEX_S config "
        "admits real long-context steps within CPU memory",
        "measured length distributions and retention at 256/512/1024 from "
        "real tokenization; the pre-registered complete_curriculum arm "
        "(1500 fitting-only steps then 1500 structural all-records steps, "
        f"seed {ARMS_SEEDS['complete_curriculum']}) compared to the "
        "matched 3000-step control; CORTEX_S feasibility probe executes "
        "REAL forward/backward steps (probe only, never claimed as "
        "full-scale training)")
    lengths = _record_length_stats(train_recs + heldout_recs, tok)
    cuts = structural_cut_summary(train_recs, tok, SEQ)
    fitting_recs = []
    for rec in train_recs:
        x, _y = make_batch([rec], tok, SEQ)
        if x is not None:
            fitting_recs.append(rec)
    arm = stage16_train_arm(
        tok, "complete_curriculum",
        {"rate": "zero", "curriculum": {"switch_step": E_STEPS // 2},
         "data_pool": "dataset_v3_train_structural",
         "fit_pool": "complete_fitting_records_256"},
        train_recs, heldout_recs, fitting_recs=fitting_recs)
    probe = cortex_s_feasibility_probe(train_recs, tok)
    out = {
        "kind": "cortex-stage16-complete-program", "owned_by":
        "Fibonera", "workstream": "4-complete-program",
        "record_length_stats": lengths,
        "structural_cut_summary": cuts,
        "fitting_records_at_256": len(fitting_recs),
        "curriculum_arm": arm,
        "control_arm_summary": {
            k: exposure["arms"]["control"][k] for k in (
                "heldout_eval_aligned_final", "target_bearing_steps",
                "supervised_tokens_per_step_mean",
                "exposure_step_frac")},
        "cortex_s_feasibility_probe": probe,
        "findings": {
            "curriculum_vs_control_parse_rate": round(
                arm["free_running_probes"]["parse_rate"]
                - exposure["arms"]["control"]["free_running_probes"]
                ["parse_rate"], 4),
            "curriculum_vs_control_heldout": round(
                arm["heldout_eval_aligned_final"]
                - exposure["arms"]["control"]
                ["heldout_eval_aligned_final"], 4),
        },
    }
    del arm
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- phase 5

def _grad_vector(model, x, y, kind: str, tok):
    model.zero_grad(set_to_none=True)
    if kind == "lm":
        _, loss = model(x, y, pad_mask=x == tok.pad_id)
    elif kind == "structural_weighted":
        logits, _ = model(x)
        loss = _weighted_ce(logits[:, :-1, :].contiguous(),
                            y[:, 1:].contiguous(), tok, 3.0)
    elif kind == "combined":
        _, loss = model(x, y, pad_mask=x == tok.pad_id)
        logits, _ = model(x)
        lw = _weighted_ce(logits[:, :-1, :].contiguous(),
                          y[:, 1:].contiguous(), tok, 3.0)
        loss = 0.5 * loss + 0.5 * lw
    else:
        raise ValueError(kind)
    loss.backward()
    vec = torch.cat([p.grad.flatten().clone()
                     for p in model.parameters() if p.grad is not None])
    loss_val = round(float(loss.detach()), 4)
    model.zero_grad(set_to_none=True)
    return vec, loss_val


def stage_5_objective(tok, exposure: dict, train_recs: list,
                      heldout_recs: list) -> dict:
    header = _experiment_header16(
        "stage16-5-objective-and-optimization",
        "the structural-transition-weighted objective (measured in "
        "isolation in stages 13/15) combined with autoregressive "
        "exposure on the SAME batches grants measurable gradient support "
        "for structure without degrading ordinary token prediction: the "
        "combined arm differs from the exposure-only arm in free-running "
        "structure",
        "REAL gradient interference measured on one fixed model state "
        "(exposure_const endpoint) for plain CE, structural-weighted CE "
        "(_weighted_ce, stage-13 formulation) and their combination: "
        "grad norms + pairwise cosine similarity recorded; ONE real "
        "3000-step optimizer arm (exposure_weight, pre-registered seed) "
        "implements the single untested combination; completion-only "
        "masking asserted everywhere; no arbitrary weights beyond the "
        "stage-13/15 measured 3.0")
    model = _load_arm("stage16-exposure_const")
    data = exposure["exposure_dataset"]
    rows = (data["aligned_rows"] + data["early_rows"])[:8]
    x, y = _rows_batch(rows, tok, SEQ)
    vecs = {}
    losses = {}
    for kind in ("lm", "structural_weighted", "combined"):
        v, l = _grad_vector(model, x, y, kind, tok)
        vecs[kind] = v
        losses[kind] = l

    def cos(a: str, b: str) -> float:
        num = float(torch.dot(vecs[a], vecs[b]))
        den = float(vecs[a].norm() * vecs[b].norm()) or 1.0
        return round(num / den, 4)

    interference = {
        "cosine_lm_vs_structural_weighted": cos(
            "lm", "structural_weighted"),
        "cosine_lm_vs_combined": cos("lm", "combined"),
        "cosine_structural_weighted_vs_combined": cos(
            "structural_weighted", "combined"),
    }
    norms = {k: round(float(v.norm()), 4) for k, v in vecs.items()}
    del model
    vecs = None
    arm = stage16_train_arm(
        tok, "exposure_weight",
        {"rate": "const", "weighted_loss": True,
         "data_pool": "dataset_v3_train_fixed"},
        train_recs, heldout_recs,
        exposure_rows=exposure["exposure_dataset"]["aligned_rows"])
    ctrl_pr = exposure["arms"]["control"]["free_running_probes"][
        "parse_rate"]
    out = {
        "kind": "cortex-stage16-objective-ablation", "owned_by":
        "Fibonera", "workstream": "5-objective",
        "gradient_interference": {
            "conditions_run": ["lm", "structural_weighted",
                               "combined"],
            "loss_values": losses, "grad_norms": norms,
            "cosine_similarity": interference,
            "model_state": "stage16-exposure_const endpoint",
            "batch": "first 8 exposure rows (aligned + early)"},
        "exposure_weight_arm": arm,
        "findings": {
            "weighted_arm_vs_control_parse_rate": round(
                arm["free_running_probes"]["parse_rate"] - ctrl_pr, 4),
            "weighted_arm_vs_exposure_const_parse_rate": round(
                arm["free_running_probes"]["parse_rate"]
                - exposure["arms"]["exposure_const"]
                ["free_running_probes"]["parse_rate"], 4),
            "weighted_arm_heldout_eval_aligned":
            arm["heldout_eval_aligned_final"],
        },
    }
    del arm
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- phase 6

def stage_6_execution_feedback(tok, train_recs: list, heldout_recs: list,
                               exposure: dict) -> dict:
    header = _experiment_header16(
        "stage16-6-execution-grounded-training",
        "execution-grounded training with full Task->Generated->"
        "Execution->Evidence->Diagnosis->Verified-correction lineage and "
        "guaranteed target coverage improves held-out over the stage-15 "
        f"feedback-SFT regression reference ({S15_FEEDBACK_HELDOUT_REF}) "
        "when warm-started from the exposure-trained base instead of the "
        "stage-12 champion",
        f"{FB16_RECORDS} train records x fixed decoding from the "
        "pre-registered primary arm; every verdict carries REAL sandbox "
        "evidence; failing candidates become compact-evidence rows "
        "(stage-15 compaction reused) whose corrected source is the "
        "record's own CPython-harvested solution (provenance-verified, "
        "contamination-checked); per-row coverage audit; ONE 600-step "
        f"optimizer arm warm-started from stage16-{STAGE16_PRIMARY_ARM} "
        f"(seed {FB16_SEED}); no synthetic successes; the stage-15 "
        "negative result remains in the experiment history")
    init_from = CKPT10_DIR / f"{CKPT_PREFIX16}-{STAGE16_PRIMARY_ARM}.pt"
    recs = _sample_train_records(train_recs, FB16_RECORDS, FB16_GEN_SEED)
    bench_instructions = {t.instruction for t in TASKS} | \
        {t.instruction for t in TASKS_V3}
    model = _load_arm(f"stage16-{STAGE16_PRIMARY_ARM}")
    model.eval()
    candidates = []
    rows = []
    torch.manual_seed(FB16_GEN_SEED)
    for rec in recs:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), rec["instruction"])).strip()
        verdict = verify_candidate(cand, rec)
        err_cls = classify_execution_failure(verdict)
        candidates.append({
            "rec_name": rec["name"],
            "instruction_sha256": hashlib.sha256(
                rec["instruction"].encode()).hexdigest(),
            "generated_source": cand,
            "generated_source_sha256": hashlib.sha256(
                cand.encode()).hexdigest() if cand else None,
            "source_checkpoint": str(init_from),
            "source_checkpoint_sha256": _sha256_file(init_from),
            "generation_config": DECODING16,
            "contaminated": rec["instruction"] in bench_instructions,
            "verdict": {
                "passed": verdict.get("passed"),
                "executed": verdict.get(
                    "executed", verdict.get("exit_code") is not None),
                "exit_code": verdict.get("exit_code"),
                "exec_stderr_tail": (verdict.get("exec_stderr")
                                     or "")[-400:],
                "doctest_failures": verdict.get("doctest_failures"),
                "doctests_total": verdict.get("doctests_total")},
            "error_class": err_cls,
            "failure_category": err_cls})
        if not verdict["passed"] and cand:
            evidence = compact_evidence(verdict)
            fb_row = make_feedback_row15(rec, evidence)
            coverage = feedback_coverage(fb_row, tok)
            rows.append({
                **fb_row,
                "lineage": {
                    "task_rec_name": rec["name"],
                    "task_instruction_sha256": hashlib.sha256(
                        rec["instruction"].encode()).hexdigest(),
                    "generated_source_sha256": hashlib.sha256(
                        cand.encode()).hexdigest(),
                    "source_checkpoint_sha256": _sha256_file(init_from),
                    "execution_error_class": err_cls,
                    "compact_evidence": evidence,
                    "corrected_source_is_record_reference": True,
                    "corrected_source_provenance": rec.get("provenance"),
                    "contaminated":
                    rec["instruction"] in bench_instructions},
                "coverage_audit": coverage})
    del model
    coverage_ok = all(r["coverage_audit"]["exact_match"]
                      or not r["coverage_audit"]["fits"]
                      for r in rows) if rows else None
    arm = stage16_train_arm(
        tok, FB16_ARM_NAME,
        {"rate": "zero", "init_from": str(init_from),
         "data_pool": "stage16_compacted_execution_feedback_rows"},
        train_recs, heldout_recs, steps=FB16_STEPS, eval_every=100,
        seed=FB16_SEED, pool_recs=rows)
    out = {
        "kind": "cortex-stage16-execution-feedback", "owned_by":
        "Fibonera", "workstream": "6-execution-feedback",
        "primary_arm": STAGE16_PRIMARY_ARM,
        "candidates": candidates,
        "candidate_error_class_distribution": _dist(
            [c["error_class"] for c in candidates]),
        "success_count": sum(c["verdict"]["passed"]
                             for c in candidates),
        "contaminated_candidates": sum(c["contaminated"]
                                       for c in candidates),
        "feedback_rows": rows,
        "coverage_audit_all_rows_target_bearing": coverage_ok,
        "feedback_arm": arm,
        "stage15_negative_reference": {
            "stage15_feedback_arm_heldout_eval_aligned":
            S15_FEEDBACK_HELDOUT_REF,
            "stage15_target_bearing_steps": "600/600 after compaction"},
        "findings": {
            "regression_vs_stage15_reference": round(
                arm["heldout_eval_aligned_final"]
                - S15_FEEDBACK_HELDOUT_REF, 4),
            "target_bearing_steps": arm["target_bearing_steps"],
        },
    }
    del arm
    return _finish(header, out, "EXECUTED")


def _dist(values: list) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------- phase 7

def _grade_v1(task, cand: str) -> dict:
    g = grade_single_quiet(task, cand)
    return {"passed": bool(g["passed"]),
            "executed": g.get("exit_code") is not None,
            "stderr_tail": g.get("stderr_tail", "")[-300:]}


def stage_7_self_correction(tok, exposure: dict, unseen_tasks: list) -> dict:
    header = _experiment_header16(
        "stage16-7-generation-and-self-correction",
        "the complete generate->execute->diagnose->revise->re-execute "
        "loop with the exposure-trained model improves "
        "execution-verified success within REPAIR_ROUNDS; every repair "
        "uses ONLY the model's own revision conditioned on compacted "
        "real error evidence",
        f"v1 (3 tasks) + t16-unseen (8 tasks); up to {REPAIR_ROUNDS} "
        "correction rounds; every transition measured independently "
        "(executed, revision differs, re-executed, passed); success "
        "counted ONLY on the original task's real tests; model-only vs "
        "system-with-execution-tool separated in the report; seed "
        f"{GEN_SEED16}; no external repair, no grammar constraints")
    tasks = ([("v1", t) for t in TASKS]
             + [("t16-unseen", t) for t in unseen_tasks])
    model = _load_arm(f"stage16-{STAGE16_PRIMARY_ARM}")
    model.eval()
    rows = []
    torch.manual_seed(GEN_SEED16)
    for suite, t in tasks:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), t.instruction)).strip()
        g = _grade_unseen16(t, cand) if suite == "t16-unseen" \
            else _grade_v1(t, cand)
        attempt = {"suite": suite, "task_id": t.task_id,
                   "attempts": 1, "direct_passed": g["passed"],
                   "direct_executed": g["executed"]}
        rounds = []
        current = cand
        for r in range(REPAIR_ROUNDS):
            if g["passed"] or not current:
                break
            evidence = compact_evidence({
                "passed": False, "executed": g["executed"],
                "exec_failure": None, "exec_stderr": g["stderr_tail"],
                "doctest_failures": 0, "doctests_total": 0})
            rep_prompt = ("# Task: " + t.instruction
                          + "\n# Attempt failed with " + evidence + "\n")
            rcand = extract_code(gen_fixed_decoding(
                _subject(tok, model), rep_prompt)).strip()
            g = _grade_unseen16(t, rcand) if suite == "t16-unseen" \
                else _grade_v1(t, rcand)
            rounds.append({
                "round": r + 1, "evidence": evidence,
                "revision_differs": rcand != current,
                "executed": g["executed"], "passed": g["passed"]})
            current = rcand
            attempt["attempts"] = r + 2
        attempt["repair_rounds"] = rounds
        attempt["final_passed"] = g["passed"]
        rows.append(attempt)
    del model
    successful = sum(r["final_passed"] for r in rows)
    direct = sum(r["direct_passed"] for r in rows)
    n_repaired = sum(bool(r["repair_rounds"]) for r in rows)
    out = {
        "kind": "cortex-stage16-self-correction", "owned_by":
        "Fibonera", "workstream": "7-self-correction",
        "primary_arm": STAGE16_PRIMARY_ARM,
        "n_tasks": len(rows), "max_repair_rounds": REPAIR_ROUNDS,
        "direct_successes": direct,
        "first_attempt_success_rate": round(
            direct / max(len(rows), 1), 4),
        "success_after_corrections": round(
            successful / max(len(rows), 1), 4),
        "n_repaired": n_repaired,
        "repair_successes": sum(
            1 for r in rows if r["repair_rounds"]
            and r["final_passed"]),
        "average_correction_attempts": round(
            sum(r["attempts"] for r in rows) / max(len(rows), 1), 2),
        "revisions_differing": sum(
            bool(rd["revision_differs"]) for r in rows
            for rd in r["repair_rounds"]),
        "regression_rate": round(sum(
            1 for r in rows if r["repair_rounds"] and r["direct_passed"]
            and not r["final_passed"]) / max(n_repaired, 1), 4),
        "per_task": rows,
    }
    return _finish(header, out, "EXECUTED")


# -------------------------------------------------- phase 8 unseen suite

@dataclass(frozen=True)
class UnseenTask16:
    task_id: str
    instruction: str
    module_name: str
    test_code: str
    reference_solution: str = ""


UNSEEN_TASKS_STAGE16: list = [
    UnseenTask16(
        task_id="t16-binary-add",
        instruction=("Write module `task.py` defining `add_binary(a, b)` "
                     "that adds two binary-number strings and returns the "
                     "sum as a binary string."),
        module_name="task.py",
        test_code=(
            "import pytest\n"
            "from task import add_binary\n\n"
            "def test_basic():\n"
            "    assert add_binary('11', '1') == '100'\n\n"
            "def test_longer():\n"
            "    assert add_binary('1010', '1011') == '10101'\n\n"
            "def test_zero():\n"
            "    assert add_binary('0', '0') == '0'\n"
        ),
        reference_solution=(
            "def add_binary(a, b):\n"
            "    i, j = len(a) - 1, len(b) - 1\n"
            "    carry = 0\n"
            "    out = []\n"
            "    while i >= 0 or j >= 0 or carry:\n"
            "        total = carry\n"
            "        if i >= 0:\n"
            "            total += int(a[i])\n"
            "            i -= 1\n"
            "        if j >= 0:\n"
            "            total += int(b[j])\n"
            "            j -= 1\n"
            "        out.append(str(total % 2))\n"
            "        carry = total // 2\n"
            "    return ''.join(reversed(out))\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-flatten-dict",
        instruction=("Write module `task.py` defining `flatten(d)` that "
                     "flattens a nested dict into a single-level dict with "
                     "dotted keys."),
        module_name="task.py",
        test_code=(
            "from task import flatten\n\n"
            "def test_one_level():\n"
            "    assert flatten({'a': {'b': 1}}) == {'a.b': 1}\n\n"
            "def test_deep():\n"
            "    assert flatten({'a': {'b': {'c': 2}}}) == "
            "{'a.b.c': 2}\n\n"
            "def test_flat():\n"
            "    assert flatten({'x': 1}) == {'x': 1}\n"
        ),
        reference_solution=(
            "def flatten(d, prefix=''):\n"
            "    out = {}\n"
            "    for k, v in d.items():\n"
            "        key = f\"{prefix}.{k}\" if prefix else k\n"
            "        if isinstance(v, dict):\n"
            "            out.update(flatten(v, key))\n"
            "        else:\n"
            "            out[key] = v\n"
            "    return out\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-run-length-encode",
        instruction=("Write module `task.py` defining `encode_rle(s)` "
                     "that returns the run-length encoding of s as a "
                     "list of (character, count) tuples."),
        module_name="task.py",
        test_code=(
            "from task import encode_rle\n\n"
            "def test_basic():\n"
            "    assert encode_rle('aaabb') == [('a', 3), ('b', 2)]\n\n"
            "def test_empty():\n"
            "    assert encode_rle('') == []\n\n"
            "def test_singletons():\n"
            "    assert encode_rle('abc') == [('a', 1), ('b', 1), "
            "('c', 1)]\n"
        ),
        reference_solution=(
            "def encode_rle(s):\n"
            "    if not s:\n"
            "        return []\n"
            "    out = []\n"
            "    count = 1\n"
            "    for i in range(1, len(s)):\n"
            "        if s[i] == s[i - 1]:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append((s[i - 1], count))\n"
            "            count = 1\n"
            "    out.append((s[-1], count))\n"
            "    return out\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-merge-intervals",
        instruction=("Write module `task.py` defining "
                     "`merge_intervals(intervals)` that merges overlapping "
                     "(start, end) tuples into sorted, non-overlapping "
                     "tuples."),
        module_name="task.py",
        test_code=(
            "from task import merge_intervals\n\n"
            "def test_overlap():\n"
            "    assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == \\\n"
            "        [(1, 6), (8, 10)]\n\n"
            "def test_empty():\n"
            "    assert merge_intervals([]) == []\n\n"
            "def test_touching():\n"
            "    assert merge_intervals([(1, 4), (4, 5)]) == [(1, 5)]\n"
        ),
        reference_solution=(
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ivs = sorted(intervals)\n"
            "    out = [list(ivs[0])]\n"
            "    for start, end in ivs[1:]:\n"
            "        if start <= out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], end)\n"
            "        else:\n"
            "            out.append([start, end])\n"
            "    return [tuple(iv) for iv in out]\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-bank-account-class",
        instruction=("Write module `task.py` defining class `Account` "
                     "with `deposit(amount)`, `withdraw(amount)` and a "
                     "`balance` attribute; withdrawing more than the "
                     "balance raises ValueError."),
        module_name="task.py",
        test_code=(
            "import pytest\n"
            "from task import Account\n\n"
            "def test_deposit():\n"
            "    a = Account()\n"
            "    a.deposit(5)\n"
            "    assert a.balance == 5\n\n"
            "def test_withdraw():\n"
            "    a = Account(10)\n"
            "    assert a.withdraw(4) == 6\n"
            "    assert a.balance == 6\n\n"
            "def test_overdraft():\n"
            "    a = Account(1)\n"
            "    with pytest.raises(ValueError):\n"
            "        a.withdraw(5)\n"
        ),
        reference_solution=(
            "class Account:\n"
            "    def __init__(self, balance=0):\n"
            "        self.balance = balance\n\n"
            "    def deposit(self, amount):\n"
            "        self.balance += amount\n"
            "        return self.balance\n\n"
            "    def withdraw(self, amount):\n"
            "        if amount > self.balance:\n"
            "            raise ValueError('insufficient funds')\n"
            "        self.balance -= amount\n"
            "        return self.balance\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-top-words",
        instruction=("Write module `task.py` defining `top_words(text, "
                     "k)` returning the k most frequent words, ties "
                     "broken alphabetically."),
        module_name="task.py",
        test_code=(
            "from task import top_words\n\n"
            "def test_basic():\n"
            "    assert top_words('a b a c a b', 2) == ['a', 'b']\n\n"
            "def test_empty():\n"
            "    assert top_words('', 3) == []\n\n"
            "def test_ties():\n"
            "    assert top_words('b a', 2) == ['a', 'b']\n"
        ),
        reference_solution=(
            "def top_words(text, k):\n"
            "    counts = {}\n"
            "    for w in text.split():\n"
            "        counts[w] = counts.get(w, 0) + 1\n"
            "    ranked = sorted(counts.items(),\n"
            "                    key=lambda kv: (-kv[1], kv[0]))\n"
            "    return [w for w, _ in ranked[:k]]\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-fib-memo",
        instruction=("Write module `task.py` defining `fib_memo(n)` that "
                     "computes the n-th Fibonacci number using "
                     "recursion with memoization."),
        module_name="task.py",
        test_code=(
            "from task import fib_memo\n\n"
            "def test_base():\n"
            "    assert fib_memo(0) == 0\n"
            "    assert fib_memo(1) == 1\n\n"
            "def test_small():\n"
            "    assert fib_memo(10) == 55\n\n"
            "def test_large():\n"
            "    assert fib_memo(30) == 832040\n"
        ),
        reference_solution=(
            "def fib_memo(n, memo=None):\n"
            "    if memo is None:\n"
            "        memo = {}\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    if n not in memo:\n"
            "        memo[n] = fib_memo(n - 1, memo) \\\n"
            "            + fib_memo(n - 2, memo)\n"
            "    return memo[n]\n"
        ),
    ),
    UnseenTask16(
        task_id="t16-rotate-matrix",
        instruction=("Write module `task.py` defining `rotate(matrix)` "
                     "that returns the matrix rotated 90 degrees "
                     "clockwise."),
        module_name="task.py",
        test_code=(
            "from task import rotate\n\n"
            "def test_square():\n"
            "    assert rotate([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]\n\n"
            "def test_rect():\n"
            "    assert rotate([[1, 2, 3], [4, 5, 6]]) == \\\n"
            "        [[4, 1], [5, 2], [6, 3]]\n\n"
            "def test_empty():\n"
            "    assert rotate([]) == []\n"
        ),
        reference_solution=(
            "def rotate(matrix):\n"
            "    if not matrix:\n"
            "        return []\n"
            "    return [list(row) for row in zip(*matrix[::-1])]\n"
        ),
    ),
]


def fingerprint_unseen16() -> str:
    return hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.module_name, t.test_code,
         t.reference_solution)
        for t in UNSEEN_TASKS_STAGE16),
        sort_keys=True).encode("utf-8")).hexdigest()


def _grade_unseen16(task, candidate: str) -> dict:
    from cortex.evaluation.tasks import EvalTask as RealTask

    rt = RealTask(task_id=task.task_id, instruction=task.instruction,
                  module_name=task.module_name, test_code=task.test_code)
    g = grade_single_quiet(rt, candidate)
    return {"task_id": task.task_id, "passed": bool(g["passed"]),
            "executed": g.get("exit_code") is not None,
            "stderr_tail": g.get("stderr_tail", "")[-300:]}


def _validate_unseen_harness16() -> dict:
    results = []
    for t in UNSEEN_TASKS_STAGE16:
        g = _grade_unseen16(t, t.reference_solution)
        results.append({"task_id": t.task_id,
                        "reference_passes": bool(g["passed"])})
    return {"per_task": results,
            "all_references_pass": all(r["reference_passes"]
                                       for r in results)}


def _contamination16(unseen_tasks: list, train_recs: list) -> dict:
    train_texts = {r["instruction"] for r in train_recs}
    bench_texts = {t.instruction for t in TASKS} | \
        {t.instruction for t in TASKS_V3}
    t15_texts = {t.instruction for t in UNSEEN_TASKS_STAGE15}
    return {
        "n_tasks": len(unseen_tasks),
        "instruction_in_train": sum(t.instruction in train_texts
                                    for t in unseen_tasks),
        "instruction_in_benchmarks": sum(
            t.instruction in bench_texts for t in unseen_tasks),
        "instruction_in_stage15_unseen": sum(
            t.instruction in t15_texts for t in unseen_tasks),
    }


def _eval_capability_suites16(model, tok) -> dict:
    rows = []
    torch.manual_seed(GEN_SEED16)
    for suite, tasks, grader in (
            ("v1", TASKS, None),
            ("v3", TASKS_V3, None),
            ("t15-unseen", UNSEEN_TASKS_STAGE15, _grade_unseen16),
            ("t16-unseen", UNSEEN_TASKS_STAGE16, _grade_unseen16)):
        for t in tasks:
            cand = extract_code(gen_fixed_decoding(
                _subject(tok, model), t.instruction)).strip()
            m = structure_metrics(cand) if cand else {
                "parses": False, "complete_program": False}
            if grader is None:
                g = _grade_v1(t, cand)
            else:
                g = grader(t, cand)
            rows.append({"suite": suite, "task_id": t.task_id,
                         "nonempty": bool(cand), "parse": m["parses"],
                         "complete_program": m["complete_program"],
                         "executed": g["executed"],
                         "passed": g["passed"]})
    out = {"per_task": rows}
    for suite in ("v1", "v3", "t15-unseen", "t16-unseen"):
        srows = [r for r in rows if r["suite"] == suite]
        out[f"{suite}_pass_at_1"] = round(
            sum(r["passed"] for r in srows) / max(len(srows), 1), 4)
        out[f"{suite}_n_tasks"] = len(srows)
        out[f"{suite}_syntax_valid_rate"] = round(
            sum(r["parse"] for r in srows) / max(len(srows), 1), 4)
        out[f"{suite}_complete_program_rate"] = round(
            sum(r["complete_program"] for r in srows)
            / max(len(srows), 1), 4)
    return out


def _pass_at_k16(tok) -> dict:
    model = _load_arm(f"stage16-{STAGE16_PRIMARY_ARM}")
    model.eval()
    per_task = []
    torch.manual_seed(GEN_SEED16)
    for t in UNSEEN_TASKS_STAGE16:
        passes = 0
        samples = []
        for si, temp in enumerate((DECODING16["temperature"], 0.7, 1.0)):
            decoding = dict(DECODING16, temperature=temp)
            cand = extract_code(gen_fixed_decoding(
                _subject(tok, model), t.instruction,
                decoding=decoding)).strip()
            g = _grade_unseen16(t, cand)
            passes += int(g["passed"])
            samples.append({"sample": si, "temperature": temp,
                            "passed": g["passed"]})
        per_task.append({"task_id": t.task_id, "passes": passes,
                         "samples": samples})
    del model
    n = len(per_task)
    return {"k": 3, "n_tasks": n,
            "pass_at_1_mean": round(
                sum(t["passes"] for t in per_task) / (3 * n), 4),
            "tasks_with_at_least_one_pass": sum(
                t["passes"] > 0 for t in per_task),
            "per_task": per_task}


def stage_8_generalization(tok, train_recs: list, exposure: dict) -> dict:
    header = _experiment_header16(
        "stage16-8-unseen-generalization",
        "any free-running capability achieved on training-distribution "
        "tasks transfers to the NEW 8-task t16 unseen suite (functions, "
        "classes, recursion, data structures, algorithms) and to "
        "perturbed formulations",
        "8 authored t16 tasks with execution-validated reference "
        "solutions (harness sanity must pass first), fingerprint "
        f"{fingerprint_unseen16()[:16]}..., zero contamination "
        "documented; perturbation probes at seed_base "
        f"{PERTURB_SEED_BASE16}; pass@3 sampling at temperatures "
        "0.2/0.7/1.0; tasks never exposed to training")
    harness = _validate_unseen_harness16()
    if not harness.get("all_references_pass"):
        raise SystemExit("t16 unseen-suite harness sanity failed: "
                         f"{harness}")
    contamination = _contamination16(UNSEEN_TASKS_STAGE16, train_recs)
    model = _load_arm(f"stage16-{STAGE16_PRIMARY_ARM}")
    model.eval()
    cap_full = _eval_capability_suites16(model, tok)
    per_task = cap_full.pop("per_task")
    suite = perturbation_suite_valid(TASKS + TASKS_V3,
                                     seed_base=PERTURB_SEED_BASE16)
    perturbed = []
    for pt in suite["probes"]:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), pt.instruction)).strip()
        g = grade_single_quiet(pt, cand)
        perturbed.append({"task_id": pt.task_id, "nonempty": bool(cand),
                          "executed": g.get("exit_code") is not None,
                          "passed": bool(g["passed"])})
    perturbed_pass = round(
        sum(1 for p in perturbed if p["passed"]) / max(len(perturbed), 1),
        4) if perturbed else None
    del model
    out = {
        "kind": "cortex-stage16-generalization", "owned_by":
        "Fibonera", "workstream": "8-generalization",
        "unseen_suite_fingerprint": fingerprint_unseen16(),
        "harness_sanity": harness,
        "contamination": contamination,
        "capability": cap_full,
        "per_task": per_task,
        "perturbation": {"n_probes": len(perturbed),
                         "pass_at_1": perturbed_pass,
                         "seed_base": PERTURB_SEED_BASE16,
                         "per_probe": perturbed,
                         "rejected": suite["rejected"]},
        "unseen_pass_at_k": _pass_at_k16(tok),
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- phase 9

def stage_9_moe_divergent(tok, train_recs: list, exposure: dict) -> dict:
    header = _experiment_header16(
        "stage16-9-moe-and-divergent-contribution",
        "the existing MoE and divergent-reasoning subsystems are probed "
        "on the stage-16 primary arm; expert diversity and hypothesis "
        "diversity without execution-verified success are not capability",
        "moe_probe on the real MoE subject; divergent_eval with repair vs "
        "plain fixed-decoding generation on identical v1 tasks; every "
        "success execution-verified; no new experts or reasoning "
        "engines added")
    moe_sub = build_moe_subject()
    batches = [train_recs[i:i + 2]
               for i in range(0, min(len(train_recs), 24), 2)]
    mp = moe_probe(moe_sub.model, batches, tok, make_batch)
    del moe_sub
    plain = []
    model = _load_arm(f"stage16-{STAGE16_PRIMARY_ARM}")
    model.eval()
    torch.manual_seed(GEN_SEED16)
    for t in TASKS[:2]:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), t.instruction)).strip()
        g = _grade_v1(t, cand)
        plain.append({"task_id": t.task_id, "passed": g["passed"],
                      "executed": g["executed"]})
    del model
    dense = dense_model_from_warm6()
    dense.eval()
    from cortex.evaluation.divergent import divergent_eval
    report = divergent_eval(_subject(tok, dense), tasks=TASKS[:2],
                            n_hypotheses=3, repair_round=True)
    if isinstance(report, dict):
        report = dict(report)
        report.pop("per_task", None)
    del dense
    out = {
        "kind": "cortex-stage16-moe-divergent", "owned_by":
        "Fibonera", "workstream": "9-moe-divergent",
        "moe_probe": mp,
        "plain_generation_v1_first2": plain,
        "divergent_eval_v1_first2": report,
        "contribution_verdict": (
            "measured; diversity without execution-verified success is "
            "not capability"),
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- phase 10

def stage_10_capacity() -> dict:
    header = _experiment_header16(
        "stage16-10-capacity-and-virtual-accounting",
        "virtual coverage is recounted, GPU re-qualified and every "
        "stage-16 checkpoint verified to restore into the reference "
        "architecture; no parameters are claimed trained beyond the "
        "authoritative recount",
        "allocated/init/updated scalar classes separated; 2B gate result "
        "recorded with honest hardware blockers; checkpoint recovery "
        "executes real torch.load + state_dict load; the authoritative "
        "store recount must equal 35292")
    from cortex.hardware.profiler import can_train_locally, profile_hardware

    from cortex.model.configs import CORTEX_2B
    profile = profile_hardware()
    feasible, reason = can_train_locally(
        profile, CORTEX_2B.theoretical_param_count())
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan = persisted_optimizer_scan(auth)
    auth.close()
    recovery = []
    for prefix, arm in ([(CKPT_PREFIX16, n) for n in (
            "control", "exposure_const", "exposure_sched",
            "exposure_sched_rep2", "exposure_early", "exposure_weight",
            "complete_curriculum", FB16_ARM_NAME)] + [
            ("cortex-sft", STAGE15_REFERENCE_ARM)]):
        p = CKPT10_DIR / f"{prefix}-{arm}.pt"
        entry = {"checkpoint": str(p), "exists": p.exists()}
        if p.exists():
            try:
                m = dense_model_from_warm6()
                ck = torch.load(p, map_location="cpu", weights_only=True)
                m.load_state_dict(ck["model_state"])
                entry["restores"] = True
                entry["steps"] = ck.get("steps")
                m = None
                ck = None
            except Exception as exc:
                entry["restores"] = False
                entry["error"] = repr(exc)
        recovery.append(entry)
    out = {
        "kind": "cortex-stage16-capacity-audit", "owned_by":
        "Fibonera", "workstream": "10-capacity",
        "hardware": {"has_cuda": profile.has_cuda,
                     "cuda_devices": profile.has_cuda
                     and torch.cuda.device_count() or 0,
                     "ram_total_gb": profile.ram_total_gb,
                     "ram_available_gb": profile.ram_available_gb},
        "two_b_gate": {"feasible": bool(feasible), "reason": reason,
                       "decision": "BLOCKED - unchanged since stage 9"},
        "virtual_store_recount": {
            "slots_updated": scan.get("slots_updated"),
            "unchanged_35292": scan.get("slots_updated") == 35292,
            "n_allocated": scan.get("n_allocated")},
        "stage16_checkpoint_recovery": recovery,
        "cortex_s_reference": {
            "declared": CORTEX_S.theoretical_param_count(),
            "max_seq_len": CORTEX_S.max_seq_len,
            "status": str(CORTEX_S.status)},
        "spend_usd": 0.0, "gpu_hours": 0.0,
    }
    return _finish(header, out, "BLOCKED")


# --------------------------------------------------------------- phase 11

def stage_11_promotion(baseline: dict, exposure: dict, complete: dict,
                       objective: dict, feedback: dict, selfcorr: dict,
                       general: dict, capacity: dict,
                       gate_h: dict | None) -> dict:
    header = _experiment_header16(
        "stage16-11-promotion-gates",
        "the eight pre-registered gates A-H decide promotion; a held-out "
        "improvement alone never promotes; thresholds fixed in the "
        "script docstring before any stage-16 execution",
        "gates evaluated from measured per-task/trajectory rows only; "
        "PROMOTED iff all of A..H PASS; champion retained otherwise")
    arms = exposure["arms"]
    prim = arms[STAGE16_PRIMARY_ARM]
    rep2 = arms["exposure_sched_rep2"]
    ctrl = arms["control"]
    cap = general["capability"]
    prim_pr = prim["free_running_probes"]["parse_rate"]
    ctrl_pr = ctrl["free_running_probes"]["parse_rate"]
    rep2_pr = rep2["free_running_probes"]["parse_rate"]
    gateA = (prim_pr - ctrl_pr >= GATE_B_PARSE_DELTA
             and rep2_pr - ctrl_pr >= GATE_B_PARSE_DELTA)
    gateB = prim_pr > 0 and prim_pr - ctrl_pr >= GATE_B_PARSE_DELTA
    gateC = (cap["v1_complete_program_rate"] > 0
             or cap["v3_complete_program_rate"] > 0)
    gateD = (cap["v1_pass_at_1"] > 0 or cap["v3_pass_at_1"] > 0
             or cap["t15-unseen_pass_at_1"] > 0)
    gateE = (cap["t16-unseen_pass_at_1"] > 0
             and general["perturbation"]["pass_at_1"] > 0)
    gateF = selfcorr["repair_successes"] >= 1
    base_ref = baseline["checkpoints"]["stage15_structural_champion"]
    gateG = bool(
        prim["heldout_native_final"] <= G_HELDOUT_LIMIT
        and prim["heldout_eval_aligned_final"] <= G_HELDOUT_LIMIT
        and prim["heldout_eval_aligned_final"]
        <= base_ref["heldout_eval_aligned"] + 0.05)
    gateH = None if gate_h is None else bool(gate_h.get("all_pass"))
    gates = {
        "gateA_reproducibility": {"ok": bool(gateA), "measured": {
            "primary_parse_rate": prim_pr, "rep2_parse_rate": rep2_pr,
            "control_parse_rate": ctrl_pr,
            "required_delta": GATE_B_PARSE_DELTA}},
        "gateB_free_running_structure": {"ok": bool(gateB), "measured": {
            "primary_parse_rate": prim_pr,
            "control_parse_rate": ctrl_pr,
            "delta": round(prim_pr - ctrl_pr, 4),
            "required_delta": GATE_B_PARSE_DELTA}},
        "gateC_complete_programs": {"ok": bool(gateC), "measured": {
            "v1_complete": cap["v1_complete_program_rate"],
            "v3_complete": cap["v3_complete_program_rate"]}},
        "gateD_functional_correctness": {"ok": bool(gateD), "measured": {
            "v1_pass_at_1": cap["v1_pass_at_1"],
            "v3_pass_at_1": cap["v3_pass_at_1"],
            "t15_pass_at_1": cap["t15-unseen_pass_at_1"]}},
        "gateE_generalization": {"ok": bool(gateE), "measured": {
            "t16_pass_at_1": cap["t16-unseen_pass_at_1"],
            "perturbed_pass_at_1":
            general["perturbation"]["pass_at_1"]}},
        "gateF_self_correction": {"ok": bool(gateF), "measured": {
            "repair_successes": selfcorr["repair_successes"],
            "n_repaired": selfcorr["n_repaired"]}},
        "gateG_regression_control": {"ok": bool(gateG), "measured": {
            "primary_heldout_native": prim["heldout_native_final"],
            "primary_heldout_eval_aligned":
            prim["heldout_eval_aligned_final"],
            "limit": G_HELDOUT_LIMIT,
            "baseline_reference_heldout_eval_aligned":
                base_ref["heldout_eval_aligned"]}},
        "gateH_independent_audit": {
            "ok": bool(gateH),
            "measured": gate_h if gate_h is not None else
            "PENDING - run scripts/stage16_audit.py",
            "status": "PASS" if gateH else
            ("FAIL" if gate_h is not None else "PENDING")},
    }
    a_to_g = all(g["ok"] for k, g in gates.items()
                 if k != "gateH_independent_audit")
    promoted = bool(a_to_g and gateH)
    out = {
        "kind": "cortex-stage16-promotion", "owned_by": "Fibonera",
        "workstream": "11-promotion",
        "primary_arm": STAGE16_PRIMARY_ARM,
        "candidate_checkpoint": str(
            CKPT10_DIR
            / f"{CKPT_PREFIX16}-{STAGE16_PRIMARY_ARM}.pt"),
        "candidate_checkpoint_sha256": _sha256_file(
            CKPT10_DIR
            / f"{CKPT_PREFIX16}-{STAGE16_PRIMARY_ARM}.pt"),
        "gates": gates,
        "gates_passed": sum(1 for g in gates.values() if g["ok"]),
        "promotion_decision": (
            "PROMOTED - all pre-registered gates A-H PASS" if promoted
            else "NOT PROMOTED - stage-12 champion retained with rollback "
                 "recorded; negative result preserved"),
        "rollback": {
            "baseline_checkpoint": str(
                CKPT10_DIR / f"cortex-sft-{CHAMPION_STAGE12}.pt"),
            "stage15_reference_checkpoint": str(
                CKPT10_DIR
                / f"cortex-sft-{STAGE15_REFERENCE_ARM}.pt"),
            "procedure": "stage-16 checkpoints are additive separate "
                         "files; restore stage-12 champion by loading "
                         "cortex-sft-stage12-chunked_control.pt"},
        "capacity_status": capacity["two_b_gate"]["decision"],
        "objective_arm_reference": {
            "weighted_arm_parse_rate": objective["exposure_weight_arm"][
                "free_running_probes"]["parse_rate"]},
        "curriculum_arm_reference": {
            "curriculum_vs_control_parse_rate":
            complete["findings"]["curriculum_vs_control_parse_rate"]},
        "feedback_arm_reference": {
            "target_bearing_steps": feedback["findings"][
                "target_bearing_steps"],
            "regression_vs_stage15_reference":
            feedback["findings"]["regression_vs_stage15_reference"]},
    }
    return _finish(header, out, "PROMOTED" if promoted
                   else "NOT PROMOTED")


# --------------------------------------------------------------- main

def _resume(name: str, fn, *args):
    """Resume support: if the workstream artifact was persisted by a
    prior (possibly interrupted) process, reuse it; otherwise run."""
    p = ART / f"{name}.json"
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            if isinstance(doc, dict) and "experiment" in doc:
                print(f"[stage16] RESUME {name} from persisted artifact",
                      flush=True)
                return doc
        except Exception:
            pass
    return fn(*args)


def main() -> int:
    head = _git_head()
    assert head.startswith(PARENT_COMMIT), (
        f"stage-16 must start from {PARENT_COMMIT}, got {head}")
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    train_recs = load_split(str(DATASET), "train")
    heldout_recs = load_split(str(DATASET), "heldout")
    print(f"[stage16] dataset train={len(train_recs)} "
          f"heldout={len(heldout_recs)} "
          f"t16-unseen={len(UNSEEN_TASKS_STAGE16)}", flush=True)
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_pre = persisted_optimizer_scan(auth)
    auth.close()
    assert scan_pre["slots_updated"] == 35292, "authoritative store moved"

    b = _resume("stage16_baseline", stage_1_baseline,
                tok, train_recs, heldout_recs)
    _save_artifact("stage16_baseline.json", b)
    print(f"[stage16] 1 baseline reproduce="
          f"{b['free_running_failure_reproduction']['parse_rate']}",
          flush=True)

    e = _resume("stage16_autoregressive_exposure", stage_2_exposure,
                tok, train_recs, heldout_recs)
    _save_artifact("stage16_autoregressive_exposure.json", e)
    print(f"[stage16] 2 exposure best={e['findings']['best_exposure_arm']} "
          f"delta={e['findings']['best_parse_rate_delta_over_control']}",
          flush=True)

    s3 = stage_3_structural(tok, heldout_recs, e)
    _save_artifact("stage16_structural_learning.json", s3)
    print(f"[stage16] 3 structural primary_parse="
          f"{s3['comparison']['primary']['free_running_parse_rate']}",
          flush=True)

    s4 = _resume("stage16_complete_program_training",
                 stage_4_complete_program, tok, train_recs, heldout_recs, e)
    _save_artifact("stage16_complete_program_training.json", s4)
    curriculum_parse = s4["curriculum_arm"]["free_running_probes"][
        "parse_rate"]
    print(f"[stage16] 4 complete-program "
          f"curriculum_parse={curriculum_parse}", flush=True)

    s5 = _resume("stage16_objective_ablation", stage_5_objective,
                 tok, e, train_recs, heldout_recs)
    _save_artifact("stage16_objective_ablation.json", s5)
    weighted_parse = s5["exposure_weight_arm"]["free_running_probes"][
        "parse_rate"]
    print(f"[stage16] 5 objective weighted_parse={weighted_parse}",
          flush=True)

    s6 = _resume("stage16_execution_feedback", stage_6_execution_feedback,
                 tok, train_recs, heldout_recs, e)
    _save_artifact("stage16_execution_feedback.json", s6)
    print(f"[stage16] 6 feedback "
          f"target_bearing={s6['findings']['target_bearing_steps']}/"
          f"{FB16_STEPS} "
          f"heldout={s6['feedback_arm']['heldout_eval_aligned_final']}",
          flush=True)

    s7 = _resume("stage16_self_correction", stage_7_self_correction,
                 tok, e, UNSEEN_TASKS_STAGE16)
    _save_artifact("stage16_self_correction.json", s7)
    print(f"[stage16] 7 self-correction "
          f"repair_successes={s7['repair_successes']}", flush=True)

    s8 = _resume("stage16_generalization", stage_8_generalization,
                 tok, train_recs, e)
    _save_artifact("stage16_generalization.json", s8)
    print(f"[stage16] 8 generalization "
          f"t16_pass@1={s8['capability']['t16-unseen_pass_at_1']}",
          flush=True)

    s9 = _resume("stage16_moe_divergent", stage_9_moe_divergent,
                 tok, train_recs, e)
    _save_artifact("stage16_moe_divergent.json", s9)

    s10 = _resume("stage16_capacity_audit", stage_10_capacity)
    _save_artifact("stage16_capacity_audit.json", s10)
    print(f"[stage16] 9/10 subsystems + capacity "
          f"(h={s10['experiment']['decision']})", flush=True)

    promo = stage_11_promotion(b, e, s4, s5, s6, s7, s8, s10, None)
    _save_artifact("stage16_promotion.json", promo)
    print(f"[stage16] 11 promotion (pre-audit): "
          f"{promo['promotion_decision']}", flush=True)

    from scripts.stage16_audit import run_audit
    audit = run_audit()
    _save_artifact("stage16_independent_audit.json", audit)
    print(f"[stage16] 12 audit: {audit['summary']}", flush=True)

    promo = stage_11_promotion(b, e, s4, s5, s6, s7, s8, s10, audit)
    _save_artifact("stage16_promotion.json", promo)
    print(f"[stage16] 11 promotion (final): "
          f"{promo['promotion_decision']} "
          f"({promo['gates_passed']}/8 gates)", flush=True)

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_post = persisted_optimizer_scan(auth)
    auth.close()
    _save_artifact("stage16_report.json", _report_payload(
        b, e, s3, s4, s5, s6, s7, s8, s9, s10, promo, audit, scan_post))
    print("[stage16] complete", flush=True)
    return 0


def _report_payload(b, e, s3, s4, s5, s6, s7, s8, s9, s10, promo, audit,
                    scan_post) -> dict:
    cap = s8["capability"]
    return {
        "kind": "cortex-stage16-report", "owned_by": "Fibonera",
        "stage": 16,
        "summary": {
            "parent_commit": PARENT_COMMIT,
            "baseline": b["experiment"]["decision"],
            "baseline_free_running_failure_reproduction": {
                k: b["free_running_failure_reproduction"][k] for k in (
                    "n_probes", "parse_rate", "eot_stop_count",
                    "failures_at_first_indent_transition",
                    "training_fragment_continuations")},
            "exposure_dataset": {
                k: e["exposure_dataset"][k] for k in (
                    "attempted", "aligned_rows_count", "early_rows_count",
                    "skip_reasons", "divergence_positions",
                    "prefix_lengths_mean")},
            "free_running_parse_rate_by_arm":
                e["findings"]["parse_rate_by_arm"],
            "heldout_eval_aligned_by_arm":
                e["findings"]["heldout_eval_aligned_by_arm"],
            "phase2_acceptance": e["findings"]["phase2_acceptance"],
            "structural_first_transition_error_examples": {
                arm: s3["comparison"][arm]["first_transition_divergence"][
                    "n_with_transition_error"]
                for arm in ("control", "primary")},
            "record_length_stats": {
                k: s4["record_length_stats"][k] for k in (
                    "n_records", "total_tokens_mean", "total_tokens_max",
                    "total_tokens_over_256", "retention")},
            "cortex_s_feasibility": {
                k: s4["cortex_s_feasibility_probe"].get(k) for k in (
                    "live_param_count", "probe_seq_width",
                    "secs_per_step_mean", "peak_rss_delta_mb",
                    "param_count_matches_declared")},
            "objective_grad_cosine": s5["gradient_interference"][
                "cosine_similarity"],
            "weighted_arm_vs_exposure_const_parse_rate": s5["findings"][
                "weighted_arm_vs_exposure_const_parse_rate"],
            "feedback": {
                "success_count": s6["success_count"],
                "candidates": len(s6["candidates"]),
                "target_bearing_steps":
                s6["findings"]["target_bearing_steps"],
                "heldout_vs_stage15_reference":
                s6["findings"]["regression_vs_stage15_reference"]},
            "self_correction": {
                k: s7[k] for k in (
                    "direct_successes", "n_repaired",
                    "repair_successes", "first_attempt_success_rate",
                    "success_after_corrections",
                    "average_correction_attempts")},
            "capability_pass_at_1": {
                k: cap[k] for k in (
                    "v1_pass_at_1", "v3_pass_at_1",
                    "t15-unseen_pass_at_1", "t16-unseen_pass_at_1",
                    "v1_syntax_valid_rate", "v3_syntax_valid_rate",
                    "v1_complete_program_rate",
                    "v3_complete_program_rate")},
            "perturbed_pass_at_1": s8["perturbation"]["pass_at_1"],
            "unseen_pass_at_k": s8["unseen_pass_at_k"],
            "moe_divergent": {
                "moe_available": "error" not in s9["moe_probe"],
                "divergent_report": bool(
                    s9["divergent_eval_v1_first2"])},
            "two_b_gate": s10["two_b_gate"]["decision"],
            "virtual_store_recount": s10["virtual_store_recount"],
            "promotion_decision": promo["experiment"]["decision"],
            "promotion_gates_passed": promo["gates_passed"],
            "promotion_gate_statuses": {
                k: v["status"] if "status" in v
                else ("PASS" if v["ok"] else "FAIL")
                for k, v in promo["gates"].items()},
            "independent_audit_summary": audit["summary"],
            "fingerprint_v1": FINGERPRINT_V1,
            "fingerprint_v2": FINGERPRINT_V2,
            "fingerprint_v3": fingerprint_tasks("v3"),
            "fingerprint_t15_unseen": fingerprint_unseen(),
            "fingerprint_t16_unseen": fingerprint_unseen16(),
            "spend_usd": 0.0, "gpu_hours": 0.0,
            "authoritative_store_untouched":
                scan_post["slots_updated"] == 35292,
        }}


if __name__ == "__main__":
    raise SystemExit(main())
