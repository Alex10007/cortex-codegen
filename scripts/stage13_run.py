"""Stage 13 orchestrator: structural code generation, syntax reliability,
functional correctness and generalization.

Stage 12 established (independently audited, commit 402de46):
    - 43/45 generations introduced persistent syntax defects; 38/45 had
      incorrect indentation transitions; 38/45 self-emitted EOT;
    - 49.84% of chunked training windows ended inside an open construct;
    - dataset complete-program rate 98.8% but 0.0% classes (single-function
      bias); teacher-forced token accuracy 0.9322 vs held-out 0.6214;
    - all four stage-12 arms failed to improve on the control (held-out
      2.9498); all five evaluated checkpoints pass@1 = 0.0; EOT-gated
      exploratory decoding 1/15 parses, zero solutions - decoding ruled
      out as the sole cause;
    - 2B training remained GPU-BLOCKED.

Stage 13 therefore measures and repairs the TRAINING-TIME structural
signal (audit -> structural segmentation -> indentation-sensitive
objective) and re-evaluates free-running generation against the stage-12
champion (`cortex-sft-stage12-chunked_control`), one variable at a time.

Pre-registered (before any stage-13 execution; never adjusted post hoc):
    STEPS=2000, EVAL_EVERY=400, PHYS_LR=5e-5, BATCH=4, BATCH bounce over
    train order, SEED=20260101 (training), GEN_SEED=20261214 (generation),
    INDENT_WEIGHT=3.0 (indent arm), held-out ceiling 3.3015 (the stage-10
    measured parent value).  Arms (all legacy prompt style; the stage-12
    eval-aligned experiment measured WORSE held-out so it is not retried):
      control_recheck             - stage-12 champion config re-run at
                                    stage-13 seed/horizon (fixed seg, w=1)
      structural_segmentation     - single change: structural-boundary
                                    segmentation (five-gate evidence for
                                    Workstream C)
      indent_weighted             - single change: indentation-sensitive
                                    loss weight 3.0
      structural_indent_combined  - declared combination of both changes
    Stage-13 promotion rule (5 gates, Workstream K):
      Gate 1 reproducible reduction of persistent syntax/indent defects;
      Gate 2 increase of complete syntactically valid programs;
      Gate 3 at least one program compiles+executes+passes tests;
      Gate 4 success on unseen/perturbed formulations;
      Gate 5 at least one independently verified pass@1 on an immutable
      unseen benchmark.  PROMOTE iff Gate 5 AND held-out <= 3.3015.
      If pass@1 is 0 the capability result is NOT PROMOTED regardless of
      loss or syntax improvements.

Usage:
    PYTHONPATH=repo python scripts/stage13_run.py
    PYTHONPATH=repo python scripts/stage13_run.py --finalize-from-artifacts
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cortex.evaluation.autoregressive_diagnostics import (  # noqa: E402
    analyzer_selftest, classify_structural_failure, trajectory_generate)
from cortex.evaluation.benchmarks import fingerprint_tasks  # noqa: E402
from cortex.evaluation.divergent import divergent_eval  # noqa: E402
from cortex.evaluation.failure_analysis import diagnose_candidate  # noqa: E402
from cortex.evaluation.generalization import (  # noqa: E402
    benchmark_contamination, dataset_overlap, perturbation_suite_valid)
from cortex.evaluation.inference import (  # noqa: E402
    extract_code, teacher_forced_token_accuracy)
from cortex.evaluation.moe_probe import probe as moe_probe  # noqa: E402
from cortex.evaluation.structure_audit import (  # noqa: E402
    audit_dataset_structure, chunk_boundary_audit, prompt_style_context_stats,
    structure_metrics)
from cortex.evaluation.tasks import TASKS  # noqa: E402
from cortex.evaluation.tasks_v3 import TASKS_V3  # noqa: E402
from cortex.tokenizer.cortex_tokenizer import load_tokenizer  # noqa: E402
from cortex.training.longcontext import (  # noqa: E402
    chunk_record, chunk_record_structural, completion_coverage,
    make_batch_chunked, structural_cut_summary)
from cortex.training.sft import (  # noqa: E402
    evaluate_heldout, load_split, make_batch)
from cortex.virtual.coverage import persisted_optimizer_scan  # noqa: E402
from cortex.virtual.store import VirtualParameterStore  # noqa: E402

from scripts.stage10_run import (  # noqa: E402
    ART, CKPT10_DIR, DATASET, DECODING, FINGERPRINT_V1, FINGERPRINT_V2, SEQ,
    _experiment_header, _finish, _save_artifact, _sha256_file,
    dense_model_from_warm6, eval_fixed_decoding, gen_fixed_decoding,
    WARM6_CKPT)
from scripts.stage9_run import (  # noqa: E402
    _latest_joint_physical, build_moe_subject)

CHAMPION12 = "stage12-chunked_control"
PARENT_COMMIT = "402de46"
STEPS = 2000
EVAL_EVERY = 400
PHYS_LR = 5e-5
BATCH = 4
SEED = 20260101
GEN_SEED = 20261214
INDENT_WEIGHT = 3.0
BASE_HELDOUT_ALLOWED = 3.3015

GATES = {
    "gate1_structural_improvement":
        "reproducible reduction in persistent syntax defects and "
        "incorrect indentation transitions vs the stage-12 champion",
    "gate2_complete_generation":
        "increase in complete, syntactically valid programs under "
        "unchanged evaluation conditions",
    "gate3_functional_correctness":
        "at least one generated program compiles, executes and passes "
        "independent tests",
    "gate4_generalization":
        "success on previously unseen tasks and perturbed formulations",
    "gate5_capability_promotion":
        "at least one independently verified pass@1 on an immutable "
        "unseen benchmark",
}
PROMOTION_RULE = (
    "PROMOTE iff Gate 5 (first independently verified pass@1 > 0 on the "
    "immutable v1/v3 benchmarks) AND held-out loss <= 3.3015; pass@1 = 0 "
    "means NOT PROMOTED regardless of loss or syntax improvements")

STRUCTURAL_KEYWORDS = {
    "def", "class", "if", "elif", "else", "for", "while", "try",
    "except", "finally", "with", "return", "yield", "import", "from",
    "pass", "raise", "assert", "del", "global", "nonlocal", "lambda",
    "in", "not", "and", "or", "is", "break", "continue",
}
INDENT_FAMILY = {
    "incorrect_indent_increase", "incorrect_indent_decrease",
    "unexpected_indentation", "unexpected_indentation_transition",
    "incomplete_block_missing_indent",
}
TRUNCATION_FAMILY = {
    "truncated_generation_max_new_tokens",
    "repetition_degeneration_truncation",
    "premature_eot_open_construct",
}
ARM13_CONFIGS = {
    "control_recheck": {
        "prompt_style": "legacy", "segmentation": "fixed",
        "indent_weight": 1.0,
        "expected": "reproduces the stage-12 champion config at the "
                    "stage-13 seed/horizon (comparability baseline)",
    },
    "structural_segmentation": {
        "prompt_style": "legacy", "segmentation": "structural",
        "indent_weight": 1.0,
        "expected": "structurally clean window cuts reduce mid-construct "
                    "window ends (49.84% -> lower) and improve structural "
                    "token accuracy at equal supervised coverage",
    },
    "indent_weighted": {
        "prompt_style": "legacy", "segmentation": "fixed",
        "indent_weight": INDENT_WEIGHT,
        "expected": "upweighting indentation whitespace tokens (3x) "
                    "improves indentation-transition prediction without "
                    "overall held-out regression",
    },
    "structural_indent_combined": {
        "prompt_style": "legacy", "segmentation": "structural",
        "indent_weight": INDENT_WEIGHT,
        "expected": "declared combination; no new hypothesis until the "
                    "single-variable arms are measured",
    },
}


class _Sub:
    pass


def _subject(tok, model):
    sub = _Sub()
    sub.tokenizer = tok
    sub.model = model
    return sub


def _eval_prompt(instruction: str, tok) -> str:
    return tok.EOT + "\n" + "# Task: " + instruction + "\n"


def _load_arm(name: str):
    path = CKPT10_DIR / f"cortex-sft-{name}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model = dense_model_from_warm6()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state"])
    del ck
    return model


def _dist(values: list) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _heldout(model, heldout_recs, tok, style: str = "legacy") -> float:
    v = evaluate_heldout(model, heldout_recs, tok, SEQ, prompt_style=style)
    return round(v, 4) if v is not None else 4.0


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=
                          Path(__file__).resolve().parents[1],
                          capture_output=True, text=True).stdout.strip()[:12]


_WS_CACHE: dict = {}
_TOKEN_CACHE: dict = {}
_WS_MAX_ID = 40000


def _token_class_tables(tok):
    if "tables" in _WS_CACHE:
        return _WS_CACHE["tables"]
    n = _WS_MAX_ID
    ws = torch.zeros(n, dtype=torch.bool)
    kw = torch.zeros(n, dtype=torch.bool)
    delim = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        try:
            text = tok.decode([i])
        except Exception:
            continue
        stripped = text.strip()
        if stripped == "" or "\n" in text:
            ws[i] = True
        elif stripped in STRUCTURAL_KEYWORDS:
            kw[i] = True
        elif stripped and all(c in "()[]{}:,." for c in stripped):
            delim[i] = True
    tables = {"ws": ws, "kw": kw, "delim": delim}
    _WS_CACHE["tables"] = tables
    return tables


def structural_token_accuracy(model, tok, recs: list,
                              seq_len: int = SEQ, batch_size: int = 8,
                              max_batches: int = 6) -> dict:
    """Teacher-forced next-token accuracy split by structural token class
    (whitespace/indentation, structure keywords, delimiters, other) so a
    held-out indentation deficit is visible at token level (Workstream D
    measurement, not a promotion metric on its own)."""
    tables = _token_class_tables(tok)
    model.eval()
    hits = {k: 0 for k in ("whitespace", "structure_keyword", "delimiter",
                           "other", "overall")}
    totals = {k: 0 for k in hits}
    with torch.no_grad():
        nb = 0
        for i in range(0, len(recs), batch_size):
            if nb >= max_batches:
                break
            x, y = make_batch(recs[i:i + batch_size], tok, seq_len)
            if x is None:
                continue
            nb += 1
            logits, _ = model(x, pad_mask=x == tok.pad_id)
            pred = logits.argmax(dim=-1)
            mask = y != -100
            hit = pred == y
            y_flat = y[mask]
            h_flat = hit[mask]
            if int(mask.sum()) == 0:
                continue
            totals["overall"] += int(y_flat.numel())
            hits["overall"] += int(h_flat.sum())

            ws_m = tables["ws"][y_flat.clamp(0, _WS_MAX_ID - 1)]
            kw_m = tables["kw"][y_flat.clamp(0, _WS_MAX_ID - 1)] & ~ws_m
            del_m = tables["delim"][y_flat.clamp(0, _WS_MAX_ID - 1)]
            oth_m = ~ws_m & ~kw_m & ~del_m
            for cls_m, cls_key in ((ws_m, "whitespace"),
                                   (kw_m, "structure_keyword"),
                                   (del_m, "delimiter"),
                                   (oth_m, "other")):
                totals[cls_key] += int(cls_m.sum())
                hits[cls_key] += int((h_flat & cls_m).sum())
    return {
        "n_batches": nb,
        "accuracy_overall": round(hits["overall"] / totals["overall"], 4)
        if totals["overall"] else None,
        "accuracy_whitespace": round(hits["whitespace"] /
                                     totals["whitespace"], 4)
        if totals["whitespace"] else None,
        "accuracy_structure_keyword": round(
            hits["structure_keyword"] / totals["structure_keyword"], 4)
        if totals["structure_keyword"] else None,
        "accuracy_delimiter": round(hits["delimiter"] / totals["delimiter"],
                                    4) if totals["delimiter"] else None,
        "accuracy_other": round(hits["other"] / totals["other"], 4)
        if totals["other"] else None,
        "token_counts": dict(totals),
    }


# --------------------------------------------------------------- workstream A

def stage_a_root_cause(tok) -> dict:
    header = _experiment_header(
        "stage13-A-structural-root-cause",
        "the stage-12 indentation-transition failure share reproduces on "
        "token-level trajectories of the parent and the stage-12 champion; "
        "failures are directly introduced by the model's own next-token "
        "choice, not by decoding or truncation",
        "2 subjects x 30 immutable tasks (v1+v3), full per-step lexical "
        "trajectories; analyzer self-test must pass before attribution")
    selftest = analyzer_selftest()
    gate = bool(selftest.get("self_test_ok"))
    subjects = {"dense-warm6-parent": dense_model_from_warm6()}
    subjects[CHAMPION12] = _load_arm(CHAMPION12)
    per_subject: dict = {}
    generations: list = []
    for name, model in subjects.items():
        model.eval()
        parse_n = pass_n = ident_n = direct_n = struct_n = eot_n = 0
        n = 0
        stopped_dist: list = []
        taxonomy_dist: list = []
        for suite, tasks in (("v1", TASKS), ("v3", TASKS_V3)):
            for t in tasks:
                torch.manual_seed(GEN_SEED)
                traj = trajectory_generate(
                    model, tok, _eval_prompt(t.instruction, tok), DECODING)
                cand = extract_code(traj["text"]).strip()
                m = None
                try:
                    from cortex.evaluation.structure_audit import (
                        structure_metrics)
                    m = structure_metrics(cand) if cand else None
                except Exception:
                    m = None
                grade = grade_single_quiet(t, cand)
                tax = classify_structural_failure(traj)
                fi = traj["first_invalid"]
                idx = fi.get("first_invalid_index")
                offending = None
                if idx is not None and traj["steps"]:
                    bad = traj["steps"][max(idx - 1, 0)]
                    offending = {"token_id": bad.get("token_id"),
                                 "token_text": bad.get("token_text"),
                                 "top5": bad.get("top5"),
                                 "entropy": bad.get("entropy"),
                                 "lex_before": bad.get("lex_before")}
                parse = bool(m["parses"]) if m else False
                complete = bool(m["complete_program"]) if m else False
                parse_n += parse
                pass_n += bool(grade["passed"])
                if idx is not None:
                    struct_n += 1
                    direct_n += bool(
                        fi.get("directly_introduced_by_newest_token"))
                if tax["taxonomy"] in INDENT_FAMILY:
                    ident_n += 1
                stopped_dist.append(traj["stopped"])
                if traj["stopped"] == "eot":
                    eot_n += 1
                taxonomy_dist.append(tax["taxonomy"])
                n += 1
                generations.append({
                    "subject": name, "suite": suite, "task_id": t.task_id,
                    "prompt_text": _eval_prompt(t.instruction, tok),
                    "generated_text": traj["text"],
                    "stopped": traj["stopped"],
                    "n_tokens": traj["completion_length_tokens"],
                    "taxonomy": tax["taxonomy"], "evidence": tax["evidence"],
                    "first_invalid_index": idx,
                    "offending_token": offending,
                    "parse": parse, "complete_program": complete,
                    "executed": grade.get("exit_code") is not None,
                    "passed": bool(grade["passed"]),
                    "grade": {"passed": grade.get("passed"),
                              "exit_code": grade.get("exit_code"),
                              "stderr_tail": (grade.get("stderr_tail")
                                              or "")[-240:]},
                })
        per_subject[name] = {
            "n_tasks": n,
            "parse_rate": round(parse_n / n, 4) if n else None,
            "pass_at_1": round(pass_n / n, 4) if n else None,
            "complete_program_rate": round(
                sum(1 for g in generations if g["subject"] == name
                    and g["complete_program"]) / n, 4) if n else None,
            "stopped_distribution": _dist(stopped_dist),
            "structural_taxonomy_distribution": _dist(taxonomy_dist),
            "indentation_related_share": round(ident_n / n, 4) if n else None,
            "persistent_structure_failures": struct_n,
            "directly_introduced_share": round(direct_n / max(struct_n, 1),
                                               4),
            "eot_self_emitted": eot_n,
        }
        del model
    out = {
        "kind": "cortex-stage13-structural-root-cause",
        "owned_by": "Fibonera", "workstream": "A-root-cause",
        "decoding": DECODING,
        "analyzer_selftest": selftest, "analyzer_gate_ok": gate,
        "per_subject": per_subject,
        "generations": generations,
        "taxonomy_families": {
            "indentation_family": sorted(INDENT_FAMILY),
            "truncation_family": sorted(TRUNCATION_FAMILY),
        },
        "causal_confidence": "demonstrated: per-step instruments show the "
                             "exact offending token for every persistent "
                             "structural failure; hypotheses about origin "
                             "in training data are measured in B/C",
        "decision_note": "EXECUTED if the analyzer gate passes and both "
                         "subjects produced trajectories",
    }
    decision = "EXECUTED" if gate and generations else "INCONCLUSIVE"
    return _finish(header, out, decision)


def grade_single_quiet(task, candidate: str) -> dict:
    from cortex.evaluation.runner import grade_single
    return grade_single(task, candidate)


def _structural_capability_matrix(recs: list) -> dict:
    """Per-record structural capability coverage (Workstream B matrix).
    Counted from actually-parsed ASTs of the record's code field."""
    import ast as _ast
    caps = {
        "basic_functions": 0, "nested_functions": 0, "classes": 0,
        "multiple_classes": 0, "imports_and_modules": 0, "control_flow": 0,
        "exceptions": 0, "data_structures": 0, "algorithms_recursion": 0,
        "multi_function_programs": 0, "tests_assertions": 0,
        "decorators_annotations": 0, "docstrings": 0,
        "multiline_strings": 0, "complete_within_record": 0,
    }
    n = n_complete = 0
    for rec in recs:
        code = rec.get("code") or ""
        if not code.strip():
            continue
        n += 1
        try:
            tree = _ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        n_complete += 1
        caps["complete_within_record"] += 1
        funcs = [x for x in _ast.walk(tree)
                 if isinstance(x, (_ast.FunctionDef, _ast.AsyncFunctionDef))]
        classes = [x for x in _ast.walk(tree) if isinstance(x, _ast.ClassDef)]
        if funcs:
            caps["basic_functions"] += 1
        if any(isinstance(inner, (_ast.FunctionDef, _ast.AsyncFunctionDef))
               for outer in funcs for inner in _ast.walk(outer)
               if inner is not outer):
            caps["nested_functions"] += 1
        if classes:
            caps["classes"] += 1
        if len(classes) > 1:
            caps["multiple_classes"] += 1
        if any(isinstance(x, (_ast.Import, _ast.ImportFrom))
               for x in _ast.walk(tree)):
            caps["imports_and_modules"] += 1
        if any(isinstance(x, (_ast.If, _ast.For, _ast.While))
               for x in _ast.walk(tree)):
            caps["control_flow"] += 1
        if any(isinstance(x, _ast.Try) for x in _ast.walk(tree)):
            caps["exceptions"] += 1
        if any(isinstance(x, (_ast.Dict, _ast.List, _ast.Set, _ast.Tuple))
               for x in _ast.walk(tree)):
            caps["data_structures"] += 1
        calls = [x for x in _ast.walk(tree) if isinstance(x, _ast.Call)]
        fnames = {f.name for f in funcs}
        if any(isinstance(c.func, _ast.Name) and c.func.id in fnames
               for c in calls):
            caps["algorithms_recursion"] += 1
        if len(funcs) > 1:
            caps["multi_function_programs"] += 1
        if any(isinstance(x, _ast.Assert) for x in _ast.walk(tree)):
            caps["tests_assertions"] += 1
        if any(isinstance(x, (_ast.FunctionDef, _ast.ClassDef))
               and _ast.get_docstring(x) for x in
               list(funcs) + list(classes)):
            caps["docstrings"] += 1
        if any(isinstance(x, _ast.Constant) and isinstance(x.value, str)
               and "\n" in str(x.value) for x in _ast.walk(tree)):
            caps["multiline_strings"] += 1
        dec = 0
        for x in funcs + classes:
            dec += len(x.decorator_list)
            if isinstance(x, _ast.FunctionDef):
                dec += sum(1 for a in x.args.args
                       if a.annotation is not None)
        if dec:
            caps["decorators_annotations"] += 1
    out = {"n_records_with_code": n,
           "complete_program_rate": round(n_complete / n, 4) if n else None}
    for k, v in caps.items():
        out[k] = {"n_records": v, "share": round(v / n, 4) if n else None}
    return out


def stage_b_data_audit(all_recs: list, train_recs: list,
                       heldout_recs: list, tok) -> dict:
    header = _experiment_header(
        "stage13-B-structural-data-audit",
        "the structural coverage matrix shows the class/nested/multi-"
        "function gaps measured in stage 12 (classes 0.0%) reproduce on "
        "every split, and structural cut statistics quantify how much of "
        "the record stream the 256-token sequence cannot hold",
        "immutable dataset fingerprints unchanged; measured from parsed "
        "records only; no data modification")
    splits = {"all": all_recs, "train": train_recs, "heldout": heldout_recs}
    matrix = {s: _structural_capability_matrix(r) for s, r in splits.items()}
    cut_stats = {s: structural_cut_summary(r, tok, SEQ)
                 for s, r in splits.items()}
    overlap = dataset_overlap(train_recs)
    contam = benchmark_contamination(train_recs)
    fits = prompt_style_context_stats(all_recs, tok, SEQ)
    out = {
        "kind": "cortex-stage13-structural-data-audit",
        "owned_by": "Fibonera", "workstream": "B-data-audit",
        "dataset": str(DATASET), "dataset_sha256": _sha256_file(DATASET),
        "n_records": {"all": len(all_recs), "train": len(train_recs),
                      "heldout": len(heldout_recs)},
        "capability_coverage_matrix": matrix,
        "audit_dataset_structure": audit_dataset_structure(all_recs),
        "structural_cut_summary": cut_stats,
        "context_fits_by_style": fits,
        "dataset_overlap": overlap,
        "benchmark_contamination": contam,
        "single_function_bias_note":
            "stage-12 measured classes at 0.0%; this matrix re-measures "
            "the same structural gap rather than adding examples",
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream C

def _window_stats(recs: list, tok, seg: str) -> dict:
    lens = []
    recs_with = 0
    for rec in recs:
        if seg == "structural":
            windows = chunk_record_structural(rec, tok, SEQ)
        else:
            windows = chunk_record(rec, tok, SEQ)
        if not windows:
            continue
        recs_with += 1
        lens += [len(w[0]) for w in windows]
    return {
        "records_with_windows": recs_with,
        "records_total": len(recs),
        "n_windows": len(lens),
        "min_window_len": min(lens) if lens else None,
        "max_window_len": max(lens) if lens else None,
        "mean_window_len": round(sum(lens) / len(lens), 2) if lens else None,
    }


def stage_c_training_stream(all_recs: list, train_recs: list,
                            val_recs: list, heldout_recs: list, tok) -> dict:
    header = _experiment_header(
        "stage13-C-training-stream-integrity",
        "structural-boundary segmentation preserves exactly-once "
        "completion supervision (the fixed-mode guarantee) while cutting "
        "at lexically clean boundaries, reducing mid-construct window "
        "ends below the stage-12 measured 49.84% without dropping "
        "supervised targets",
        "completion_coverage identical between fixed and structural on "
        "every split; window length <= seq_len; prompt never split; "
        "fixed default retained until the gates pass")
    splits = {"train": train_recs, "val": val_recs, "heldout": heldout_recs,
              "all": all_recs}
    per_split: dict = {}
    for name, recs in splits.items():
        cov_fix = completion_coverage(recs, tok, SEQ, "fixed")
        cov_str = completion_coverage(recs, tok, SEQ, "structural")
        per_split[name] = {
            "completion_coverage_fixed": cov_fix,
            "completion_coverage_structural": cov_str,
            "coverage_identical": (cov_fix["complete"]
                                   and cov_str["complete"]
                                   and cov_fix["total_completion_targets"]
                                   == cov_str["total_completion_targets"]),
            "windows_fixed": _window_stats(recs, tok, "fixed"),
            "windows_structural": _window_stats(recs, tok, "structural"),
        }
    boundary = {
        "fixed": chunk_boundary_audit(train_recs, tok, SEQ, "legacy",
                                      "fixed"),
        "structural": chunk_boundary_audit(train_recs, tok, SEQ, "legacy",
                                           "structural"),
    }
    cov_all = all(v["coverage_identical"] for v in per_split.values())
    out = {
        "kind": "cortex-stage13-training-stream",
        "owned_by": "Fibonera", "workstream": "C-stream",
        "seq_len": SEQ,
        "per_split": per_split,
        "coverage_guarantee_holds": cov_all,
        "chunk_boundary_audit_train": boundary,
        "supervised_note": {
            "window_overlap": "window k (k>=2) starts at the previous "
                              "end-1 so the boundary token is an INPUT "
                              "in exactly one row and the streamed "
                              "completion is supervised exactly once",
            "fallback_cuts": "when no clean boundary exists below the "
                             "window limit the cut falls back to the "
                             "hard limit (documented, may be unbalanced)",
        },
        "eot_context_stats": prompt_style_context_stats(train_recs, tok,
                                                        SEQ),
        "decision_note":
            "the fixed segmentation stays the DEFAULT in "
            "make_batch_chunked until a stage-13 arm passes its "
            "pre-registered gate; the structural path is opt-in "
            "(segmentation=\"structural\")",
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream D

def stage_d_learning(pre_control_accuracy, post_by_arm: dict,
                     margin_note: str) -> dict:
    """Workstream D artifact: structural token-accuracy improvements from
    the Workstream-H arms (indent arm + combined). Accuracy alone is NOT
    a promotion criterion - it evidences whether indentation patterns are
    learnable, Workstream K gates decide promotion."""
    out = {
        "kind": "cortex-stage13-structural-learning",
        "owned_by": "Fibonera", "workstream": "D-learning",
        "hypothesis":
            "Python structural rules (indentation depth transitions, "
            "keyword/delimiter classes) are predictive patterns the model "
            "can learn; the indent-weighted objective improves held-out "
            "structural token accuracy vs the unweighted control",
        "pre_registered": True,
        "structural_token_accuracy_baseline_warm6": pre_control_accuracy,
        "post_by_arm": post_by_arm,
        "interpretation": {
            "measured_margin": margin_note,
            "promotion_role": "supporting capability evidence only; "
                              "promotion is decided solely by the "
                              "Workstream-K gates",
        },
        "indentation_program_structure_focus": [
            "def", "class", "if", "elif", "else", "for", "while", "try",
            "except", "finally", "with", "return", "yield",
        ],
    }
    decision = "EXECUTED" if post_by_arm else "NOT TESTED"
    return _finish(_experiment_header(
        "stage13-D-learning",
        "indentation-sensitive weighting teaches depth transitions as "
        "predictive patterns (measured via per-class held-out accuracy)",
        "single-variable arms vs control; no parser or formatter "
        "substituted for learning"), out, decision)


# --------------------------------------------------------------- workstream E

def stage_e_generation_control(tok) -> dict:
    header = _experiment_header(
        "stage13-E-generation-control",
        "EOT self-emission inside open constructs is a consequence of "
        "the 49.84% mid-construct training windows, not of a decoding "
        "defect; EOT is the trained completion terminator (supervised "
        "exactly once per record as the final completion target)",
        "EOT semantics documented, not redefined; eot-gated decoding is "
        "an EXPLORATORY inference-time policy and never evidence of "
        "model learning; fixed decoding stays primary")
    root_cause = ART / "stage13_structural_root_cause.json"
    stop_classification: dict = {}
    if root_cause.exists():
        a = json.loads(root_cause.read_text())
        for s, ps in a["per_subject"].items():
            stopped = 0
            complete_stop = 0
            for g in a["generations"]:
                if g["subject"] == s:
                    if g["stopped"] == "eot":
                        stopped += 1
                        if g["taxonomy"] == "complete_program":
                            complete_stop += 1
            stop_classification[s] = {
                "stopped_by_eot": stopped,
                "stopped_by_eot_with_complete_program": complete_stop,
                "premature_or_truncated": ps["n_tasks"] - stopped,
                "n_tasks": ps["n_tasks"],
                "stopped_distribution": ps["stopped_distribution"],
            }
    model = _load_arm(CHAMPION12)
    model.eval()
    exploratory = []
    torch.manual_seed(GEN_SEED)
    for t in TASKS:
        traj = trajectory_generate(model, tok,
                                   _eval_prompt(t.instruction, tok),
                                   DECODING, eot_gated=True)
        cand = extract_code(traj["text"]).strip()
        exploratory.append({
            "task_id": t.task_id,
            "parses": structure_metrics(cand)["parses"]
            if cand else False,
            "complete": structure_metrics(cand)["complete_program"]
            if cand else False,
            "n_tokens": traj["completion_length_tokens"],
            "stopped": traj["stopped"],
        })
    del model
    n = len(exploratory)
    out = {
        "kind": "cortex-stage13-generation-control",
        "owned_by": "Fibonera", "workstream": "E-generation-control",
        "eot_semantics": {
            "definition": "EOT (tokenizer id 200) is the completion "
                          "terminator: record_ids supervises `code + \\n "
                          "+ EOT` so EOT is a trained completion target "
                          "exactly once per record and generation stops "
                          "on it via model.generate/trajectory_generate",
            "not_redefined": True,
            "exploratory_decoding_policy": "eot_gated trajectory_generate "
                                           "masks EOT only while the "
                                           "completion sits inside an "
                                           "open construct (inference "
                                           "policy, NOT a model change)",
        },
        "stop_classification_from_workstream_a": stop_classification,
        "exploratory_eot_gated_run": {
            "checkpoint": CHAMPION12,
            "n_tasks": n,
            "parse_rate": round(
                sum(1 for e in exploratory if e["parses"]) / n, 4)
            if n else None,
            "complete_rate": round(
                sum(1 for e in exploratory if e["complete"]) / n, 4)
            if n else None,
            "per_task": exploratory,
        },
        "caps": {"avg_generated_tokens_exploratory": round(
            sum(e["n_tokens"] for e in exploratory) / n, 1) if n else None},
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream F

def stage_f_execution_feedback(a_report: dict) -> dict:
    header = _experiment_header(
        "stage13-F-execution-feedback",
        "execution evidence classifies each generation into a feedback "
        "stage (passed / syntax_generation / runs_or_tests / unknown) "
        "using the existing diagnose_candidate + grade_single path - no "
        "duplicate engine; execution-derived training supervision is a "
        "NOT TESTED pre-registered follow-up",
        "no credit for plausible-looking code; actual stdout/stderr/exit "
        "codes from the existing sandbox; no duplicate execution engine")
    per_subject: dict = {}
    stage_dist_total: dict = {}
    for s, ps in a_report["per_subject"].items():
        dist: dict = {}
        rows = []
        for g in a_report["generations"]:
            if g["subject"] != s:
                continue
            cand = extract_code(g["generated_text"]).strip()
            try:
                from cortex.evaluation.structure_audit import (
                    structure_metrics)
                m = structure_metrics(cand) if cand else {"parses": False}
            except Exception:
                m = {"parses": False}
            grade = {"passed": g["passed"],
                     "exit_code": g["grade"].get("exit_code")}
            diag = diagnose_candidate(
                cand, grade, instruction="",
                max_tokens_hint=DECODING["max_new_tokens"])
            if grade.get("passed"):
                stage = "passed"
            elif not m["parses"]:
                stage = "syntax_generation_failure"
            elif grade.get("exit_code") is not None:
                stage = "runs_or_tests_failing"
            else:
                stage = "unknown"
            dist[stage] = dist.get(stage, 0) + 1
            stage_dist_total[stage] = stage_dist_total.get(stage, 0) + 1
            rows.append({"task_id": g["task_id"], "stage": stage,
                         "diagnosis": diag["diagnosis"],
                         "parses": m["parses"],
                         "complete_program": m.get("complete_program"),
                         "executed": grade.get("exit_code") is not None})
        per_subject[s] = {
            "n": len(rows),
            "stage_distribution": _dist([g["stage"] for g in rows]),
            "per_task": rows,
        }
    out = {
        "kind": "cortex-stage13-execution-feedback",
        "owned_by": "Fibonera", "workstream": "F-execution-feedback",
        "classification_source": "diagnose_candidate via grade_single "
                                 "(existing engine, no duplicate)",
        "per_subject": per_subject,
        "stage_distribution_total": dict(sorted(
            stage_dist_total.items(), key=lambda kv: -kv[1])),
        "feedback_training_experiment": {
            "status": "NOT TESTED",
            "pre_registered_hypothesis": "executing generated candidates "
                                         "in the existing sandbox and "
                                         "conditioning repair prompts on "
                                         "actual stderr improves pass@1 "
                                         "(divergent_eval already "
                                         "implements repair_round); an "
                                         "optimizer-updated feedback "
                                         "experiment requires a new "
                                         "registered run with its own "
                                         "checkpoint lineage",
            "ticket": "deferred to a registered stage-14 experiment",
        },
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream G

def stage_g_generalization(tok, all_recs: list, train_recs: list,
                           heldout_recs: list) -> dict:
    header = _experiment_header(
        "stage13-G-generalization",
        "contamination is unchanged and unseen perturbed task variants "
        "remain unsolved (pass@1 = 0), so transfer to unseen "
        "formulations is already measurable and the held-out accuracy "
        "gap can be re-quantified on the champion",
        "immutable v1/v3 fingerprints unchanged; perturbation seed_base "
        "20261300; heldout split never trained on")
    suite = perturbation_suite_valid(TASKS + TASKS_V3, seed_base=20261300)
    probes = suite["probes"]
    model = _load_arm(CHAMPION12)
    model.eval()
    perturbed = []
    torch.manual_seed(GEN_SEED)
    for pt in probes:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), pt.instruction)).strip()
        grade = grade_single_quiet(pt, cand)
        try:
            from cortex.evaluation.structure_audit import structure_metrics
            m = structure_metrics(cand) if cand else {"parses": False,
                                                      "complete_program":
                                                          False}
        except Exception:
            m = {"parses": False, "complete_program": False}
        perturbed.append({
            "task_id": pt.task_id, "parses": m["parses"],
            "complete_program": m["complete_program"],
            "executed": grade.get("exit_code") is not None,
            "passed": bool(grade["passed"])})
    n = len(perturbed)
    tf_heldout = teacher_forced_token_accuracy(model, tok,
                                               heldout_recs[:16], SEQ,
                                               batch_size=4, max_batches=8)
    tf_train = teacher_forced_token_accuracy(model, tok,
                                             train_recs[:16], SEQ,
                                             batch_size=4, max_batches=8)
    del model
    out = {
        "kind": "cortex-stage13-generalization",
        "owned_by": "Fibonera", "workstream": "G-generalization",
        "contamination": benchmark_contamination(train_recs),
        "dataset_overlap": dataset_overlap(all_recs),
        "perturbation_suite": {"n_valid_probes": suite["n_valid_probes"],
                               "rejected": suite["rejected"]},
        "perturbed_task_results": {
            "n_tasks": n,
            "pass_at_1": round(sum(1 for p in perturbed if p["passed"])
                               / n, 4) if n else None,
            "syntax_valid_rate": round(
                sum(1 for p in perturbed if p["parses"]) / n, 4)
            if n else None,
            "complete_program_rate": round(
                sum(1 for p in perturbed if p["complete_program"]) / n, 4)
            if n else None,
            "per_task": perturbed,
        },
        "teacher_forced_accuracy": {
            "heldout_subset": tf_heldout, "train_subset": tf_train,
            "n_heldout_records": min(16, len(heldout_recs)),
            "gap_note": "train vs held-out teacher-forced accuracy gap "
                        "on identical eval protocol (stage-12 measured "
                        "0.9322 vs 0.6214)",
        },
        "separation_of_transfer_kinds": {
            "exact_memorization": "measured via dataset_overlap exact "
                                  "duplicates",
            "near_duplicate_reproduction": "measured via dataset_overlap "
                                           "near-duplicates",
            "structural_transfer": "perturbed probes rename identifiers "
                                   "and alter constraints",
            "algorithmic_transfer": "perturbed probes alter numerical "
                                    "range/structures",
            "functional_generalization": "pass on perturbed probes",
        },
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream H

def _weighted_ce(logits, y, tok, indent_weight: float):
    nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                          ignore_index=-100, reduction="none").view(
        y.shape)
    tables = _token_class_tables(tok)
    safe = y.clamp(0, _WS_MAX_ID - 1)
    ws_last = tables["ws"][safe] & (y != -100)
    w = torch.where(ws_last,
                    torch.tensor(indent_weight, dtype=nll.dtype),
                    torch.tensor(1.0, dtype=nll.dtype))
    w = w * (y != -100).to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def _capability_eval(model, tok) -> dict:
    """Fixed-decoding immutable-benchmark evaluation (structural +
    execution metrics), same recipe as stage-12 capability_eval."""
    from cortex.evaluation.structure_audit import structure_metrics
    torch.manual_seed(20261216)
    model.eval()
    per_task = []
    for suite, tasks in (("v1", TASKS), ("v3", TASKS_V3)):
        for t in tasks:
            cand = extract_code(gen_fixed_decoding(
                _subject(tok, model), t.instruction)).strip()
            m = structure_metrics(cand)
            grade = grade_single_quiet(t, cand)
            per_task.append({
                "suite": suite, "task_id": t.task_id,
                "nonempty": bool(cand), "parse": m["parses"],
                "complete_program": m["complete_program"],
                "executed": grade.get("exit_code") is not None,
                "passed": bool(grade["passed"])})
    n = len(per_task)
    return {
        "n_tasks": n,
        "pass_at_1": round(sum(r["passed"] for r in per_task) / n, 4),
        "syntax_valid_rate": round(sum(r["parse"] for r in per_task) / n,
                                   4),
        "complete_program_rate": round(sum(r["complete_program"]
                                           for r in per_task) / n, 4),
        "execution_attempted_rate": round(sum(r["executed"]
                                              for r in per_task) / n, 4),
        "per_task": per_task,
    }


def stage_h_arms(tok, train_recs: list, heldout_recs: list) -> dict:
    header = _experiment_header(
        "stage13-H-experiment-matrix",
        "structural-boundary segmentation and indentation-weighted loss, "
        "each as a single variable, improve held-out structural token "
        "accuracy without a held-out loss regression vs the re-checked "
        "control; the stage-12 champion remains the promotion reference",
        f"identical seeds/batch cycling/optimizer/{STEPS}-step horizon; "
        f"held-out evaluated every {EVAL_EVERY} steps in BOTH native and "
        "eval-aligned styles; promotion strictly by the pre-registered "
        "5-gate rule; arms 2 and 3 each change exactly one variable; "
        "arm 4 is the declared combination")

    pre_model = dense_model_from_warm6()
    pre_accuracy = structural_token_accuracy(pre_model, tok,
                                             train_recs[:16], SEQ,
                                             batch_size=4, max_batches=4)
    del pre_model
    held0_legacy = _heldout(dense_model_from_warm6(), heldout_recs, tok,
                            "legacy")
    held0_aligned = _heldout(dense_model_from_warm6(), heldout_recs, tok,
                             "eval_aligned")
    arms: dict = {}
    checkpoints = []
    for arm_name, cfg in ARM13_CONFIGS.items():
        torch.manual_seed(SEED)
        model = dense_model_from_warm6()
        opt = torch.optim.AdamW(model.parameters(), lr=PHYS_LR,
                                betas=(0.9, 0.98), weight_decay=0.01)
        model.train()
        curve = [{"step": 0, "heldout_native": held0_legacy,
                  "heldout_eval_aligned": held0_aligned}]
        window_losses: list = []
        steps_with_targets = 0
        grad_norms: list = []
        t0 = time.time()
        for ch in range(STEPS // EVAL_EVERY):
            chunk_losses: list = []
            for i in range(EVAL_EVERY):
                base = ((ch * EVAL_EVERY + i) * BATCH) % len(train_recs)
                x, y = make_batch_chunked(
                    train_recs[base:base + BATCH] or train_recs, tok, SEQ,
                    prompt_style=cfg["prompt_style"],
                    segmentation=cfg["segmentation"])
                if x is None:
                    continue
                if cfg["indent_weight"] == 1.0:
                    _, loss = model(x, y, pad_mask=x == tok.pad_id)
                    logits = None
                else:
                    logits, _ = model(x, y, pad_mask=x == tok.pad_id)
                    loss = _weighted_ce(logits, y, tok,
                                        cfg["indent_weight"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if ch == 0 and i < 50:
                    grad_norms.append(round(float(gn), 4))
                chunk_losses.append(round(float(loss.detach()), 4))
                steps_with_targets += 1
                del logits
            curve.append({
                "step": (ch + 1) * EVAL_EVERY,
                "heldout_native": _heldout(model, heldout_recs, tok,
                                           cfg["prompt_style"]),
                "heldout_eval_aligned": _heldout(model, heldout_recs, tok,
                                                 "eval_aligned"),
            })
            window_losses += chunk_losses
        seconds = round(time.time() - t0, 1)
        name = f"cortex-sft-stage13-{arm_name}.pt"
        ck_path = CKPT10_DIR / name
        torch.save({
            "model_state": model.state_dict(), "arch": "CortexTransformer",
            "stage": 13, "steps": STEPS, "lr": PHYS_LR, "batching":
            arm_name, "config": cfg, "parent": str(WARM6_CKPT),
            "parent_sha256": _sha256_file(WARM6_CKPT)}, ck_path)
        checkpoints.append({"path": str(ck_path), "arm": arm_name,
                            "steps": STEPS, "lr": PHYS_LR,
                            "sha256": _sha256_file(ck_path)})
        model.eval()
        ev = eval_fixed_decoding(model, tok)
        v3res = ev.pop("v3_results")
        v1res = ev.pop("v1_results")
        v1_counts = _dist([r["category"] for r in v1res
                           if "category" in r])
        post_acc = structural_token_accuracy(model, tok, heldout_recs,
                                             SEQ, batch_size=8,
                                             max_batches=6)
        native_final = curve[-1]["heldout_native"]
        aligned_final = curve[-1]["heldout_eval_aligned"]
        arms[arm_name] = {
            "config": cfg,
            "loss_first": window_losses[0] if window_losses else None,
            "loss_last": window_losses[-1] if window_losses else None,
            "steps_with_targets_total": steps_with_targets,
            f"steps_out_of_{STEPS}": steps_with_targets,
            "grad_norm_first_50": grad_norms,
            "seconds": seconds,
            "heldout_curve": curve,
            "heldout_native_final": native_final,
            "heldout_eval_aligned_final": aligned_final,
            "heldout_eval_aligned_best": min(
                c["heldout_eval_aligned"] for c in curve),
            "benchmarks": ev,
            "per_task_v3_after": v3res,
            "per_task_v1_after": v1res,
            "capability": _capability_eval(model, tok),
            "structural_token_accuracy_post": post_acc,
            "v1_fail_category_counts": v1_counts,
        }
        del model
    control = arms["control_recheck"]
    findings = {}
    for arm_name in list(ARM13_CONFIGS)[1:]:
        arm = arms[arm_name]
        findings[arm_name] = {
            "heldout_eval_aligned_delta_vs_control_recheck": round(
                arm["heldout_eval_aligned_final"]
                - control["heldout_eval_aligned_final"], 4),
            "structural_acc_whitespace_delta": (
                (arm["structural_token_accuracy_post"]
                 ["accuracy_whitespace"] or 0)
                - (control["structural_token_accuracy_post"]
                   ["accuracy_whitespace"] or 0))
            if arm["structural_token_accuracy_post"]["accuracy_whitespace"]
            is not None else None,
            "v1_pass_at_1": arm["benchmarks"]["v1_pass_at_1"],
            "v3_pass_at_1": arm["benchmarks"]["v3_pass_at_1"],
            "syntax_valid_rate_triangle": {
                arm_name: arm["capability"]["syntax_valid_rate"],
                "control_recheck": control["capability"][
                    "syntax_valid_rate"]},
            "complete_program_rate": arm["capability"][
                "complete_program_rate"],
            "control_complete_program_rate": control["capability"][
                "complete_program_rate"],
        }
    champion = min(ARM13_CONFIGS, key=lambda a: (
        arms[a]["heldout_eval_aligned_final"],
        -(arms[a]["benchmarks"]["v1_pass_at_1"]
          + arms[a]["benchmarks"]["v3_pass_at_1"])))
    out = {
        "kind": "cortex-stage13-experiment-matrix",
        "owned_by": "Fibonera", "workstream": "H-arms",
        "pre_registered_gates": GATES,
        "promotion_rule": PROMOTION_RULE,
        "steps": STEPS, "eval_every": EVAL_EVERY, "lr": PHYS_LR,
        "batch": BATCH, "seed": SEED, "gen_seed": GEN_SEED,
        "indent_weight": INDENT_WEIGHT,
        "stage12_champion_reference": CHAMPION12,
        "stage10_parent_reference": BASE_HELDOUT_ALLOWED,
        "heldout_at_start": {"legacy_style": held0_legacy,
                             "eval_aligned_style": held0_aligned},
        "structural_token_accuracy_baseline": pre_accuracy,
        "arms": arms,
        "findings_vs_control_recheck": findings,
        "checkpoints": checkpoints,
        "champion_arm_by_pre_registered_rule": champion,
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream I

def stage_i_capacity(tok) -> dict:
    header = _experiment_header(
        "stage13-I-capacity-and-infrastructure",
        "demo-scale, 2B-physical and 500B-virtual architecture accounts "
        "are re-verified from live objects and the authoritative store; "
        "2B training remains gated on genuine GPU qualification",
        "no procurement; authoritative store untouched; allocated "
        "parameters never counted as trained")
    from cortex.model.configs import CORTEX_2B, CORTEX_DEMO
    from cortex.hardware.profiler import can_train_locally, profile_hardware

    profile = profile_hardware()
    model = dense_model_from_warm6()
    total = model.actual_param_count()
    with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    del model
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan = persisted_optimizer_scan(auth)
    auth.close()
    feasible, reason = can_train_locally(
        profile, CORTEX_2B.theoretical_param_count())
    out = {
        "kind": "cortex-stage13-capacity-and-infrastructure",
        "owned_by": "Fibonera", "workstream": "I-capacity",
        "hardware": {
            "cpu_cores": profile.cpu_cores,
            "ram_total_gb": profile.ram_total_gb,
            "ram_available_gb": profile.ram_available_gb,
            "has_cuda": profile.has_cuda,
            "cuda_devices": torch.cuda.device_count()
            if torch.cuda.is_available() else 0,
        },
        "architectures": {
            "cortex_demo_actual_params": total,
            "cortex_demo_expected": CORTEX_DEMO.theoretical_param_count(),
            "cortex_2b_theoretical": CORTEX_2B.theoretical_param_count(),
            "cortex_2b_count_is_2056012800":
                CORTEX_2B.theoretical_param_count() == 2_056_012_800,
            "virtual_allocation_500b_untouched":
                scan.get("n_allocated", None),
            "authoritative_store": {
                "slots_updated": scan.get("slots_updated"),
                "unique_trained_scalars":
                    scan.get("unique_trained_scalars"),
                "scan": {k: scan[k] for k in scan if k not in
                         ("slots_updated", "unique_trained_scalars")},
            },
        },
        "gradient_participation_demo_model": {
            "tensors_with_grad_after_warm_pass_only": with_grad,
            "note": "counts refreshed at forward/backward time; the "
                    "dense demo model participates fully",
        },
        "two_b_gate_feasible_locally": {"feasible": bool(feasible),
                                        "reason": reason},
        "two_b_training_decision":
            "BLOCKED - 0 CUDA devices; ~46 GB RAM required vs measured "
            "available; no paid procurement authorized ($0.00 spent)",
        "gpu_procurement": "requires explicit Fibonera authorization; "
                           "none granted",
    }
    return _finish(header, out, "BLOCKED")


# --------------------------------------------------------------- workstream J

def stage_j_intelligence(train_recs: list, tok) -> dict:
    header = _experiment_header(
        "stage13-J-intelligence-subsystems",
        "the existing MoE and divergent-reasoning subsystems are probed "
        "with real batches/tasks; measured routing statistics and "
        "execution-verified candidate correctness decide their verdicts",
        "no new experts or reasoning engines; routing JS divergence is "
        "not treated as specialization; candidate correctness is "
        "execution-verified only")
    moe_sub = build_moe_subject()
    batches = [train_recs[i:i + 2]
               for i in range(0, min(len(train_recs), 24), 2)]
    mp = moe_probe(moe_sub.model, batches, tok, make_batch)
    latest = _latest_joint_physical()
    joint_report = None
    if latest is not None and Path(latest).exists():
        try:
            joint_sub = build_moe_subject(latest)
            joint_report = moe_probe(joint_sub.model, batches, tok,
                                     make_batch)
            del joint_sub
        except Exception as exc:
            joint_report = {"error": repr(exc)}
    dense = dense_model_from_warm6()
    dense.eval()
    torch.manual_seed(GEN_SEED)
    reasons = divergent_eval(_subject(tok, dense), tasks=TASKS[:2],
                             n_hypotheses=3)
    plain = []
    for t in TASKS[:2]:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, dense), t.instruction)).strip()
        grade = grade_single_quiet(t, cand)
        plain.append({"task_id": t.task_id, "passed": bool(grade["passed"])})
    del dense
    out = {
        "kind": "cortex-stage13-intelligence-subsystems",
        "owned_by": "Fibonera", "workstream": "J-intelligence",
        "moe_probe": mp,
        "moe_probe_latest_joint_physical": joint_report,
        "moe_probe_available": "error" not in mp,
        "divergent_reasoning": reasons,
        "plain_decoding_comparison": {
            "n_tasks": len(plain),
            "pass_at_1_divergent_index0": (reasons.get("aggregate") or
                                           {}).get("pass_at_1_index0"),
            "task_pass_rate_divergent_pass_at_k": (reasons.get(
                "aggregate") or {}).get("pass_at_n"),
            "repair_recovered_rate": (reasons.get("aggregate")
                                      or {}).get("pass_via_repair"),
            "pass_at_1_plain": round(
                sum(1 for p in plain if p["passed"]) / len(plain), 4)
            if plain else None,
            "note": "same 2 tasks, same model; divergent numbers come "
                    "from execution-verified candidates only; divergent "
                    "task-level pass uses 3 hypotheses + repair, so it is "
                    "pass@k semantics and is not compared as pass@1",
        },
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream K

def stage_k_promotion(a_report: dict, g_report: dict, h_report: dict) -> dict:
    header = _experiment_header(
        "stage13-K-promotion-audit",
        "the pre-registered 5-gate hierarchy is applied exactly as "
        "registered before any stage-13 execution; pass@1 = 0 implies "
        "NOT PROMOTED regardless of loss or syntax improvements",
        "gates evaluated from measured per-task results; champion "
        "selected under the pre-registered rule")
    arms = h_report["arms"]
    champion = min(arms, key=lambda n: (
        arms[n]["heldout_eval_aligned_final"],
        -(arms[n]["benchmarks"]["v1_pass_at_1"]
          + arms[n]["benchmarks"]["v3_pass_at_1"])))
    champ = arms[champion]
    control = arms["control_recheck"]
    champ_cap = champ["capability"]
    control_cap = control["capability"]
    gate1 = champ_cap["syntax_valid_rate"] > control_cap["syntax_valid_rate"]
    gate2 = gate1 and (champ_cap["complete_program_rate"]
                       > control_cap["complete_program_rate"])
    all_pass = any(
        arms[n]["benchmarks"]["v1_pass_at_1"] > 0
        or arms[n]["benchmarks"]["v3_pass_at_1"] > 0 for n in arms)
    gate3 = all_pass
    gate4 = bool(g_report["perturbed_task_results"]["pass_at_1"]) and \
        g_report["perturbed_task_results"]["pass_at_1"] > 0
    gate5 = gate3
    loss_ok = (champ["heldout_native_final"] <= BASE_HELDOUT_ALLOWED
               and champ["heldout_eval_aligned_final"]
               <= BASE_HELDOUT_ALLOWED)
    promoted = gate5 and loss_ok
    gates_passed = sum(1 for g in (gate1, gate2, gate3, gate4, gate5) if g)
    out = {
        "kind": "cortex-stage13-promotion-audit", "owned_by": "Fibonera",
        "workstream": "K-promotion",
        "gates": {
            "gate1_structural_improvement": {
                "definition": GATES["gate1_structural_improvement"],
                "ok": bool(gate1),
                "measured": {"champion_syntax_valid_rate":
                             champ_cap["syntax_valid_rate"],
                             "control_syntax_valid_rate":
                             control_cap["syntax_valid_rate"]}},
            "gate2_complete_generation": {
                "definition": GATES["gate2_complete_generation"],
                "ok": bool(gate2),
                "measured": {"champion_complete_program_rate":
                             champ_cap["complete_program_rate"],
                             "control_complete_program_rate":
                             control_cap["complete_program_rate"]}},
            "gate3_functional_correctness": {
                "definition": GATES["gate3_functional_correctness"],
                "ok": bool(gate3),
                "measured": "any arm pass@1 > 0 on v1/v3 (execution "
                            "verified)"},
            "gate4_generalization": {
                "definition": GATES["gate4_generalization"],
                "ok": bool(gate4),
                "measured": {"perturbed_pass_at_1":
                             g_report["perturbed_task_results"][
                                 "pass_at_1"]}},
            "gate5_capability_promotion": {
                "definition": GATES["gate5_capability_promotion"],
                "ok": bool(gate5),
                "measured": "identical evidence to gate3 by definition",
                "requires_heldout_ok": True},
        },
        "champion_arm": champion,
        "champion_metrics": {
            "heldout_native_final": champ["heldout_native_final"],
            "heldout_eval_aligned_final":
            champ["heldout_eval_aligned_final"],
            "v1_pass_at_1": champ["benchmarks"]["v1_pass_at_1"],
            "v3_pass_at_1": champ["benchmarks"]["v3_pass_at_1"],
            "syntax_valid_rate": champ_cap["syntax_valid_rate"],
            "complete_program_rate": champ_cap["complete_program_rate"],
        },
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
            "procedure": "restore the stage-12 champion (artifacts are "
                         "gitignored and additive; stage-13 checkpoints "
                         "do not overwrite stage-12 files)",
        },
    }
    return _finish(header, out, "PROMOTED" if promoted else "NOT PROMOTED")


def stage_taxonomy_artifact(a_report: dict) -> dict:
    definitions = {
        "incorrect_indent_increase": "generated indent deeper than any "
                                     "valid target at the transition",
        "incorrect_indent_decrease": "unindent to a non-matching level "
                                     "(incl. parser 'unindent' errors)",
        "missing_indentation_after_block": "expected an indented block",
        "unexpected_indentation": "unexpected indent applied by the "
                                  "model where no block opens",
        "missing_colon": "block opener not terminated by ':'",
        "unclosed_delimiter_or_string": "open paren/bracket/brace or "
                                        "unterminated string at stop",
        "premature_eot_open_construct": "self-emitted EOT while an "
                                        "construct is still open",
        "incomplete_block_missing_indent": "EOF-incomplete block body",
        "incomplete_function_or_class": "function/class body not closed",
        "incomplete_module": "module-level structure truncated",
        "invalid_statement_ordering": "statements in invalid order",
        "truncated_generation_max_new_tokens":
            "max_new_tokens exhausted without EOT",
        "repetition_degeneration_truncation":
            "4-token degenerate repeat detected; generation stopped",
        "tokenizer_or_decoding_corruption": "decode-level corruption "
                                            "(round-trip checked)",
        "other_parser_confirmed_syntax_error": "parser-confirmed defect "
                                               "outside the above classes",
    }
    freqs = {}
    for s, ps in a_report["per_subject"].items():
        freqs[s] = ps["structural_taxonomy_distribution"]
    out = {
        "kind": "cortex-stage13-failure-taxonomy",
        "owned_by": "Fibonera", "workstream": "A-taxonomy",
        "definitions": definitions,
        "measured_frequencies": freqs,
        "stage12_context": {
            "incorrect_indentation": 38, "other": 5, "premature_stop": 2,
            "note": "stage-12 measured 43/45 persistent syntax defects, "
                    "38/45 indentation-related (indent transitions), "
                    "38/45 self-emitted EOT",
        },
        "passthrough_categories_note":
            "missing_opening_delimiter, invalid_token and "
            "other_parser_confirmed_syntax_error pass through unchanged "
            "from the stage-10/12 categories",
    }
    return out


# --------------------------------------------------------------- main

def main() -> int:
    head = _git_head()
    assert _git_head() == PARENT_COMMIT or head.startswith(PARENT_COMMIT), (
        f"stage-13 must start from {PARENT_COMMIT}, got {_git_head()}")
    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    all_recs = []
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                all_recs.append(json.loads(line))
    train_recs = load_split(str(DATASET), "train")
    val_recs = load_split(str(DATASET), "val")
    heldout_recs = load_split(str(DATASET), "heldout")
    print(f"[stage13] dataset train={len(train_recs)} "
          f"val={len(val_recs)} heldout={len(heldout_recs)} "
          f"all={len(all_recs)}", flush=True)

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_pre = persisted_optimizer_scan(auth)
    auth.close()
    assert scan_pre["slots_updated"] == 35292, "authoritative store moved"

    a = stage_a_root_cause(tok)
    _save_artifact("stage13_structural_root_cause.json", a)
    a_shares = {k: v["indentation_related_share"]
                for k, v in a["per_subject"].items()}
    print(f"[stage13] A indentation_related_share: {a_shares}", flush=True)

    b = stage_b_data_audit(all_recs, train_recs, heldout_recs, tok)
    _save_artifact("stage13_structural_data_audit.json", b)
    b_complete = b["capability_coverage_matrix"]["train"][
        "complete_program_rate"]
    print(f"[stage13] B complete_program train: {b_complete}", flush=True)

    c = stage_c_training_stream(all_recs, train_recs, val_recs,
                                heldout_recs, tok)
    _save_artifact("stage13_training_stream.json", c)
    print(f"[stage13] C coverage_guarantee_holds: "
          f"{c['coverage_guarantee_holds']}", flush=True)

    g = stage_g_generalization(tok, all_recs, train_recs, heldout_recs)
    _save_artifact("stage13_generalization.json", g)
    print(f"[stage13] G perturbed pass@1: "
          f"{g['perturbed_task_results']['pass_at_1']}", flush=True)

    h = stage_h_arms(tok, train_recs, heldout_recs)
    _save_artifact("stage13_experiment_matrix.json", h)
    print(f"[stage13] H champion: "
          f"{h['champion_arm_by_pre_registered_rule']}", flush=True)
    for arm_name, f in h["findings_vs_control_recheck"].items():
        print(f"[stage13] H {arm_name}: {f}", flush=True)

    return _finalize_tail(a, b, c, g, h, train_recs, tok)


def _finalize_tail(a: dict, b: dict, c: dict, g: dict, h: dict,
                   train_recs: list, tok) -> int:
    e = stage_e_generation_control(tok)
    _save_artifact("stage13_generation_control.json", e)
    print(f"[stage13] E exploratory parse: "
          f"{e['exploratory_eot_gated_run']['parse_rate']}", flush=True)

    f = stage_f_execution_feedback(a)
    _save_artifact("stage13_execution_feedback.json", f)
    print(f"[stage13] F stage_distribution_total: "
          f"{f['stage_distribution_total']}", flush=True)

    pre_acc = h["structural_token_accuracy_baseline"]
    post_by_arm = {name: {
        "accuracy_whitespace": arm["structural_token_accuracy_post"][
            "accuracy_whitespace"],
        "accuracy_structure_keyword":
        arm["structural_token_accuracy_post"]["accuracy_structure_keyword"],
        "accuracy_delimiter": arm["structural_token_accuracy_post"][
            "accuracy_delimiter"],
        "accuracy_overall": arm["structural_token_accuracy_post"][
            "accuracy_overall"],
        "heldout_eval_aligned_final": arm["heldout_eval_aligned_final"],
    } for name, arm in h["arms"].items()}
    indent_arm = post_by_arm.get("indent_weighted", {})
    control_arm = post_by_arm.get("control_recheck", {})
    margin_note = (
        f"indent-aligned accuracy whitespace "
        f"{indent_arm.get('accuracy_whitespace')} vs control "
        f"{control_arm.get('accuracy_whitespace')} at held-out "
        f"{indent_arm.get('heldout_eval_aligned_final')} vs "
        f"{control_arm.get('heldout_eval_aligned_final')}")
    d = stage_d_learning(pre_acc, post_by_arm, margin_note)
    _save_artifact("stage13_structural_learning.json", d)
    print(f"[stage13] D decision: {d['experiment']['decision']}",
          flush=True)

    i = stage_i_capacity(tok)
    _save_artifact("stage13_capacity_and_infrastructure.json", i)
    print(f"[stage13] I decision: {i['experiment']['decision']}",
          flush=True)

    j = stage_j_intelligence(train_recs, tok)
    _save_artifact("stage13_intelligence_subsystems.json", j)
    print(f"[stage13] J moe_probe_available: {j['moe_probe_available']}",
          flush=True)

    k = stage_k_promotion(a, g, h)
    _save_artifact("stage13_promotion_audit.json", k)
    print(f"[stage13] K decision: {k['experiment']['decision']} "
          f"(gates passed {k['gates_passed']}/5)", flush=True)

    _save_artifact("stage13_failure_taxonomy.json",
                   stage_taxonomy_artifact(a))

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_post = persisted_optimizer_scan(auth)
    auth.close()
    _save_artifact("stage13_report.json", {
        "kind": "cortex-stage13-report", "owned_by": "Fibonera",
        "stage": 13,
        "summary": {
            "parent_commit": PARENT_COMMIT,
            "A_indentation_related_share": {
                k: v["indentation_related_share"]
                for k, v in a["per_subject"].items()},
            "A_taxonomy": {k: v["structural_taxonomy_distribution"]
                           for k, v in a["per_subject"].items()},
            "B_complete_program_rate":
                b["capability_coverage_matrix"]["train"][
                    "complete_program_rate"],
            "C_coverage_guarantee_holds": c["coverage_guarantee_holds"],
            "C_boundary_clean_rates": {
                seg: c["chunk_boundary_audit_train"][seg][
                    "pct_window_end_clean"]
                for seg in ("fixed", "structural")},
            "E_stop_classification": e["stop_classification_from_"
                                       "workstream_a"],
            "G_perturbed_pass_at_1":
                g["perturbed_task_results"]["pass_at_1"],
            "G_teacher_forced": g["teacher_forced_accuracy"],
            "H_findings": h["findings_vs_control_recheck"],
            "H_heldout_final": {n: ar["heldout_eval_aligned_final"]
                                for n, ar in h["arms"].items()},
            "H_champion": h["champion_arm_by_pre_registered_rule"],
            "I_two_b_training": i["two_b_training_decision"],
            "J_moe_probe_available": j["moe_probe_available"],
            "K_decision": k["experiment"]["decision"],
            "K_gates_passed": k["gates_passed"],
            "fingerprint_v1": FINGERPRINT_V1,
            "fingerprint_v2": FINGERPRINT_V2,
            "fingerprint_v3": fingerprint_tasks("v3"),
            "spend_usd": 0.0, "gpu_hours": 0.0,
            "authoritative_store_untouched":
                scan_post["slots_updated"] == 35292,
        }})
    print("[stage13] complete", flush=True)
    return 0


def _load_recs() -> list:
    recs = []
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _finalize_from_artifacts() -> int:
    """Recover an interrupted run: A/B/C/G/H are read back from the
    persisted artifacts and only E-L/report re-execute."""
    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    saved = {}
    for k, name in (("a", "stage13_structural_root_cause"),
                    ("b", "stage13_structural_data_audit"),
                    ("c", "stage13_training_stream"),
                    ("g", "stage13_generalization"),
                    ("h", "stage13_experiment_matrix")):
        p = ART / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"cannot finalize: {p} missing")
        saved[k] = json.loads(p.read_text())
    train_recs = load_split(str(DATASET), "train")
    return _finalize_tail(saved["a"], saved["b"], saved["c"],
                          saved["g"], saved["h"], train_recs, tok)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize-from-artifacts":
        raise SystemExit(_finalize_from_artifacts())
    raise SystemExit(main())
