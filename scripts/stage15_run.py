"""Stage 15 orchestrator: structural-learning consolidation,
execution-feedback training redesign, execution-verified data
generation, generalization, self-correction, optimization audit,
capacity/subsystem audits, 6-gate promotion attempt and report.

All workstreams run against the real demo model, real tokenizer, real
training coordinator and real execution sandbox. Nothing is simulated.
$0.00 is spent and the authoritative 500B virtual store is never moved.

Pre-registered hypotheses (fixed BEFORE this file ever executed):

  A) the stage-14 structural-segmentation held-out improvement over the
     same-horizon control (2.9124 vs 2.9250 eval-aligned at 6000 steps
     from warm-6) reproduces at a fresh pre-registered seed.
  B) the structural arm's learned behavior shows up as improved
     indentation-transition and structural-token prediction, and some
     nested/class structures appear free-running.
  C) evidence COMPACTED feedback prompts (error class + single final
     error line, cap 160 chars) restore 100% target-bearing step
     coverage vs stage-14's 102/600 and avoid the stage-14 held-out
     regression (+0.2676); outcome judged by pre-registered gates.
  D) failures of multiple independently generated candidates classify
     into reproducible execution-error categories with zero benchmark
     contamination.
  E) success on the immutable benchmark does not yet exist; the stage
     measures whether ANY execution-verified success appears on the new
     unseen suite or perturbations.
  F) model-generated repair conditioned on its own compacted error
     evidence does not improve success over direct generation unless
     evidence-consistent revisions appear (stage-14 was 0/15).
  G) structural-token weighting and feedback data change gradient
     magnitudes measurably; only measured, no combined arms.
"""
from __future__ import annotations

import json
import random
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
from cortex.evaluation.structure_audit import (  # noqa: E402
    structure_metrics)
from cortex.evaluation.tasks import TASKS  # noqa: E402
from cortex.evaluation.tasks_v3 import TASKS_V3  # noqa: E402
from cortex.training.sft import load_split, make_batch  # noqa: E402
from cortex.virtual.coverage import persisted_optimizer_scan  # noqa: E402
from cortex.virtual.store import VirtualParameterStore  # noqa: E402

from scripts.stage10_run import (  # noqa: E402
    ART, CKPT10_DIR, DATASET, FINGERPRINT_V1, FINGERPRINT_V2, SEQ,
    _finish, _save_artifact, _sha256_file, dense_model_from_warm6,
    gen_fixed_decoding)
from scripts.stage13_run import (  # noqa: E402
    grade_single_quiet, _weighted_ce)
from scripts.stage9_run import build_moe_subject  # noqa: E402
from scripts.stage14_run import (  # noqa: E402
    STAGE13_CHAMPION_HELDOUT, _capability_eval,
    _dist, _heldout, _load_arm, _sample_train_records,
    _subject, _with_suite_pass, make_feedback_row, structural_token_accuracy,
    train_arm, verify_candidate)

PARENT_COMMIT = "318e601"
DECODING15 = {"temperature": 0.2, "top_k": 50, "max_new_tokens": 180}
A_STEPS = 6000
A_EVAL_EVERY = 1000
A_LR = 5e-5
A_BATCH = 4
SEED_A_REP2 = 20261601
STAGE15_REFERENCE_ARM = "stage14-structural_h6000"
FB15_RECORDS = 32
FB15_STEPS = 600
FB15_LR = 5e-5
FB15_BATCH = 4
FB15_SEED = 20261621
FB15_GEN_SEED = 20261620
COMPACT_EVIDENCE_CHARS = 160
D_RECORDS = 12
D_SAMPLES = 3
GEN_SEED15 = 20261630
PERTURB_SEED_BASE15 = 20261700
G_HELDOUT_LIMIT = 3.3015
STAGE14_CHAMPION_REF = {
    "heldout_native": 2.9074, "heldout_eval_aligned": 2.9124,
    "syntax_valid_rate": 0.0, "complete_program_rate": 0.0,
    "v1_pass_at_1": 0.0, "v3_pass_at_1": 0.0,
    "feedback_arm_heldout_eval_aligned": 3.2174,
    "feedback_target_bearing_steps": 102,
}
STAGE14_REP1 = {"control_heldout_eval_aligned": 2.9250,
                "structural_heldout_eval_aligned": 2.9124}


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z")


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True).stdout.strip()[:12]


def _experiment_header15(experiment_id: str, hypothesis: str,
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


# --------------------------------------------------------------- unseen suite

@dataclass(frozen=True)
class UnseenTask:
    task_id: str
    instruction: str
    module_name: str
    test_code: str
    reference_solution: str = ""


UNSEEN_TASKS_STAGE15: list[UnseenTask] = [
    UnseenTask(
        task_id="t15-matrix-transpose",
        instruction=("Write module `task.py` defining `transpose(matrix)` "
                     "that returns the transpose of a list of lists."),
        module_name="task.py",
        test_code=(
            "from task import transpose\n\n"
            "def test_square():\n"
            "    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n\n"
            "def test_rect():\n"
            "    assert transpose([[1, 2, 3]]) == [[1], [2], [3]]\n\n"
            "def test_empty():\n"
            "    assert transpose([]) == []\n"
        ),
        reference_solution=(
            "def transpose(matrix):\n"
            "    if not matrix:\n"
            "        return []\n"
            "    return [list(row) for row in zip(*matrix)]\n"
        ),
    ),
    UnseenTask(
        task_id="t15-caesar-shift",
        instruction=("Write module `task.py` defining `caesar(text, shift)` "
                     "that shifts letters by shift positions, keeps case "
                     "and leaves other characters unchanged."),
        module_name="task.py",
        test_code=(
            "from task import caesar\n\n"
            "def test_basic():\n"
            "    assert caesar('abc', 1) == 'bcd'\n\n"
            "def test_wrap():\n"
            "    assert caesar('xyz', 3) == 'abc'\n\n"
            "def test_case_and_punct():\n"
            "    assert caesar('Hello, World!', 13) == 'Uryyb, Jbeyq!'\n"
        ),
        reference_solution=(
            "def caesar(text, shift):\n"
            "    out = []\n"
            "    for ch in text:\n"
            "        if 'a' <= ch <= 'z':\n"
            "            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))\n"
            "        elif 'A' <= ch <= 'Z':\n"
            "            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))\n"
            "        else:\n"
            "            out.append(ch)\n"
            "    return ''.join(out)\n"
        ),
    ),
    UnseenTask(
        task_id="t15-multi-function-stats",
        instruction=("Write module `task.py` defining `mean(values)` and "
                     "`zscores(values)`; zscores must use mean."),
        module_name="task.py",
        test_code=(
            "from task import mean, zscores\n\n"
            "def test_mean():\n"
            "    assert mean([2, 4, 6]) == 4.0\n\n"
            "def test_zscores_zero_spread():\n"
            "    assert zscores([5, 5, 5]) == [0.0, 0.0, 0.0]\n\n"
            "def test_zscores_symmetric():\n"
            "    z = zscores([1, 2, 3])\n"
            "    assert abs(z[0] + z[2]) < 1e-9 and z[1] == 0.0\n"
        ),
        reference_solution=(
            "def mean(values):\n"
            "    return sum(values) / len(values)\n\n\n"
            "def zscores(values):\n"
            "    m = mean(values)\n"
            "    var = sum((v - m) ** 2 for v in values) / len(values)\n"
            "    if var == 0:\n"
            "        return [0.0] * len(values)\n"
            "    std = var ** 0.5\n"
            "    return [(v - m) / std for v in values]\n"
        ),
    ),
    UnseenTask(
        task_id="t15-closure-adder",
        instruction=("Write module `task.py` defining `make_adder(n)` that "
                     "returns a one-argument function adding n to its "
                     "input (nested function required)."),
        module_name="task.py",
        test_code=(
            "from task import make_adder\n\n"
            "def test_basic():\n"
            "    add5 = make_adder(5)\n"
            "    assert add5(2) == 7\n\n"
            "def test_independent():\n"
            "    a = make_adder(1)\n"
            "    b = make_adder(10)\n"
            "    assert a(0) == 1 and b(0) == 10\n\n"
            "def test_negative():\n"
            "    assert make_adder(-3)(3) == 0\n"
        ),
        reference_solution=(
            "def make_adder(n):\n"
            "    def adder(x):\n"
            "        return x + n\n"
            "    return adder\n"
        ),
    ),
    UnseenTask(
        task_id="t15-safe-div",
        instruction=("Write module `task.py` defining `safe_div(a, b)` "
                     "returning a/b as float, and None when the "
                     "denominator is zero (exception handling required)."),
        module_name="task.py",
        test_code=(
            "from task import safe_div\n\n"
            "def test_normal():\n"
            "    assert safe_div(7, 2) == 3.5\n\n"
            "def test_zero():\n"
            "    assert safe_div(1, 0) is None\n\n"
            "def test_negative():\n"
            "    assert safe_div(-6, 3) == -2.0\n"
        ),
        reference_solution=(
            "def safe_div(a, b):\n"
            "    try:\n"
            "        return a / b\n"
            "    except ZeroDivisionError:\n"
            "        return None\n"
        ),
    ),
    UnseenTask(
        task_id="t15-word-counter-class",
        instruction=("Write module `task.py` defining class `WordCounter` "
                     "with `add(word)`, `add_text(text)` and "
                     "`count(word)` methods (classes required)."),
        module_name="task.py",
        test_code=(
            "from task import WordCounter\n\n"
            "def test_add():\n"
            "    c = WordCounter()\n"
            "    c.add('a')\n"
            "    c.add('a')\n"
            "    assert c.count('a') == 2\n\n"
            "def test_add_text():\n"
            "    c = WordCounter()\n"
            "    c.add_text('the cat the dog')\n"
            "    assert c.count('the') == 2\n\n"
            "def test_missing():\n"
            "    c = WordCounter()\n"
            "    assert c.count('nope') == 0\n"
        ),
        reference_solution=(
            "class WordCounter:\n"
            "    def __init__(self):\n"
            "        self.counts = {}\n\n"
            "    def add(self, word):\n"
            "        self.counts[word] = self.counts.get(word, 0) + 1\n\n"
            "    def add_text(self, text):\n"
            "        for w in text.split():\n"
            "            self.add(w)\n\n"
            "    def count(self, word):\n"
            "        return self.counts.get(word, 0)\n"
        ),
    ),
    UnseenTask(
        task_id="t15-group-by-length",
        instruction=("Write module `task.py` defining `group_by_length("
                     "words)` returning a dict mapping each word length "
                     "to the sorted list of words of that length."),
        module_name="task.py",
        test_code=(
            "from task import group_by_length\n\n"
            "def test_basic():\n"
            "    assert group_by_length(['bb', 'a', 'ccc', 'dd']) == \\\n"
            "        {1: ['a'], 2: ['bb', 'dd'], 3: ['ccc']}\n\n"
            "def test_empty():\n"
            "    assert group_by_length([]) == {}\n\n"
            "def test_sorted():\n"
            "    assert group_by_length(['ba', 'ab']) == {2: ['ab', 'ba']}\n"
        ),
        reference_solution=(
            "def group_by_length(words):\n"
            "    groups = {}\n"
            "    for w in words:\n"
            "        groups.setdefault(len(w), []).append(w)\n"
            "    return {k: sorted(v) for k, v in groups.items()}\n"
        ),
    ),
    UnseenTask(
        task_id="t15-window-max",
        instruction=("Write module `task.py` defining `max_window("
                     "values, k)` returning the list of maximum values "
                     "of each consecutive window of size k (use an "
                     "empty list when len(values) < k)."),
        module_name="task.py",
        test_code=(
            "from task import max_window\n\n"
            "def test_basic():\n"
            "    assert max_window([1, 3, 2, 5], 2) == [3, 3, 5]\n\n"
            "def test_short():\n"
            "    assert max_window([1], 2) == []\n\n"
            "def test_k_equals_len():\n"
            "    assert max_window([4, 2], 2) == [4]\n"
        ),
        reference_solution=(
            "def max_window(values, k):\n"
            "    if len(values) < k or k <= 0:\n"
            "        return []\n"
            "    return [max(values[i:i + k])\n"
            "            for i in range(len(values) - k + 1)]\n"
        ),
    ),
]


def fingerprint_unseen() -> str:
    import hashlib
    return hashlib.sha256(json.dumps(sorted(
        (t.task_id, t.instruction, t.module_name, t.test_code,
         t.reference_solution)
        for t in UNSEEN_TASKS_STAGE15),
        sort_keys=True).encode("utf-8")).hexdigest()


# ------------------------------------------- execution-failure classification

_ERROR_SIGNS = (
    ("syntax_failure",
     ("SyntaxError", "IndentationError", "TabError", "invalid syntax")),
    ("import_failure",
     ("ModuleNotFoundError", "ImportError", "No module named")),
    ("timeout", ("TimeoutExpired", "timed out")),
)
_HIDDEN_TEST_SIGNS = ("AssertionError", "== FAILED", "short test summary")


def classify_execution_failure(verdict: dict) -> str:
    """Classify one execution verdict (verify_candidate or grade_single
    shape) into the stage-15 execution-error category from its own
    captured evidence."""
    if verdict.get("passed"):
        return "success"
    err = verdict.get("exec_stderr") or ""
    fail = verdict.get("exec_failure") or ""
    executed = bool(verdict.get("executed") or
                    verdict.get("exit_code") is not None)
    if not executed and not err and not fail:
        return "incomplete_generation"
    if int(verdict.get("doctest_failures") or 0) > 0:
        return "failed_reference_doctest"
    for cls, signs in _ERROR_SIGNS:
        if any(s in err for s in signs):
            return cls
    if any(s in err for s in _HIDDEN_TEST_SIGNS):
        return "failed_hidden_test"
    if executed:
        return "runtime_exception"
    if fail:
        if any(s in fail for s in _ERROR_SIGNS[2][1]):
            return "timeout"
        return "runtime_exception"
    return "incomplete_generation"


def _last_error_line(stderr: str) -> str:
    lines = [l.strip() for l in (stderr or "").strip().splitlines()
             if l.strip()]
    return lines[-1][:COMPACT_EVIDENCE_CHARS] if lines else ""


def _signature_error_line(text: str) -> str:
    """Source-relevant evidence line: for doctest output the failing
    example's call line; otherwise the last non-empty line."""
    lines = [l.strip() for l in (text or "").strip().splitlines()
             if l.strip()]
    if not lines:
        return ""
    for i, l in enumerate(lines):
        if "Failed example" in l:
            return (lines[i + 1] if i + 1 < len(lines) else l)[
                :COMPACT_EVIDENCE_CHARS]
    return lines[-1][:COMPACT_EVIDENCE_CHARS]


def compact_evidence(verdict: dict) -> str:
    """Pre-registered evidence compaction: error class + one final
    source-relevant error line (cap 160 chars), preserving the original
    sandbox stderr in the data record."""
    cls = classify_execution_failure(verdict)
    if cls == "success":
        return ""
    line = _signature_error_line(verdict.get("exec_stderr"))
    if not line:
        line = _signature_error_line(verdict.get("doctest_evidence"))
    return f"{cls}: {line}".strip()[:COMPACT_EVIDENCE_CHARS + 32]


def make_feedback_row15(rec: dict, evidence: str) -> dict:
    prompt_text = ("# Task: " + rec["instruction"]
                   + "\n# Attempt failed with " + evidence + "\n")
    return {"name": f"fb15-{rec['name']}", "instruction": prompt_text,
            "code": rec["code"], "split": "train",
            "provenance": {"kind": "stage15-compacted-execution-feedback",
                           "error_class": evidence.split(":")[0],
                           "original_stderr_preserved_in_record": True,
                           "parent": rec.get("provenance")},
            "original_stderr_tail": (verdict_stderr(rec["name"]) or "")}


_VERDICT_STDERR: dict = {}


def verdict_stderr(name: str):
    return _VERDICT_STDERR.get(name)


def feedback_coverage(row: dict, tok) -> dict:
    """Coverage audit: does the intended solution completion receive
    gradients in a real make_batch training row?"""
    x, y = make_batch([row], tok, SEQ)
    comp = tok.encode(row["code"].strip() + "\n" + tok.EOT)
    if x is None:
        return {"fits": False, "n_supervised": 0, "len_completion":
                len(comp), "exact_match": False}
    supervised = [v for v in y[0].tolist() if v != -100]
    return {"fits": True, "n_supervised": len(supervised),
            "len_completion": len(comp),
            "exact_match": supervised == comp}


# --------------------------------------------------------------- workstream A

def stage_a_reproducibility(tok, train_recs: list, heldout_recs: list
                            ) -> dict:
    header = _experiment_header15(
        "stage15-A-structural-reproducibility",
        "the stage-14 structural-segmentation held-out improvement over "
        "the same-horizon control (2.9124 vs 2.9250 eval-aligned at 6000 "
        "steps from warm-6) reproduces at the fresh pre-registered seed "
        f"{SEED_A_REP2}",
        "two new 6000-step arms differing ONLY in segmentation "
        f"(fixed vs structural), seed {SEED_A_REP2}, lr {A_LR}, batch "
        f"{A_BATCH}, eval every {A_EVAL_EVERY} steps in both prompt "
        "styles; no seed selection after results; promotion strictly by "
        "the pre-registered gates")
    arms = {
        "control_rep2": train_arm(
            tok, "control_rep2", {"prompt_style": "legacy",
                                  "segmentation": "fixed"},
            train_recs, heldout_recs, A_STEPS, A_EVAL_EVERY,
            seed=SEED_A_REP2, checkpoint_prefix="cortex-sft-stage15"),
        "structural_rep2": train_arm(
            tok, "structural_rep2", {"prompt_style": "legacy",
                                             "segmentation": "structural"},
            train_recs, heldout_recs, A_STEPS, A_EVAL_EVERY,
            seed=SEED_A_REP2, checkpoint_prefix="cortex-sft-stage15"),
    }
    ctrl = arms["control_rep2"]
    struct = arms["structural_rep2"]
    delta_rep2 = round(struct["heldout_eval_aligned_final"]
                       - ctrl["heldout_eval_aligned_final"], 4)
    rep1_delta = round(STAGE14_REP1["structural_heldout_eval_aligned"]
                       - STAGE14_REP1["control_heldout_eval_aligned"], 4)
    findings = {
        "rep1_delta_stage14": rep1_delta,
        "rep2_delta_stage15_seed": delta_rep2,
        "structural_beats_control_in_both_reps": bool(
            rep1_delta < 0 and delta_rep2 < 0),
        "rep2_structural_vs_stage13_champion": round(
            struct["heldout_eval_aligned_final"]
            - STAGE13_CHAMPION_HELDOUT["eval_aligned"], 4),
        "reproduces": bool(rep1_delta < 0 and delta_rep2 < 0),
    }
    out = {
        "kind": "cortex-stage15-reproducibility",
        "owned_by": "Fibonera", "workstream": "A-reproducibility",
        "steps": A_STEPS, "lr": A_LR, "batch": A_BATCH,
        "rep1_stage14_reference": STAGE14_REP1,
        "arms": arms,
        "findings": findings,
    }
    return _finish(header, out,
                   "PASS" if findings["reproduces"] else
                   "FAIL - not reproduced, negative result preserved")


# --------------------------------------------------------------- workstream B

STRUCTURAL_PROMPTS15 = [
    ("nested_function", "Write a module with `make_counter()` that defines "
     "an inner step function and returns it."),
    ("nested_control", "Write a function `classify_grid(grid)` with nested "
     "for loops and inner if/else conditions."),
    ("class_with_methods", "Write a class `Stack` with `__init__`, `push`, "
     "`pop` and `is_empty` methods."),
    ("exception_handling", "Write a function `parse_int(s)` using try/except "
     "returning the int or None."),
    ("imports_and_module", "Write a module that imports functools and defines "
     "`apply_twice(f, v)` using functools.reduce."),
    ("multi_function", "Write a module with `normalize(values)` and "
     "`scaled(values, lo, hi)` where the second uses the first."),
    ("docstring_program", "Write a documented function `euclid(a, b)` with a "
     "full docstring and doctest examples."),
    ("longer_program", "Write a module with `fib(n)`, `prime(n)` and a "
     "module-level main guard executing both."),
]


def indent_transition_stats(model, tok, recs) -> dict:
    """Teacher-forced exact-token accuracy at indentation-transition
    positions of the reference completions (real forward passes, no
    repair, no grammar constraints)."""
    model.eval()
    stats = {"positions": 0, "correct": 0, "per_example": []}
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
            good = bad = 0
            for j in range(len(comp)):
                if not cum[j].endswith("\n"):
                    continue
                line_idx = cum[j].count("\n")
                if line_idx <= 0 or line_idx >= n_lines:
                    continue
                nxt = lines[line_idx]
                prev = lines[line_idx - 1]
                ind_n = len(nxt) - len(nxt.lstrip(" "))
                ind_p = len(prev) - len(prev.lstrip(" "))
                if ind_n == ind_p:
                    continue
                ref_id = comp[j]
                ok = preds[len(prompt_ids) + j - 1] == ref_id
                if nxt.strip():
                    stats["positions"] += 1
                    stats["correct"] += int(ok)
                    good += int(ok)
                    bad += int(not ok)
            if good + bad:
                stats["per_example"].append(
                    {"name": rec["name"], "transitions": good + bad,
                     "correct": good})
    stats["accuracy"] = round(stats["correct"] / stats["positions"], 4) \
        if stats["positions"] else None
    del stats["per_example"][:24]
    return stats


def stage_b_structural(tok, heldout_recs: list) -> dict:
    header = _experiment_header15(
        "stage15-B-structural-consolidation",
        "the structural arm's held-out improvement corresponds to "
        "measurably better structural-token and indentation-transition "
        "prediction, and free-running generation produces structural "
        "constructs (classes/nesting) absent from training data",
        "teacher-forced per-class structural accuracy + exact-token "
        "indentation-transition accuracy on the real held-out split; 8 "
        "free-running structural probes captured with per-step "
        "trajectories and classified by the existing 13-category "
        "taxonomy; no parser-based repair before evaluation")
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    stacc = structural_token_accuracy(model, tok, heldout_recs, SEQ)
    ind = indent_transition_stats(model, tok, heldout_recs)
    free_running = []
    torch.manual_seed(GEN_SEED15)
    for label, prompt in STRUCTURAL_PROMPTS15:
        traj = trajectory_generate(model, tok, prompt, DECODING15)
        text = traj["text"]
        m = structure_metrics(extract_code(text)) if text.strip() \
            else {"parses": False, "complete_program": False}
        cls = classify_structural_failure(traj)
        scan = structure_scan(text or "")
        free_running.append({
            "probe": label, "generated_text_tail": (text or "")[-400:],
            "parse": m["parses"], "complete_program":
            m["complete_program"], "first_persistent_error":
            cls.get("taxonomy"),
            "lex_state_end": {"paren_depth": scan.get("paren_depth"),
                              "brace_depth": scan.get("brace_depth"),
                              "open_quote": scan.get("open_quote"),
                              "unbalanced": scan.get("unbalanced")},
            "def_count": text.count("def "), "class_count":
            text.count("class "),
            "nested_def": "\n    def " in text,
            "stopped": traj.get("stopped"),
        })
    del model
    n = len(free_running)
    out = {
        "kind": "cortex-stage15-structural-consolidation",
        "owned_by": "Fibonera", "workstream": "B-structural",
        "reference_arm": STAGE15_REFERENCE_ARM,
        "teacher_forced": {
            "structural_token_accuracy": stacc,
            "indentation_transition_accuracy": ind,
        },
        "free_running": {
            "n_probes": n,
            "parse_rate": round(sum(p["parse"] for p in free_running) / n,
                                4) if n else None,
            "complete_program_rate": round(
                sum(p["complete_program"] for p in free_running) / n, 4)
            if n else None,
            "probes_with_class_construct": sum(
                p["class_count"] > 0 for p in free_running),
            "probes_with_nested_def": sum(p["nested_def"] for p in
                                          free_running),
            "per_probe": free_running,
        },
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream C

def stage_c_feedback(tok, train_recs: list, heldout_recs: list) -> dict:
    header = _experiment_header15(
        "stage15-C-feedback-coverage-redesign",
        "compact evidence-bearing prompts (error class + final error "
        f"line, cap {COMPACT_EVIDENCE_CHARS} chars) restore 100% "
        "target-bearing step coverage and remove the stage-14 held-out "
        "regression; outcome judged only by the pre-registered gates",
        f"FB15_RECORDS={FB15_RECORDS}, FB15_STEPS={FB15_STEPS}, lr "
        f"{FB15_LR}, batch {FB15_BATCH}, seed {FB15_SEED}; warm-start "
        f"from the pre-declared stage-15 reference arm "
        f"{STAGE15_REFERENCE_ARM} (configuration-based, not result-based "
        "selection); every training row proves full solution supervision "
        "before training (coverage audit); stage-14's 600-step arm "
        "remains the negative control")
    recs = _sample_train_records(train_recs, FB15_RECORDS, FB15_GEN_SEED)
    bench_instructions = {t.instruction for t in TASKS} | \
        {t.instruction for t in TASKS_V3}
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    generations = []
    contamination_hits = 0
    torch.manual_seed(FB15_GEN_SEED)
    for rec in recs:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), rec["instruction"])).strip()
        verdict = verify_candidate(cand, rec)
        if rec["instruction"] in bench_instructions:
            contamination_hits += 1
        _VERDICT_STDERR[rec["name"]] = verdict.get("exec_stderr") or ""
        generations.append({
            "rec_name": rec["name"], "executed": verdict["executed"],
            "exec_stderr": verdict["exec_stderr"],
            "doctests_total": verdict["doctests_total"],
            "doctest_failures": verdict["doctest_failures"],
            "passed": verdict["passed"],
            "error_class": classify_execution_failure(verdict)})
    del model
    coverage = []
    old_style_coverage = []
    ordered = []
    for rec, gen in zip(recs, generations):
        if gen["passed"]:
            continue
        evidence = compact_evidence({
            "passed": gen["passed"], "executed": gen["executed"],
            "exec_failure": "sandbox_exec_failed"
            if not gen["executed"] else None,
            "exec_stderr": gen["exec_stderr"],
            "doctest_failures": gen["doctest_failures"],
            "doctests_total": gen["doctests_total"]})
        row = make_feedback_row15(rec, evidence)
        cov = feedback_coverage(row, tok)
        cov["rec_name"] = rec["name"]
        coverage.append(cov)
        old_row = make_feedback_row(rec, _VERDICT_STDERR[rec["name"]]
                                    or "unknown failure")
        old_cov = feedback_coverage(old_row, tok)
        old_cov["rec_name"] = rec["name"]
        old_style_coverage.append(old_cov)
        if cov["exact_match"]:
            ordered.append(rec["name"])

    model = _load_arm(STAGE15_REFERENCE_ARM)
    opt = torch.optim.AdamW(model.parameters(), lr=FB15_LR,
                            betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    losses = []
    steps_with_targets = 0
    supervised_tokens_total = 0
    t0 = time.time()
    torch.manual_seed(FB15_SEED)
    shuffled = []
    for rec, gen in zip(recs, generations):
        if gen["passed"] or rec["name"] not in ordered:
            continue
        evidence = compact_evidence({
            "passed": gen["passed"], "executed": gen["executed"],
            "exec_failure": "sandbox_exec_failed"
            if not gen["executed"] else None,
            "exec_stderr": gen["exec_stderr"],
            "doctest_failures": gen["doctest_failures"],
            "doctests_total": gen["doctests_total"]})
        shuffled.append(make_feedback_row15(rec, evidence))
    random.Random(FB15_SEED).shuffle(shuffled)
    training_ran = bool(shuffled)
    for i in range(FB15_STEPS if training_ran else 0):
        base = (i * FB15_BATCH) % len(shuffled)
        batch = shuffled[base:base + FB15_BATCH]
        if len(batch) < FB15_BATCH:
            batch = (batch + shuffled)[:FB15_BATCH]
        x, y = make_batch(batch, tok, SEQ)
        if x is None:
            continue
        n_sup = int((y != -100).sum())
        supervised_tokens_total += n_sup
        if n_sup == 0:
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
    fb_ckpt = CKPT10_DIR / "cortex-sft-stage15-feedback.pt"
    torch.save({
        "model_state": model.state_dict(), "arch": "CortexTransformer",
        "stage": 15, "kind": "compacted-execution-feedback",
        "steps": FB15_STEPS, "lr": FB15_LR, "batch": FB15_BATCH,
        "n_rows": len(ordered),
        "parent": str(CKPT10_DIR
                      / f"cortex-sft-{STAGE15_REFERENCE_ARM}.pt"),
        "parent_sha256": _sha256_file(CKPT10_DIR
                                      / f"cortex-sft-{STAGE15_REFERENCE_ARM}.pt")},
        fb_ckpt)
    model.eval()
    ev = _with_suite_pass(_capability_eval(model, tok))
    held_native = _heldout(model, heldout_recs, tok, "legacy")
    held_aligned = _heldout(model, heldout_recs, tok, "eval_aligned")
    stacc = structural_token_accuracy(model, tok, heldout_recs, SEQ)
    del model

    n_fit = len(ordered)
    all_match = bool(coverage) and all(c["exact_match"] for c in coverage)
    out = {
        "kind": "cortex-stage15-feedback-training-redesign",
        "owned_by": "Fibonera", "workstream": "C-feedback",
        "reference_arm_warm_start": STAGE15_REFERENCE_ARM,
        "sampled_records": len(recs),
        "benchmark_instruction_contamination_hits": contamination_hits,
        "candidate_outcomes": {
            "n": len(generations),
            "passed_reference_doctests": sum(
                g["passed"] for g in generations),
            "error_class_distribution": _dist(
                [g["error_class"] for g in generations]),
            "per_record": [{k: g[k] for k in (
                "rec_name", "executed", "doctests_total",
                "doctest_failures", "passed", "error_class")}
                for g in generations],
        },
        "coverage_audit": {
            "rows_built": len(coverage),
            "rows_fitting_seq": n_fit,
            "rows_with_exact_solution_supervision": sum(
                c["exact_match"] for c in coverage),
            "all_rows_supervise_the_full_solution": all_match,
            "stage14_style_rows_fitting_seq": sum(
                c["fits"] for c in old_style_coverage),
            "stage14_style_rows_built": len(old_style_coverage),
            "per_row": coverage,
        },
        "training": {
            "steps": FB15_STEPS,
            "target_bearing_steps": steps_with_targets,
            "stage14_negative_control_target_bearing_steps":
                STAGE14_CHAMPION_REF["feedback_target_bearing_steps"],
            "supervised_tokens_total": supervised_tokens_total,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "lr": FB15_LR, "batch": FB15_BATCH, "seed": FB15_SEED,
            "seconds": seconds, "checkpoint": str(fb_ckpt),
            "checkpoint_sha256": _sha256_file(fb_ckpt),
        },
        "evaluation": {
            "capability": ev,
            "heldout_native_final": held_native,
            "heldout_eval_aligned_final": held_aligned,
            "structural_token_accuracy": stacc,
            "heldout_delta_vs_stage14_master_champion": round(
                held_aligned - STAGE14_CHAMPION_REF[
                    "heldout_eval_aligned"], 4),
            "heldout_delta_vs_stage14_negative_feedback_control": round(
                held_aligned - STAGE14_CHAMPION_REF[
                    "feedback_arm_heldout_eval_aligned"], 4),
        },
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream D

def stage_d_execution_data(tok, train_recs: list) -> dict:
    header = _experiment_header15(
        "stage15-D-execution-verified-data",
        "multiple independently generated candidates per record produce "
        "classifiable positive and negative execution evidence with zero "
        "benchmark contamination; success only via real execution and "
        "reference-doctest verification",
        f"D_RECORDS={D_RECORDS}, D_SAMPLES={D_SAMPLES} independent "
        f"samples per record at temperatures {DECODING15['temperature']}, "
        "0.7, 1.0; every verdict carries real sandbox evidence; failures "
        "classified by the stage-15 execution-error taxonomy; provenance "
        "and contamination status recorded per candidate")
    recs = _sample_train_records(train_recs, D_RECORDS, GEN_SEED15)
    bench_instructions = {t.instruction for t in TASKS} | \
        {t.instruction for t in TASKS_V3}
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    candidates = []
    torch.manual_seed(GEN_SEED15)
    for rec in recs:
        for si, temp in enumerate((DECODING15["temperature"], 0.7, 1.0)):
            decoding = dict(DECODING15, temperature=temp)
            cand = extract_code(gen_fixed_decoding(
                _subject(tok, model), rec["instruction"],
                decoding=decoding)).strip()
            verdict = verify_candidate(cand, rec)
            candidates.append({
                "rec_name": rec["name"], "sample": si,
                "temperature": temp,
                "contaminated": rec["instruction"] in bench_instructions,
                "n_tokens": len(tok.encode(cand)) if cand else 0,
                **verdict,
                "error_class": classify_execution_failure(verdict)})
    del model
    n = len(candidates)
    per_record = {}
    for c in candidates:
        per_record.setdefault(c["rec_name"], []).append(c)
    out = {
        "kind": "cortex-stage15-execution-data",
        "owned_by": "Fibonera", "workstream": "D-execution-data",
        "reference_arm": STAGE15_REFERENCE_ARM,
        "n_candidates": n,
        "success_count": sum(c["passed"] for c in candidates),
        "error_class_distribution": _dist(
            [c["error_class"] for c in candidates]),
        "contaminated_candidates": sum(c["contaminated"]
                                       for c in candidates),
        "records_with_any_success": sum(
            any(c["passed"] for c in v) for v in per_record.values()),
        "per_record": [
            {"rec_name": k,
             "outcomes": [c["error_class"] for c in v],
             "any_passed": any(c["passed"] for c in v)}
            for k, v in sorted(per_record.items())],
        "candidates": candidates,
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream E

def _grade_unseen(task: UnseenTask, candidate: str) -> dict:
    """Grade an unseen-suite candidate by real sandbox pytest execution,
    reusing the existing grader on an equivalent immutable task."""
    from cortex.evaluation.tasks import EvalTask as RealTask

    rt = RealTask(task_id=task.task_id, instruction=task.instruction,
                  module_name=task.module_name, test_code=task.test_code)
    g = grade_single_quiet(rt, candidate)
    return {"task_id": task.task_id, "passed": bool(g["passed"]),
            "executed": g.get("exit_code") is not None,
            "stderr_tail": g.get("stderr_tail", "")[-300:]}


def _validate_unseen_harness() -> dict:
    """Harness sanity for the unseen suite: every authored reference
    solution must really pass its own tests (execution-validated)."""
    results = []
    for t in UNSEEN_TASKS_STAGE15:
        g = _grade_unseen(t, t.reference_solution)
        results.append({"task_id": t.task_id, "reference_passes":
                        bool(g["passed"])})
    return {"per_task": results,
            "all_references_pass": all(r["reference_passes"]
                                       for r in results)}


def _eval_capability_suite(model, tok) -> dict:
    """Fixed-decoding evaluation across immutable + unseen suites."""
    rows = []
    torch.manual_seed(GEN_SEED15)
    for suite, tasks, grader in (
            ("v1", TASKS, None),
            ("v3", TASKS_V3, None),
            ("unseen", UNSEEN_TASKS_STAGE15, _grade_unseen)):
        for t in tasks:
            cand = extract_code(gen_fixed_decoding(
                _subject(tok, model), t.instruction)).strip()
            m = structure_metrics(cand) if cand else {
                "parses": False, "complete_program": False}
            if grader is None:
                g = grade_single_quiet(t, cand)
                passed = bool(g["passed"])
                executed = g.get("exit_code") is not None
            else:
                gu = grader(t, cand)
                passed = gu["passed"]
                executed = gu["executed"]
            rows.append({"suite": suite, "task_id": t.task_id,
                         "nonempty": bool(cand), "parse": m["parses"],
                         "complete_program": m["complete_program"],
                         "executed": executed, "passed": passed})
    out = {"per_task": rows}
    for suite in ("v1", "v3", "unseen"):
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


def stage_e_generalization(tok) -> dict:
    header = _experiment_header15(
        "stage15-E-generalization-unseen-suite",
        "any capability appearing on train-sample tasks transfers to "
        "previously unseen tasks (new function names, classes, nested "
        "functions, exception handling, multi-function modules) and "
        "perturbed formulations",
        "8 authored unseen tasks with execution-validated reference "
        "solutions (harness sanity must pass first); suite fingerprint "
        f"recorded ({fingerprint_unseen()[:16]}...); tasks never exposed to "
        "training; perturbation probes at seed_base "
        f"{PERTURB_SEED_BASE15}; perturbation pass@k with 3 sampled "
        "candidates per unseen task at temperatures 0.2/0.7/1.0; "
        "promotion strictly by the pre-registered gates")
    harness = _validate_unseen_harness()
    if not harness.get("all_references_pass"):
        raise SystemExit("unseen-suite harness sanity failed: "
                         f"{harness}")
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    torch.manual_seed(GEN_SEED15)
    cap_full = _eval_capability_suite(model, tok)
    cap = {k: v for k, v in cap_full.items() if k != "per_task"}
    per_task_cap = cap_full["per_task"]
    suite = perturbation_suite_valid(TASKS + TASKS_V3,
                                     seed_base=PERTURB_SEED_BASE15)
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
        "kind": "cortex-stage15-generalization",
        "owned_by": "Fibonera", "workstream": "E-generalization",
        "unseen_suite_fingerprint": fingerprint_unseen(),
        "harness_sanity": harness,
        "capability": cap,
        "per_task": per_task_cap,
        "perturbation": {"n_probes": len(perturbed),
                         "pass_at_1": perturbed_pass,
                         "seed_base": PERTURB_SEED_BASE15,
                         "per_probe": perturbed,
                         "rejected": suite["rejected"]},
        "unseen_pass_at_k": _pass_at_k(tok),
    }
    return _finish(header, out, "EXECUTED")


def _pass_at_k(tok) -> dict:
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    per_task = []
    torch.manual_seed(GEN_SEED15)
    for t in UNSEEN_TASKS_STAGE15:
        passes = 0
        samples = []
        for si, temp in enumerate((DECODING15["temperature"], 0.7, 1.0)):
            decoding = dict(DECODING15, temperature=temp)
            cand = extract_code(gen_fixed_decoding(
                _subject(tok, model), t.instruction,
                decoding=decoding)).strip()
            g = _grade_unseen(t, cand)
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


# --------------------------------------------------------------- workstream F

def stage_f_repair(tok) -> dict:
    header = _experiment_header15(
        "stage15-F-self-correction-repair",
        "model-generated repair conditioned on the model's OWN compacted "
        "error evidence improves execution-verified success over direct "
        "generation on the same tasks; stage-14 was 0/15",
        "direct generation baseline first on v1 + unseen tasks; every "
        "failing candidate repaired by regeneration with its actual "
        "error evidence prepended (no external repair, no grammar "
        f"constraints); transitions measured independently; seed "
        f"{GEN_SEED15}; repair counts only with execution-verified pass "
        "on the original task")
    tasks = [("v1", t.task_id, t.instruction, t) for t in TASKS] + \
        [("unseen", t.task_id, t.instruction, t) for t in
         UNSEEN_TASKS_STAGE15]
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    rows = []
    torch.manual_seed(GEN_SEED15)
    for suite, tid, instruction, t in tasks:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), instruction)).strip()
        if suite == "unseen":
            g = _grade_unseen(t, cand)
        else:
            g0 = grade_single_quiet(t, cand)
            g = {"passed": bool(g0["passed"]),
                 "executed": g0.get("exit_code") is not None,
                 "stderr_tail": g0.get("stderr_tail", "")[-300:]}
        row = {"suite": suite, "task_id": tid,
               "direct_passed": g["passed"], "direct_executed":
               g["executed"], "direct_nonempty": bool(cand)}
        if not g["passed"] and cand:
            evidence = compact_evidence({
                "passed": False, "executed": g["executed"],
                "exec_failure": None, "exec_stderr": g["stderr_tail"],
                "doctest_failures": 0, "doctests_total": 0})
            rep_prompt = ("# Task: " + instruction
                          + "\n# Attempt failed with " + evidence + "\n")
            rcand = extract_code(gen_fixed_decoding(
                _subject(tok, model), rep_prompt)).strip()
            if suite == "unseen":
                rg = _grade_unseen(t, rcand)
            else:
                g0 = grade_single_quiet(t, rcand)
                rg = {"passed": bool(g0["passed"]),
                      "executed": g0.get("exit_code") is not None}
            row.update({
                "repair_evidence": evidence,
                "repair_changed_candidate": rcand != cand,
                "repair_executed": rg["executed"],
                "repair_passed": rg["passed"],
                "revised_nonempty": bool(rcand)})
        rows.append(row)
    del model
    n_repaired = sum(1 for r in rows if "repair_passed" in r)
    out = {
        "kind": "cortex-stage15-self-correction",
        "owned_by": "Fibonera", "workstream": "F-self-correction",
        "reference_arm": STAGE15_REFERENCE_ARM,
        "n_tasks": len(rows),
        "direct_successes": sum(r["direct_passed"] for r in rows),
        "n_repaired": n_repaired,
        "repair_attempts": n_repaired,
        "repair_successes": sum(1 for r in rows
                                if r.get("repair_passed")),
        "repair_changed_candidate_count": sum(
            bool(r.get("repair_changed_candidate")) for r in rows),
        "transitions": {
            "error_identified_with_real_evidence": n_repaired,
            "revised_candidate_executed": sum(
                bool(r.get("repair_executed")) for r in rows),
            "revised_candidate_passed": sum(
                bool(r.get("repair_passed")) for r in rows),
        },
        "per_task": rows,
    }
    out["repair_success_rate"] = round(
        out["repair_successes"] / max(n_repaired, 1), 4) if n_repaired \
        else None
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream G

def stage_g_optimization(tok, train_recs: list) -> dict:
    header = _experiment_header15(
        "stage15-G-optimization-audit",
        "control data, compacted execution-feedback data and structural "
        "weighting produce measurably different gradient signals; causal "
        "alignment and completion-only masking hold in every case",
        "gradient diagnostics measured with REAL backward passes on "
        "identical model states; no combined training arms; gradient "
        "norm, mean absolute gradient, and per-position masking "
        "invariants recorded per data condition")
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    conditions = {}

    def first_fitting(rows, k, probe_arg):
        out = []
        for row in rows:
            x, _y = make_batch([row], probe_arg, SEQ)
            if x is not None:
                out.append(row)
            if len(out) == k:
                break
        return out

    recs_ctrl = first_fitting(
        train_recs, 4, tok) + first_fitting(
        list(reversed(train_recs)), 4, tok) + first_fitting(
        train_recs[40:120], 4, tok)
    feedback_examples = first_fitting(
        [make_feedback_row15(rec, "syntax_failure: SyntaxError: "
                             "invalid syntax (line 2)")
         for rec in train_recs[8:120]], 4, tok)

    def measure(batch, tag, weighted):
        x, y = make_batch(batch, tok, SEQ)
        if x is None:
            conditions[tag] = {"skipped": True}
            return
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y, pad_mask=x == tok.pad_id)
        if weighted:
            logits, _ = model(x)
            loss = _weighted_ce(logits[:, :-1, :].contiguous(),
                                y[:, 1:].contiguous(), tok, 3.0)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm() for g in grads]))
        masked_ok = bool((y == -100).any())
        conditions[tag] = {
            "skipped": False, "loss": round(float(loss.detach()), 4),
            "grad_norm": round(float(norm), 4),
            "n_grad_tensors": len(grads),
            "mean_abs_grad": round(float(
                sum(float(g.abs().mean()) for g in grads) / len(grads)), 8),
            "completion_masking_present": masked_ok}
        model.zero_grad(set_to_none=True)

    measure(recs_ctrl[:4], "control_data_fixed_segmentation", False)
    measure([r for r in feedback_examples[:4]], "compacted_feedback_data",
            False)
    measure(recs_ctrl[:4], "structural_weighted_loss", True)

    out = {
        "kind": "cortex-stage15-optimization-audit",
        "owned_by": "Fibonera", "workstream": "G-optimization",
        "reference_arm": STAGE15_REFERENCE_ARM,
        "gradient_conditions": conditions,
        "alignment_note": "make_batch supervises completion tokens only; "
                          "prompt positions are -100",
    }
    return _finish(header, out, "EXECUTED")


# --------------------------------------------------------------- workstream H

def stage_h_capacity() -> dict:
    header = _experiment_header15(
        "stage15-H-capacity-and-virtual-accounting",
        "virtual coverage is recounted, GPU re-qualified, and every "
        "stage-15 checkpoint verified to restore into the reference "
        "architecture (recovery check)",
        "no parameter-count inflation; allocated/init/updated scalar "
        "classes separated; 2B gate result recorded; checkpoint recovery "
        "executes real torch.load + state_dict load")
    from cortex.hardware.profiler import can_train_locally, profile_hardware

    from cortex.model.configs import CORTEX_2B
    profile = profile_hardware()
    feasible, reason = can_train_locally(profile,
                                         CORTEX_2B.theoretical_param_count())
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan = persisted_optimizer_scan(auth)
    auth.close()
    recovery = []
    for name in ("control_rep2", "structural_rep2", "feedback"):
        if name == "feedback":
            ck_name = "stage15-feedback"
        else:
            ck_name = f"stage15-{name}"
        p = CKPT10_DIR / f"cortex-sft-{ck_name}.pt"
        entry = {"checkpoint": str(p), "exists": p.exists()}
        if p.exists():
            try:
                m = dense_model_from_warm6()
                ck = torch.load(p, map_location="cpu", weights_only=True)
                m.load_state_dict(ck["model_state"])
                entry["restores"] = True
                entry["steps"] = ck.get("steps")
                del m, ck
            except Exception as exc:
                entry["restores"] = False
                entry["error"] = repr(exc)
        recovery.append(entry)
    out = {
        "kind": "cortex-stage15-capacity-audit",
        "owned_by": "Fibonera", "workstream": "H-capacity",
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
        "checkpoint_recovery": recovery,
        "spend_usd": 0.0, "gpu_hours": 0.0,
    }
    return _finish(header, out, "BLOCKED")


# --------------------------------------------------------------- workstream I

def stage_i_subsystems(tok, train_recs: list) -> dict:
    header = _experiment_header15(
        "stage15-I-subsystem-contribution",
        "the existing MoE and divergent-reasoning subsystems are probed "
        "with controlled ablations; expert diversity and hypothesis "
        "diversity do not count as capability without execution-verified "
        "success",
        "moe_probe on the real MoE subject; divergent_eval with repair "
        "vs plain fixed-decoding generation on identical tasks; "
        "every success execution-verified")
    moe_sub = build_moe_subject()
    batches = [train_recs[i:i + 2]
               for i in range(0, min(len(train_recs), 24), 2)]
    mp = moe_probe(moe_sub.model, batches, tok, make_batch)
    del moe_sub

    plain = []
    model = _load_arm(STAGE15_REFERENCE_ARM)
    model.eval()
    torch.manual_seed(GEN_SEED15)
    for t in TASKS[:2]:
        cand = extract_code(gen_fixed_decoding(
            _subject(tok, model), t.instruction)).strip()
        g = grade_single_quiet(t, cand)
        plain.append({"task_id": t.task_id, "passed": bool(g["passed"]),
                      "executed": g.get("exit_code") is not None})
    del model

    dense = dense_model_from_warm6()
    dense.eval()
    torch.manual_seed(GEN_SEED15)
    div = divergent_eval_local(_subject(tok, dense), tasks=TASKS[:2],
                               n_hypotheses=3)
    del dense
    out = {
        "kind": "cortex-stage15-subsystem-contribution",
        "owned_by": "Fibonera", "workstream": "I-subsystems",
        "moe_probe": mp,
        "plain_generation_v1_first2": plain,
        "divergent_eval_v1_first2": div,
        "contribution_verdict": (
            "measured; any expert/hypothesis diversity without "
            "execution-verified success is not capability"),
    }
    return _finish(header, out, "EXECUTED")


def divergent_eval_local(subject, tasks, n_hypotheses: int = 3) -> dict:
    """Thin adapter over the existing divergent_eval (existing engine,
    real sandbox execution) so subsystem ablation stays in one artifact."""
    from cortex.evaluation.divergent import divergent_eval

    report = divergent_eval(subject, tasks=tasks,
                            n_hypotheses=n_hypotheses,
                            repair_round=True)
    if isinstance(report, dict):
        report = dict(report)
        report.pop("per_task", None)
    return report


# --------------------------------------------------------------- workstream J

def stage_j_promotion(a: dict, b: dict, c: dict, e: dict, f: dict) -> dict:
    header = _experiment_header15(
        "stage15-J-promotion-gates",
        "the six pre-registered stage-15 gates decide promotion; gates "
        "are evaluated from measured per-task rows only",
        "gate1: structural arm <= control in BOTH repetitions; gate2: "
        "champion syntax_valid_rate > 0 and complete_program_rate > 0 on "
        "immutable suites; gate3: unseen-suite pass@1 > 0 "
        "(execution-verified); gate4: perturbed pass@1 > 0; gate5: "
        "immutable v1/v3 pass@1 > 0 conclusively verified; gate6: >= 2 "
        "execution-verified successes across all suites; PROMOTED iff "
        "gate5 AND gate6 AND held-out <= 3.3015 in both styles; "
        "thresholds fixed before any stage-15 execution")
    struct = a["arms"]["structural_rep2"]
    cap = e["capability"]
    perturbed_pass = e["perturbation"]["pass_at_1"]
    unseen_pass = cap["unseen_pass_at_1"]
    v1_pass = cap["v1_pass_at_1"]
    v3_pass = cap["v3_pass_at_1"]
    total_successes = (
        (e["unseen_pass_at_k"]["tasks_with_at_least_one_pass"])
        + (f["direct_successes"]))
    gate1 = bool(a["findings"]["reproduces"])
    gate2 = (cap["v1_syntax_valid_rate"] > 0
             or cap["v3_syntax_valid_rate"] > 0) and (
        cap["v1_complete_program_rate"] > 0
        or cap["v3_complete_program_rate"] > 0)
    gate3 = unseen_pass > 0
    gate4 = bool(perturbed_pass) and perturbed_pass > 0
    gate5 = (v1_pass > 0 or v3_pass > 0)
    gate6 = total_successes >= 2
    heldout_ok = (struct["heldout_native_final"] <= G_HELDOUT_LIMIT
                  and struct["heldout_eval_aligned_final"]
                  <= G_HELDOUT_LIMIT)
    promoted = gate5 and gate6 and heldout_ok
    gates_passed = sum(1 for g in (gate1, gate2, gate3, gate4, gate5,
                                   gate6) if g)
    out = {
        "kind": "cortex-stage15-promotion-audit", "owned_by": "Fibonera",
        "workstream": "J-promotion",
        "reference_constants": {
            "stage14_champion": STAGE14_CHAMPION_REF,
            "heldout_limit": G_HELDOUT_LIMIT},
        "gates": {
            "gate1_reproducible_structural_improvement": {
                "ok": bool(gate1), "measured": a["findings"]},
            "gate2_complete_generation": {
                "ok": bool(gate2),
                "measured": {
                    "v1_syntax": cap["v1_syntax_valid_rate"],
                    "v3_syntax": cap["v3_syntax_valid_rate"],
                    "v1_complete": cap["v1_complete_program_rate"],
                    "v3_complete": cap["v3_complete_program_rate"]}},
            "gate3_execution_improvement": {
                "ok": bool(gate3), "measured": {"unseen_pass_at_1":
                                                unseen_pass}},
            "gate4_generalization": {
                "ok": bool(gate4), "measured": {
                    "perturbed_pass_at_1": perturbed_pass,
                    "n_probes": e["perturbation"]["n_probes"],
                    "seed_base": PERTURB_SEED_BASE15}},
            "gate5_independent_coding_success": {
                "ok": bool(gate5), "measured": {"v1_pass_at_1": v1_pass,
                                                "v3_pass_at_1": v3_pass}},
            "gate6_reliable_improvement": {
                "ok": bool(gate6), "measured": {
                    "tasks_with_at_least_one_pass_unseen":
                        e["unseen_pass_at_k"]["tasks_with_at_least_one_pass"],
                    "repair_plus_direct_successes": f[
                        "direct_successes"],
                    "total_counted": total_successes}},
        },
        "heldout_loss_gate": {"limit": G_HELDOUT_LIMIT,
                              "ok": bool(heldout_ok)},
        "champion_candidate": {
            "arm": STAGE15_REFERENCE_ARM,
            "checkpoint": str(CKPT10_DIR / f"cortex-sft-"
                              f"{STAGE15_REFERENCE_ARM}.pt"),
            "checkpoint_sha256": _sha256_file(
                CKPT10_DIR / f"cortex-sft-{STAGE15_REFERENCE_ARM}.pt")},
        "gates_passed": gates_passed,
        "promotion_decision": (
            "PROMOTED - reproducible execution-verified coding success on "
            "the immutable benchmark with held-out within the limit" if
            promoted else
            "NOT PROMOTED - immutable benchmark pass@1 remains 0.0 (or is "
            "not reproducible across tasks); negative result preserved; "
            "stage-14 structural arm retained as the best held-out "
            "configuration"),
        "rollback": {
            "baseline_checkpoint": str(CKPT10_DIR
                                       / f"cortex-sft-{CHAMPION_REF15}.pt"),
            "procedure": "restore the pre-declared stage-15 reference "
                         "arm (stage-14 structural); stage-15 "
                         "checkpoints are additive separate files",
        },
    }
    return _finish(header, out,
                   "PROMOTED" if promoted else "NOT PROMOTED")


CHAMPION_REF15 = "stage14-structural_h6000"


# --------------------------------------------------------------- main

def _resume(name: str, fn, *args):
    """Resume support: if the workstream artifact was persisted by a
    prior (possibly interrupted) process, reuse it; otherwise run."""
    p = ART / f"{name}.json"
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            if isinstance(doc, dict) and "experiment" in doc:
                print(f"[stage15] RESUME {name} from persisted artifact",
                      flush=True)
                return doc
        except Exception:
            pass
    return fn(*args)


def main() -> int:
    head = _git_head()
    assert head.startswith(PARENT_COMMIT), (
        f"stage-15 must start from {PARENT_COMMIT}, got {head}")
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    train_recs = load_split(str(DATASET), "train")
    heldout_recs = load_split(str(DATASET), "heldout")
    print(f"[stage15] dataset train={len(train_recs)} "
          f"heldout={len(heldout_recs)} unseen={len(UNSEEN_TASKS_STAGE15)}",
          flush=True)
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_pre = persisted_optimizer_scan(auth)
    auth.close()
    assert scan_pre["slots_updated"] == 35292, "authoritative store moved"

    a = _resume("stage15_reproducibility", stage_a_reproducibility,
                tok, train_recs, heldout_recs)
    _save_artifact("stage15_reproducibility.json", a)
    f = a["findings"]
    print(f"[stage15] A reproduces={f['reproduces']} "
          f"rep2_delta={f['rep2_delta_stage15_seed']}", flush=True)

    b = _resume("stage15_structural_consolidation", stage_b_structural,
                tok, heldout_recs)
    _save_artifact("stage15_structural_consolidation.json", b)
    bind_acc = b["teacher_forced"]["indentation_transition_accuracy"][
        "accuracy"]
    print(f"[stage15] B indent_transition_acc={bind_acc}", flush=True)

    c = _resume("stage15_feedback_training", stage_c_feedback,
                tok, train_recs, heldout_recs)
    _save_artifact("stage15_feedback_training.json", c)
    print(f"[stage15] C target_bearing={c['training']['target_bearing_steps']}/"
          f"{c['training']['stage14_negative_control_target_bearing_steps']}_stage14 "
          f"heldout={c['evaluation']['heldout_eval_aligned_final']}", flush=True)

    d = _resume("stage15_execution_data", stage_d_execution_data,
                tok, train_recs)
    _save_artifact("stage15_execution_data.json", d)
    print(f"[stage15] D successes={d['success_count']}/{d['n_candidates']}",
          flush=True)

    e = _resume("stage15_generalization", stage_e_generalization, tok)
    _save_artifact("stage15_generalization.json", e)
    print(f"[stage15] E unseen_pass@1={e['capability']['unseen_pass_at_1']} "
          f"perturbed_pass@1={e['perturbation']['pass_at_1']}", flush=True)

    f = _resume("stage15_self_correction", stage_f_repair, tok)
    _save_artifact("stage15_self_correction.json", f)
    print(f"[stage15] F repair={f['repair_successes']}/{f['repair_attempts']}",
          flush=True)

    g = stage_g_optimization(tok, train_recs)
    _save_artifact("stage15_optimization_audit.json", g)
    h = stage_h_capacity()
    _save_artifact("stage15_capacity_audit.json", h)
    i = stage_i_subsystems(tok, train_recs)
    _save_artifact("stage15_subsystem_contribution.json", i)
    print(f"[stage15] G/H/I complete (h={h['experiment']['decision']})",
          flush=True)

    j = stage_j_promotion(a, b, c, e, f)
    _save_artifact("stage15_promotion_audit.json", j)
    print(f"[stage15] J decision: {j['experiment']['decision']} "
          f"(gates passed {j['gates_passed']}/6)", flush=True)

    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_post = persisted_optimizer_scan(auth)
    auth.close()
    _save_artifact("stage15_report.json",
                   _report_payload(a, b, c, d, e, f, g, h, i, j, scan_post))
    _save_artifact("stage15_failure_taxonomy.json",
                   _taxonomy15(c, d))
    print("[stage15] complete", flush=True)
    return 0


def _report_payload(a, b, c, d, e, f, g, h, i, j, scan_post) -> dict:
    return {
        "kind": "cortex-stage15-report", "owned_by": "Fibonera",
        "stage": 15,
        "summary": {
            "parent_commit": PARENT_COMMIT,
            "A_reproduces": a["findings"]["reproduces"],
            "B_indent_transition_accuracy": b["teacher_forced"][
                "indentation_transition_accuracy"]["accuracy"],
            "B_free_running": b["free_running"],
            "C_target_bearing_steps": c["training"][
                "target_bearing_steps"],
            "C_heldout_eval_aligned": c["evaluation"][
                "heldout_eval_aligned_final"],
            "C_capability": {k: c["evaluation"]["capability"][k]
                             for k in ("v1_pass_at_1", "v3_pass_at_1",
                                       "v1_syntax_valid_rate",
                                       "v3_syntax_valid_rate")
                             if k in c["evaluation"]["capability"]},
            "D_success_count": d["success_count"],
            "E_v1": e["capability"]["v1_pass_at_1"],
            "E_v3": e["capability"]["v3_pass_at_1"],
            "E_unseen": e["capability"]["unseen_pass_at_1"],
            "E_perturbed_pass_at_1": e["perturbation"]["pass_at_1"],
            "E_unseen_pass_at_k": e["unseen_pass_at_k"],
            "F_repair_success_rate": f["repair_success_rate"],
            "G_gradient_conditions": g["gradient_conditions"],
            "H_two_b": h["two_b_gate"]["decision"],
            "H_store_untouched": h["virtual_store_recount"][
                "unchanged_35292"],
            "I_moe_available": "error" not in i["moe_probe"],
            "J_decision": j["experiment"]["decision"],
            "J_gates_passed": j["gates_passed"],
            "fingerprint_v1": FINGERPRINT_V1,
            "fingerprint_v2": FINGERPRINT_V2,
            "fingerprint_v3": fingerprint_tasks("v3"),
            "unseen_suite_fingerprint": fingerprint_unseen(),
            "spend_usd": 0.0, "gpu_hours": 0.0,
            "authoritative_store_untouched":
                scan_post["slots_updated"] == 35292,
        }}


def _taxonomy15(c: dict, d: dict) -> dict:
    class_dist = d["error_class_distribution"]
    return {
        "kind": "cortex-stage15-failure-taxonomy",
        "owned_by": "Fibonera", "workstream": "taxonomy",
        "definitions": {
            "success": "candidate executed and passed the required "
                       "verification (execution + reference doctests or "
                       "hidden tests)",
            "syntax_failure": "sandbox execution or test collection "
                              "failed with a Python syntax/indentation "
                              "error",
            "import_failure": "candidate module or import missing",
            "runtime_exception": "candidate executed but raised an "
                                 "unclassified exception",
            "failed_reference_doctest": "executed but reference-doctest "
                                        "examples failed",
            "failed_hidden_test": "executed but failed an assertion or "
                                  "hidden pytest check",
            "timeout": "sandbox or pytest execution timed out",
            "incomplete_generation": "empty or truncated candidate",
            "full_evidence_prompts": "RETIRED: stage-14 full-evidence "
                                     "prompts (long stderr) lost solution "
                                     "supervision; superseded by compact "
                                     "evidence (see coverage_audit)",
        },
        "measured_frequencies": {
            "execution_data_candidates": class_dist,
            "feedback_arm_candidates": c["candidate_outcomes"][
                "error_class_distribution"],
        },
        "stage14_context": {
            "feedback_coverage_defect": "47/48 rows lost solution "
                                        "supervision (102/600 target-"
                                        "bearing steps) due to evidence "
                                        "length; fixed by compaction with "
                                        "proven coverage audit",
        },
    }


def _finalize_from_artifacts() -> int:
    """Recover an interrupted run: complete workstreams re-execute only
    from persisted artifacts; the promotion audit and report rebuild."""
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    required = ("stage15_reproducibility", "stage15_structural_consolidation",
                "stage15_feedback_training", "stage15_execution_data",
                "stage15_generalization", "stage15_self_correction")
    saved = {}
    for name in required:
        p = ART / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"cannot finalize: {p} missing")
        saved[name] = json.loads(p.read_text())
    g = stage_g_optimization(tok, load_split(str(DATASET), "train"))
    _save_artifact("stage15_optimization_audit.json", g)
    h = stage_h_capacity()
    _save_artifact("stage15_capacity_audit.json", h)
    i = stage_i_subsystems(tok, load_split(str(DATASET), "train"))
    _save_artifact("stage15_subsystem_contribution.json", i)
    j = stage_j_promotion(saved["stage15_reproducibility"],
                          saved["stage15_structural_consolidation"],
                          saved["stage15_feedback_training"],
                          saved["stage15_generalization"],
                          saved["stage15_self_correction"])
    _save_artifact("stage15_promotion_audit.json", j)
    print(f"[stage15] J decision: {j['experiment']['decision']}", flush=True)
    auth = VirtualParameterStore.open(ART / "virtual-store-502b")
    scan_post = persisted_optimizer_scan(auth)
    auth.close()
    _save_artifact("stage15_report.json", _report_payload(
        saved["stage15_reproducibility"],
        saved["stage15_structural_consolidation"],
        saved["stage15_feedback_training"],
        saved["stage15_execution_data"],
        saved["stage15_generalization"],
        saved["stage15_self_correction"],
        g, h, i, j, scan_post))
    print("[stage15] report rebuilt from artifacts", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize-from-artifacts":
        raise SystemExit(_finalize_from_artifacts())
    raise SystemExit(main())
