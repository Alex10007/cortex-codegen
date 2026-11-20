"""Stage 13 tests: structural segmentation correctness (exact-once
completion coverage, prompt integrity, window validity), the fine
structural failure taxonomy, the stage-13 data/stream audit helpers and
post-run artifact integrity. Real components and the real tokenizer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
DATASET = ARTIFACTS / "sft" / "dataset_v3.jsonl"
SEQ = 256


@pytest.fixture(scope="module")
def tok():
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(ARTIFACTS / "tokenizer" / "tokenizer.json"))


@pytest.fixture(scope="module")
def train_recs():
    from cortex.training.sft import load_split

    return load_split(str(DATASET), "train")


def _long_record(tok):
    code = "\n".join(
        f"def helper_{i}(x):"
        f"\n    return x + {i}\n" for i in range(30))
    return {"name": "synthetic-long", "instruction":
            "define thirty helpers",
            "code": code}


# --------------------------------------- structural segmentation correctness

def test_structural_windows_cover_completion_exactly_once(tok):
    from cortex.training.longcontext import (chunk_record_structural,
                                             completion_coverage,
                                             record_ids)

    rec = _long_record(tok)
    wins = chunk_record_structural(rec, tok, SEQ)
    assert wins is not None and len(wins) >= 2
    prompt, completion = record_ids(rec, tok)
    cat = []
    for i, (win_ids, win_labels) in enumerate(wins):
        assert len(win_ids) == len(win_labels)
        cat += win_labels[1:]
        if i > 0:
            assert win_ids[0] == wins[i - 1][0][-1], \
                "window k must start at window k-1's final token"
    hit = [v for v in cat if v != -100]
    assert hit == completion, \
        "every completion token supervised exactly once, in order"


def test_structural_windows_fit_sequence_length(tok):
    from cortex.training.longcontext import chunk_record_structural

    rec = _long_record(tok)
    wins = chunk_record_structural(rec, tok, SEQ)
    for win_ids, _ in wins:
        assert len(win_ids) <= SEQ


def test_structural_windows_never_split_prompt_and_end_clean(tok):
    from cortex.evaluation.autoregressive_diagnostics import structure_scan
    from cortex.training.longcontext import (chunk_record_structural,
                                             record_ids)

    rec = _long_record(tok)
    prompt, completion = record_ids(rec, tok)
    wins = chunk_record_structural(rec, tok, SEQ)
    assert all(t in wins[0][0] for t in prompt), "prompt never split"
    for wi, _ in wins[:-1]:
        text = tok.decode(wi)
        assert text.endswith("\n")
        assert not structure_scan(text)["unbalanced"]
    assert len(wins) >= 2


def test_batch_segmentation_produces_valid_supervised_rows(tok):
    import torch

    from cortex.training.longcontext import make_batch_chunked

    rec = _long_record(tok)
    x_legacy, y_legacy = make_batch_chunked([rec], tok, SEQ)
    assert x_legacy is not None
    x_struct, y_struct = make_batch_chunked([rec], tok, SEQ,
                                            segmentation="structural")
    assert x_struct is not None
    assert x_struct.shape[1] == SEQ - 1
    assert y_struct.shape == x_struct.shape
    n_sup = int((y_struct != -100).sum())
    assert n_sup == len(tok.encode(rec["code"].strip() + "\n"
                                   + tok.EOT))


def test_completion_coverage_matches_fixed_on_real_train_split(
        tok, train_recs):
    from cortex.training.longcontext import completion_coverage

    cov_fixed = completion_coverage(train_recs, tok, SEQ)
    cov_struct = completion_coverage(train_recs, tok, SEQ,
                                     segmentation="structural")
    assert cov_fixed["complete"] is True
    assert cov_struct["complete"] is True
    assert cov_fixed["total_completion_targets"] == \
        cov_struct["total_completion_targets"]


def test_structural_cut_summary_counts_fallbacks(train_recs, tok):
    from cortex.training.longcontext import structural_cut_summary

    s = structural_cut_summary(train_recs, tok, SEQ)
    assert s["records_needing_cuts"] > 0
    assert s["clean_cuts"] >= 0
    assert s["clean_cuts"] + s["fallback_hard_cuts"] == s["cuts"]


def test_boundary_audit_supports_segmentation_param(train_recs, tok):
    from cortex.evaluation.structure_audit import chunk_boundary_audit

    fixed = chunk_boundary_audit(train_recs[:40], tok, SEQ)
    struct = chunk_boundary_audit(train_recs[:40], tok, SEQ,
                                  segmentation="structural")
    assert fixed["n_windows"] > 0 and struct["n_windows"] > 0
    assert 0.0 <= struct["pct_window_end_inside_open_construct"] <= 1.0


# ------------------------------------------------- taxonomy classification

def _mk_traj(text, steps=None, stopped="eot"):
    from cortex.evaluation.autoregressive_diagnostics import (
        first_invalid_token_incremental, lexical_state)

    prefixes = [text[:i] for i in range(0, len(text) + 1)]
    return {
        "first_invalid": first_invalid_token_incremental(prefixes),
        "text": text,
        "stopped": stopped,
        "completion_length_tokens": len(steps or []),
        "steps": steps if steps is not None else [
            {"step": i + 1, "lex_before": lexical_state(text[:i]),
             "token_text": text[i:i + 1], "token_id": i}
            for i in range(len(text))],
    }


def test_taxonomy_complete_program():
    from cortex.evaluation.autoregressive_diagnostics import \
        classify_structural_failure

    traj = _mk_traj("def f(x):\n    return x + 1\n")
    assert traj["first_invalid"]["failure_class"] == "valid_program"
    out = classify_structural_failure(traj)
    assert out["taxonomy"] == "complete_program"


def test_taxonomy_missing_colon():
    from cortex.evaluation.autoregressive_diagnostics import \
        classify_structural_failure

    out = classify_structural_failure(_mk_traj("def f()\n    return 1\n"))
    assert out["taxonomy"] == "missing_colon"
    assert out["evidence"]["parser_message"]


def test_taxonomy_premature_eot_open_construct():
    from cortex.evaluation.autoregressive_diagnostics import \
        classify_structural_failure

    out = classify_structural_failure(
        _mk_traj("def f(x):\n    y = (x", stopped="eot"))
    assert out["taxonomy"] == "premature_eot_open_construct"


def test_taxonomy_indent_change_after_newline_token():
    from cortex.evaluation.autoregressive_diagnostics import (
        classify_structural_failure, lexical_state)

    text = "def f():\n    if x:\n        return 1\n      return 2\n"
    lex_steps = []
    for i in range(len(text)):
        lex_steps.append({"step": i + 1, "lex_before":
                          lexical_state(text[:i]),
                          "token_text": text[i:i + 1], "token_id":
                          i})
    out = classify_structural_failure(_mk_traj(text, steps=lex_steps))
    assert out["taxonomy"] in ("incorrect_indent_decrease",
                               "unexpected_indentation_transition")
    assert out["evidence"]["token_line_indent"] is not None


def test_taxonomy_truncation_no_eot():
    from cortex.evaluation.autoregressive_diagnostics import \
        classify_structural_failure

    out = classify_structural_failure(
        _mk_traj("def f(x):\n    y = (x", stopped="max_new_tokens"))
    assert out["taxonomy"] == "truncated_generation_max_new_tokens"


def test_taxonomy_unexpected_indent():
    from cortex.evaluation.autoregressive_diagnostics import (
        classify_structural_failure, lexical_state)

    text = "x = 1\n    y = 2\n"
    lex_steps = [{"step": i + 1, "lex_before": lexical_state(text[:i]),
                  "token_text": text[i:i + 1], "token_id": i}
                 for i in range(len(text))]
    out = classify_structural_failure(_mk_traj(text, steps=lex_steps))
    assert out["taxonomy"] in ("unexpected_indentation",
                               "unexpected_indentation_transition")


def test_taxonomy_unterminated_string_is_open_construct():
    from cortex.evaluation.autoregressive_diagnostics import (
        classify_structural_failure, lexical_state)

    text = "def f():\n    s = 'abc\n    return s\n"
    lex_steps = [{"step": i + 1, "lex_before": lexical_state(text[:i]),
                  "token_text": text[i:i + 1], "token_id": i}
                 for i in range(len(text))]
    out = classify_structural_failure(_mk_traj(text, steps=lex_steps))
    assert out["taxonomy"] == "premature_eot_open_construct"


def test_taxonomy_unclosed_delimiter_direct():
    from cortex.evaluation.autoregressive_diagnostics import (
        classify_structural_failure, lexical_state)

    text = "def f():\n    return 1)\n"
    lex_steps = [{"step": i + 1, "lex_before": lexical_state(text[:i]),
                  "token_text": text[i:i + 1], "token_id": i}
                 for i in range(len(text))]
    out = classify_structural_failure(_mk_traj(text, steps=lex_steps))
    assert out["taxonomy"] == "missing_opening_delimiter"


# ------------------------------------------------- artifact integrity

STAGE13_EXPERIMENT_ARTIFACTS = [
    "stage13_structural_root_cause.json",
    "stage13_structural_data_audit.json",
    "stage13_training_stream.json",
    "stage13_structural_learning.json",
    "stage13_generation_control.json",
    "stage13_execution_feedback.json",
    "stage13_generalization.json",
    "stage13_experiment_matrix.json",
    "stage13_capacity_and_infrastructure.json",
    "stage13_intelligence_subsystems.json",
    "stage13_promotion_audit.json",
]

STAGE13_PLAIN_ARTIFACTS = [
    "stage13_independent_audit.json",
    "stage13_report.json",
    "stage13_failure_taxonomy.json",
]


@pytest.mark.parametrize("name", STAGE13_EXPERIMENT_ARTIFACTS)
def test_stage13_artifacts_owned_and_complete(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
    assert "experiment" in obj and "decision" in obj["experiment"]


@pytest.mark.parametrize("name", STAGE13_PLAIN_ARTIFACTS)
def test_stage13_plain_artifacts_owned(name):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip("artifact not yet produced")
    import json as _json

    obj = _json.loads(path.read_text())
    assert obj.get("owned_by") == "Fibonera"
