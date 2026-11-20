"""Stage 12 tests: autoregressive diagnostics correctness, prompt-style
extensions to the training stream, structural complete-window filtering,
generalization probes and post-run artifact/promotion integrity. Real
components; the generation-equivalence check runs the actual model.
"""
from __future__ import annotations

import json

import pytest

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
DATASET = ARTIFACTS / "sft" / "dataset_v3.jsonl"
SEQ = 256


@pytest.fixture(scope="module")
def tok():
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(ARTIFACTS / "tokenizer" / "tokenizer.json"))


@pytest.fixture(scope="module")
def data():
    from cortex.training.sft import load_split

    return (load_split(str(DATASET), "train"),
            load_split(str(DATASET), "heldout"))


# ------------------------------------------------- analyzer correctness

def test_analyzer_selftest_passes():
    from cortex.evaluation.autoregressive_diagnostics import \
        analyzer_selftest

    st = analyzer_selftest()
    assert st["self_test_ok"] is True, st


def test_first_invalid_incremental_persistence_rule():
    from cortex.evaluation.autoregressive_diagnostics import \
        first_invalid_token_incremental

    text = "def f(:\n    return 1\n"
    prefixes = [text[:i] for i in range(1, len(text) + 1)]
    row = first_invalid_token_incremental(prefixes)
    assert row["first_invalid_index"] is not None
    assert row["failure_class"] == "structural_inconsistency"
    assert row["directly_introduced_by_newest_token"] is False
    ok_text = "def f():\n    return 1"
    ok_prefixes = [ok_text[:i] for i in range(1, len(ok_text) + 1)]
    row_ok = first_invalid_token_incremental(ok_prefixes)
    assert row_ok["first_invalid_index"] is None


# ------------------------------------------------- generation equivalence

def test_trajectory_instrumentation_matches_raw_forward_math(tok):
    """Deterministic equivalence: run a greedy rollout and require the
    trajectory's per-step top-5 (ids + probs) to match an independent
    full-forward replication bit-for-bit. Sampled-token equality against
    model.generate is deliberately NOT asserted: CPU torch.multinomial
    draws were observed to differ across process launches under the same
    seed (within-process repeats are stable), so a sampled-sequence gate
    would be environment-flaky rather than a correctness property."""
    import torch
    import torch.nn.functional as F

    from cortex.evaluation.autoregressive_diagnostics import \
        trajectory_generate
    from scripts.stage10_run import dense_model_from_warm6

    model = dense_model_from_warm6()
    model.eval()
    dec = {"temperature": 0.05, "top_k": 50, "max_new_tokens": 5}
    prompt = tok.decode([tok.eot_id] + tok.encode(
        "# Task: return the number 7\n"))
    torch.manual_seed(7)
    traj = trajectory_generate(model, tok, prompt, dec)
    ids = traj["prompt_token_ids"]
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for i, step in enumerate(traj["steps"][:3]):
            logits, _ = model(tokens[:, -model.cfg.max_seq_len:])
            last = logits[0, -1, :] / max(dec["temperature"], 1e-6)
            kth = torch.topk(last, min(dec["top_k"],
                                       last.size(-1)))[0][-1:]
            masked = last.masked_fill(last < kth, float("-inf"))
            probs = F.softmax(masked, dim=-1)
            top = torch.topk(probs, k=5)
            ref5 = [{"id": int(t), "p": round(float(p), 6)}
                    for t, p in zip(top.indices, top.values)]
            assert step["top5"] == ref5, (i, step["top5"], ref5)
            tokens = torch.cat(
                [tokens, torch.tensor([[ref5[0]["id"]]])], dim=1)


# ------------------------------------------------- prompt-style extension

def test_eval_aligned_record_ids_match_eval_prompt(tok, data):
    train_recs, _ = data
    from cortex.training.longcontext import eval_prompt_text, record_ids

    for prompt_style in ("legacy", "eval_aligned"):
        for rec in train_recs[:5]:
            prompt, completion = record_ids(rec, tok, prompt_style)
            if prompt_style == "eval_aligned":
                expected = tok.encode(eval_prompt_text(rec, tok))
                assert prompt == expected
            else:
                assert prompt == tok.encode(rec["instruction"] + "\n")
            assert completion == tok.encode(
                rec["code"].strip() + "\n" + tok.EOT)
    with pytest.raises(ValueError):
        record_ids(train_recs[0], tok, "bogus_style")


def test_make_batch_eot_never_precedes_supervised_target(data, tok):
    from cortex.training.sft import make_batch

    train_recs, _ = data
    x_legacy, y_legacy = make_batch(train_recs[:32], tok, SEQ)
    x_aligned, y_aligned = make_batch(train_recs[:32], tok, SEQ,
                                      prompt_style="eval_aligned")
    assert x_legacy is not None and x_aligned is not None

    def eot_before_first_supervised(x, y):
        total = 0
        for row_x, row_y in zip(x, y):
            sup = (row_y != -100).nonzero()
            if sup.numel() == 0:
                continue
            total += int(
                ((row_x == tok.eot_id)[:int(sup[0, 0])]).sum())
        return total

    assert eot_before_first_supervised(x_legacy, y_legacy) == 0, \
        "in legacy rows EOT must never precede a supervised target"
    assert eot_before_first_supervised(x_aligned, y_aligned) > 0, \
        "eval-aligned rows must condition supervised targets on EOT, " \
        "matching benchmark prompt surfaces"


# ------------------------------------------------- complete-window filter

def test_complete_rows_only_removes_unbalanced_windows(data, tok):
    from cortex.evaluation.structure_audit import structure_scan
    from cortex.training.longcontext import make_batch_chunked

    train_recs, _ = data
    x_all, _y = make_batch_chunked(train_recs[:8], tok, SEQ, "legacy",
                                   complete_rows_only=False)
    x_clean, _ = make_batch_chunked(train_recs[:8], tok, SEQ, "legacy",
                                    complete_rows_only=True)
    assert x_clean is not None
    assert x_clean.shape[0] <= x_all.shape[0]
    for row in x_clean:
        window_text = tok.decode(row.tolist())
        assert structure_scan(window_text)["unbalanced"] is False, \
            "kept windows must end at clean structural boundaries"


# ------------------------------------------------- generalization probes

def test_dataset_overlap_deterministic(data):
    from cortex.evaluation.generalization import dataset_overlap

    all_recs = data[0] + data[1]
    assert dataset_overlap(all_recs) == dataset_overlap(all_recs)


def test_perturbation_probes_rename_consistently():
    from cortex.evaluation.generalization import perturbation_suite_valid
    from cortex.evaluation.tasks import TASKS

    probes = perturbation_suite_valid(TASKS[:3], seed_base=20261200)
    assert probes["n_valid_probes"] >= 1
    names = [pt.task_id for pt in probes["probes"]]
    assert len(set(names)) == len(names)
    renames = [t for t in probes["probes"]
               if t.task_id.endswith("-perturbed")]
    assert len(renames) == len(probes["probes"]), \
        "every probe must be renamed away from the immutable original"


# ------------------------------------------------- post-run artifacts

def _load(name: str):
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text())


def test_stage12_diagnostics_artifact_integrity():
    a = _load("stage12_autoregressive_diagnostics.json")
    assert a["owned_by"] == "Fibonera"
    assert a["analyzer_self_test_ok"] is True
    for subj in a["per_subject"].values():
        assert subj["n_tasks"] == 15
        from collections import Counter
        suites = Counter(r["suite"] for r in subj["per_task"])
        assert suites == {"v1": 3, "v3": 12}
        assert 0.0 <= subj["parse_rate"] <= 1.0
        assert 0.0 <= subj["pass_at_1"] <= 1.0
    assert a["eot_in_training_context"]["legacy"][
        "eot_in_context_total"] == 0
    assert a["eot_in_training_context"]["eval_aligned"][
        "eot_in_context_total"] > 0
    assert "never contain EOT before a supervised target" in a[
        "eot_claim"]


def test_stage12_capability_and_promotion_rule():
    f = _load("stage12_capability_evaluation.json")
    h = _load("stage12_promotion_audit.json")
    assert f["owned_by"] == "Fibonera" and h["owned_by"] == "Fibonera"
    any_pass = any(r["pass_at_1"] > 0.0 for r in f["results"].values())
    if any_pass:
        assert f["promotion_decision"].startswith("PROMOTED")
        assert h["promotion_decision"] == "PROMOTED"
    else:
        assert "NOT PROMOTED" in f["promotion_decision"]
        assert "NOT PROMOTED" in h["promotion_decision"]
    d = _load("stage12_training_experiments.json")
    assert d["pre_registered_rule"] is not None
    assert set(d["arms"]) == {"chunked_control", "eval_aligned",
                              "complete_windows",
                              "eval_aligned_complete"}


def test_stage12_gates_and_capacity():
    g = _load("stage12_capacity_readiness.json")
    assert g["experiment"]["decision"] == "BLOCKED"
    assert g["architectures"]["cortex_2b_count_is_2056012800"] is True
    r = _load("stage12_report.json")
    assert r["owned_by"] == "Fibonera"
    assert r["summary"]["spend_usd"] == 0.0
    assert r["summary"]["gpu_hours"] == 0.0
    assert r["summary"]["authoritative_store_untouched"] is True
