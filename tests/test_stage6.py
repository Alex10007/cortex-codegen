"""Stage 6: long-sequence chunking, FSDP sharded training mechanics,
2B launcher hardware gate, monitoring metrics, fingerprint stability."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cortex.evaluation import benchmarks
from cortex.model.configs import get_config
from cortex.model.transformer import CortexTransformer
from cortex.training.fsdp import spawn_fsdp_validation
from cortex.training.longcontext import (
    chunk_record,
    completion_coverage,
    dataset_length_stats,
    make_batch_chunked,
    record_ids,
)
from cortex.training.scaling import cluster_plans
from cortex.training.sft import fine_tune, make_batch
from cortex.tokenizer.cortex_tokenizer import load_tokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

LONG_CODE = "\n".join(
    f"    acc_{i} = value + {i}\n    if acc_{i} > {i}:\n        acc_{i} -= 1\n"
    for i in range(12)) + "\n    return acc_0\n"


def _long_rec():
    return {"name": "longfn", "module": "m", "kind": "function",
            "instruction": "Implement longfn that transforms a value.\n",
            "code": "def longfn(value):\n" + LONG_CODE}


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tokstage6")
    pool = ROOT / "cortex" / "learning"
    (tmp / "stdlib_sample.py").write_text(
        (pool / "reference_algorithms.py").read_text())
    train_tokenizer([str(tmp)], str(tmp), vocab_size=512)
    return tmp


@pytest.fixture(scope="module")
def tok(tok_dir):
    return load_tokenizer(str(tok_dir / "tokenizer.json"))


def test_chunk_record_windows_reconstruct_the_stream_exactly(tok):
    rec = _long_rec()
    prompt, completion = record_ids(rec, tok)
    ids = prompt + completion
    labels = [-100] * len(prompt) + completion[:]
    windows = chunk_record(rec, tok, 256)
    assert windows is not None and len(windows) >= 2
    assert all(len(win_ids) <= 256 for win_ids, _ in windows)
    cat_inputs: list[int] = []
    cat_targets: list[int] = []
    for win_ids, win_labels in windows:
        assert len(win_ids) == len(win_labels)
        cat_inputs += list(win_ids[:-1])
        cat_targets += list(win_labels[1:])
    assert cat_inputs == ids[:-1]
    assert cat_targets == labels[1:]
    assert all(v == -100 for v in cat_targets[:len(prompt) - 1])
    assert sum(1 for v in cat_targets if v != -100) == len(completion)


def test_completion_coverage_helper_reports_complete(tok):
    cov = completion_coverage([_long_rec(), tok_short_rec()], tok, 256)
    assert cov["complete"] is True
    assert cov["total_completion_targets"] > 0


def tok_short_rec():
    return {"name": "short", "module": "m", "kind": "function",
            "instruction": "Add two.\n",
            "code": "def add2(a, b):\n    return a + b\n"}


def test_chunked_batch_recovers_legacy_dropped_record(tok):
    rec = _long_rec()
    assert make_batch([rec], tok, 256)[0] is None
    x, y = make_batch_chunked([rec], tok, 256)
    assert x is not None
    assert int((y != -100).sum()) > 0
    assert x.shape[0] >= 2 and x.shape[1] == 255


def test_chunked_single_window_matches_legacy_batch(tok):
    x1, y1 = make_batch([tok_short_rec()], tok, 256)
    x2, y2 = make_batch_chunked([tok_short_rec()], tok, 256)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_real_training_step_on_chunked_batch(tok):
    torch.manual_seed(4242)
    model = CortexTransformer(get_config("cortex-demo"))
    x, y = make_batch_chunked([_long_rec()], tok, 256)
    _, loss = model(x, y, pad_mask=x == tok.pad_id)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters()
             if p.grad is not None]
    assert grads and sum(grads) > 0


def test_instruction_alone_too_long_is_counted_not_truncated(tok):
    rec = {"name": "huge", "module": "m", "kind": "function",
           "instruction": ("Implement it. " * 400), "code": "def z():\n    return 1\n"}
    assert len(tok.encode(rec["instruction"])) > 255
    assert chunk_record(rec, tok, 256) is None
    assert make_batch_chunked([rec], tok, 256)[0] is None


def test_dataset_length_stats_measured():
    tok = load_tokenizer(str(ART / "tokenizer" / "tokenizer.json"))
    s256 = dataset_length_stats(str(ART / "sft" / "dataset_v3.jsonl"), tok,
                                256)["splits"]["train"]
    assert s256["records"] == 241
    assert s256["fit_unmodified"] == 74
    assert s256["recovered_by_chunking"] == 111
    assert s256["legacy_drop_rate"] == pytest.approx(0.6929)
    assert s256["remaining_exclusion_rate"] == pytest.approx(0.2324)
    assert s256["trainable_with_chunking"] == 185
    s8192 = dataset_length_stats(str(ART / "sft" / "dataset_v3.jsonl"), tok,
                                 8192)["splits"]["train"]
    assert s8192["remaining_exclusion_rate"] == 0.0
    assert s8192["trainable_with_chunking"] == 241


def test_fine_tune_chunked_records_grad_norm_and_throughput(tok_dir, tmp_path):
    jsonl = tmp_path / "ds.jsonl"
    rows = [
        {**_long_rec(), "split": "train", "doctests_attempted": 1,
         "provenance": {"source_uri": "cortex://t", "license":
                        "project-local", "version": "s6"}},
        {**tok_short_rec(), "split": "heldout", "doctests_attempted": 1,
         "provenance": {"source_uri": "cortex://t", "license":
                        "project-local", "version": "s6"}},
    ]
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = fine_tune("cortex-demo", str(tok_dir / "tokenizer.json"),
                        str(jsonl), None, str(tmp_path / "out"), epochs=1,
                        batch_size=1, log_every=1, chunk_long_records=True)
    metrics_rows = [json.loads(l) for l in
                    open(summary["metrics"], encoding="utf-8") if l.strip()]
    assert metrics_rows
    assert all("grad_norm" in r and "tokens_per_s" in r
               and "batch_rows" in r for r in metrics_rows)
    assert all(np.isfinite(r["grad_norm"]) for r in metrics_rows)
    assert summary["param_count"] == 8071296


def test_fsdp_validation_real_processes(tmp_path):
    ids = np.random.default_rng(1).integers(256, 500, size=2000).astype(np.int32)
    np.save(tmp_path / "tokens.npy", ids)
    res = spawn_fsdp_validation("cortex-demo", str(tmp_path / "tokens.npy"),
                                str(tmp_path / "out"), world_size=2,
                                steps=4, checkpoint_every=2, resume_steps=1)
    assert res["backend"] == "gloo"
    assert res["sharding_verified"]
    assert res["max_checksum_spread_across_ranks"] == pytest.approx(0.0)
    assert res["full_state_reloaded_into_plain_model"]
    assert res["resume"]["resumed_ok"]
    assert res["resume"]["resumed_from_step"] == 4
    assert res["resume"]["losses_finite"]


def test_cluster_matrix_2b_world8_fits_40gb():
    plans = cluster_plans(get_config("cortex-2b"), batch_size=8)
    assert plans["params"] == 2056012800
    w8 = next(p for p in plans["plans"] if p["world_size"] == 8)
    assert w8["fits_40gb_gpu"]
    w2 = next(p for p in plans["plans"] if p["world_size"] == 2)
    assert not w2["fits_40gb_gpu"] and not w2["fits_80gb_gpu"]


def test_2b_launcher_refuses_without_gpus():
    out = tempfile.mkdtemp()
    res = subprocess.run(
        [sys.executable, "scripts/train_2b_fsdp.py", "--config", "cortex-2b",
         "--steps", "1", "--out-dir", out],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert res.returncode == 2
    assert "REFUSED-HARDWARE-GATE" in res.stdout
    assert "2056012800" in res.stdout


def test_benchmark_integrity_unchanged():
    intg = benchmarks.check_integrity()
    assert intg["status"] == "unmodified"
    assert intg["fingerprint"] == (
        "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177")
