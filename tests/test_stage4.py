"""Stage 4: label-shift correctness, EOT stopping, 2B verification, scaling,
multi-file verified examples, dataset v3 immutability."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cortex.evaluation import benchmarks
from cortex.learning.multifile_examples import build_multifile_pairs
from cortex.model.configs import CORTEX_2B, CORTEX_DEMO, get_config
from cortex.model.transformer import CortexTransformer
from cortex.training.dataset_v2 import build_dataset_v2, build_dataset_v3
from cortex.training.scaling import (
    estimate_memory_bytes,
    hardware_profile,
    supported_precision,
    training_plan,
    validate_scalable_path,
)
from cortex.training.sft import make_batch
from cortex.tokenizer.cortex_tokenizer import train_tokenizer


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tokstage4")
    pool = Path(__file__).resolve().parents[1] / "cortex" / "learning"
    merged = tmp / "stdlib_sample.py"
    merged.write_text("\n\n".join((pool / name).read_text() for name in
                                  ("reference_algorithms.py",
                                   "doctest_harvester.py",
                                   "multifile_examples.py")))
    train_tokenizer([str(tmp)], str(tmp), vocab_size=512)
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(tmp / "tokenizer.json"))


def test_make_batch_labels_are_shifted_next_tokens(tok):
    rec = {"instruction": "Write a function that adds two numbers.\n",
           "code": "def add2(a, b):\n    return a + b\n"}
    x, y = make_batch([rec], tok, 256)
    full = tok.encode(rec["instruction"] + "\n")
    prompt_len = len(full)
    assert x.shape == y.shape
    pad = tok.pad_id
    first_completion_pos = prompt_len - 1
    assert int(x[0, first_completion_pos].item()) == full[-1]
    assert int(y[0, first_completion_pos].item()) == \
        tok.encode(rec["code"].strip())[0]
    for t in range(first_completion_pos):
        assert int(y[0, t].item()) == -100
    completion = tok.encode(rec["code"].strip() + "\n" + tok.EOT)
    non_pad = (x[0] != pad).nonzero().flatten()
    last_real = int(non_pad[-1])
    assert last_real == prompt_len - 1 + len(completion)
    assert int(y[0, last_real - 1].item()) == tok.eot_id
    assert int(y[0, last_real].item()) == -100


def test_generate_stops_at_eot():
    model = CortexTransformer(get_config("cortex-demo"))
    eot = 17
    V = model.cfg.vocab_size

    def stub_forward(self, tokens, targets=None):
        B, T = tokens.shape
        logits = torch.full((B, T, V), -1e4)
        logits[:, :, 5] = 5.0
        logits[:, :, eot] = 6.0
        return logits, None

    model.forward = stub_forward.__get__(model)
    x = torch.zeros(1, 4, dtype=torch.long)
    out = model.generate(x.clone(), 10, temperature=0.01, top_k=1, eot_id=eot)
    assert out.shape[1] == 5
    assert int(out[0, -1].item()) == eot
    out2 = model.generate(x.clone(), 10, temperature=0.01, top_k=1)
    assert out2.shape[1] == 14


def test_2b_parameter_count_exact():
    theoretical = CORTEX_2B.theoretical_param_count()
    with torch.device("meta"):
        model = CortexTransformer(CORTEX_2B)
    actual = model.actual_param_count()
    assert actual == theoretical
    assert theoretical >= 2_000_000_000


def test_demo_param_count_exact():
    model = CortexTransformer(CORTEX_DEMO)
    assert model.actual_param_count() == CORTEX_DEMO.theoretical_param_count()


def test_hardware_profile_measured():
    prof = hardware_profile()
    assert prof["cores"] and prof["cores"] > 0
    assert prof["ram_gb"] > 0
    assert "available" in prof["cuda"]


def test_2b_training_plan_refuses_to_claim_trainability():
    plan = training_plan(CORTEX_2B, batch_size=8)
    mem = estimate_memory_bytes(CORTEX_2B, 8, CORTEX_2B.max_seq_len)
    assert mem["params"] >= 2_000_000_000
    assert mem["total_gb_est"] > 0
    assert plan["config"] == "cortex-2b"
    assert ("FSDP" in plan["strategy"]) or ("multi-GPU" in plan["strategy"])
    assert plan["host_fits_full_state"] is False or \
        plan["host"]["ram_gb"] > 200


def test_scalable_path_real_training_and_resume_roundtrip(tmp_path):
    # real tokens for a real short run
    ids = np.random.default_rng(0).integers(256, 500, size=4000).astype(np.int32)
    bin_path = tmp_path / "tokens.npy"
    np.save(bin_path, ids)
    tmp_tok = tmp_path / "corpus"
    tmp_tok.mkdir()
    (tmp_tok / "c.py").write_text("def f(a):\n    return a\n")
    train_tokenizer([str(tmp_tok)], str(tmp_path), vocab_size=600)
    res = validate_scalable_path(
        "cortex-demo", str(tmp_path / "t.json"), str(bin_path),
        str(tmp_path / "out"), steps=3, batch_size=2)
    assert res["losses_finite"]
    assert res["params"] == CORTEX_DEMO.theoretical_param_count()
    assert res["checkpoint_saved_and_reloaded"]


def test_multifile_examples_verified():
    pairs, log = build_multifile_pairs()
    assert len(pairs) >= 3
    for name, verdicts in log.items():
        for target, v in verdicts.items():
            assert v.get("ok"), f"{name}/{target} not verified: {v}"
    for p in pairs:
        assert p["kind"] == "multi-file"
        assert p["code"].startswith("```")
        assert p["provenance"]["license"] == "project-local"


def test_dataset_v3_keeps_heldout_and_val_immutable(tmp_path):
    from cortex.training.sft import build_dataset

    pairs = [
        {"name": f"f{i}", "module": "m", "kind": "function",
         "instruction": f"Implement the Python function func{i}.\n",
         "code": f"def func{i}(x):\n    return {i}\n",
         "doctests_attempted": 1,
         "provenance": {"source_uri": "file:///stdlib", "license":
                        "Python-2.0 / PSF", "version": "cpython-3.12"}}
        for i in range(30)]
    v2_jsonl = str(tmp_path / "v2.jsonl")
    build_dataset_v2(pairs, v2_jsonl, str(tmp_path / "v2_manifest.json"))
    extra = [{"name": "repo::extra", "module": "m", "kind": "multi-file",
              "instruction": "Create a package.\n",
              "code": "```x.py\ndef z():\n    return 1\n```\n",
              "doctests_attempted": 2,
              "provenance": {"source_uri": "cortex://x", "license":
                             "project-local", "version": "s4"}}]
    v3 = build_dataset_v3(v2_jsonl, extra, str(tmp_path / "v3.jsonl"),
                          str(tmp_path / "v3_manifest.json"))
    v2m = json.loads(Path(tmp_path / "v2_manifest.json").read_text())
    assert v3["fingerprints"]["heldout"] == v2m["fingerprints"]["heldout"]
    assert v3["fingerprints"]["val"] == v2m["fingerprints"]["val"]
    assert v3["counts"]["train"] - v2m["counts"]["train"] == 1
    assert Path(tmp_path / "v3_manifest.json").exists()


def test_referenced_heldout_load_guard_uses_v3_manifest(tmp_path):
    from cortex.training.dataset_v2 import load_split_v2

    pairs = [
        {"name": f"f{i}", "module": "m", "kind": "function",
         "instruction": f"Implement the Python function func{i}.\n",
         "code": f"def func{i}(x):\n    return {i}\n",
         "doctests_attempted": 1,
         "provenance": {"source_uri": "file:///stdlib", "license":
                        "Python-2.0 / PSF", "version": "cpython-3.12"}}
        for i in range(20)]
    build_dataset_v2(pairs, str(tmp_path / "v2.jsonl"),
                     str(tmp_path / "m2.json"))
    recs = load_split_v2(str(tmp_path / "v2.jsonl"), "heldout",
                         str(tmp_path / "m2.json"))
    assert recs
    bad = json.loads(Path(tmp_path / "m2.json").read_text())
    bad["fingerprints"]["heldout"] = "0" * 64
    Path(tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        load_split_v2(str(tmp_path / "v2.jsonl"), "heldout",
                      str(tmp_path / "bad.json"))


def test_benchmark_integrity_unchanged():
    intg = benchmarks.check_integrity()
    assert intg["status"] == "unmodified"
    assert intg["fingerprint"] == (
        "74cc760709f5cec015849251f932498f0628aee281284886515eabea9c8cc177")


def test_supported_precision_reports_real_mode():
    mode = supported_precision()
    assert mode in ("bf16-gpu", "bf16-cpu", "fp32")
