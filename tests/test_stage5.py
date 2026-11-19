"""Stage 5: independent audit of the training pipeline, overfit proof,
pad-key attention masking, DDP mechanics and the 2B cluster matrix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cortex.model.configs import CORTEX_2B, get_config
from cortex.model.transformer import CortexTransformer
from cortex.training.audit import (
    audit_alignment,
    audit_gradient_flow,
    audit_loss_and_padding,
    overfit_probe,
)
from cortex.training.distributed import spawn_ddp_validation
from cortex.training.scaling import cluster_plans
from cortex.training.sft import make_batch
from cortex.tokenizer.cortex_tokenizer import train_tokenizer


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tokstage5")
    pool = Path(__file__).resolve().parents[1] / "cortex" / "learning"
    merged = tmp / "stdlib_sample.py"
    merged.write_text((pool / "reference_algorithms.py").read_text())
    train_tokenizer([str(tmp)], str(tmp), vocab_size=512)
    return tmp


@pytest.fixture(scope="module")
def tok(tok_dir):
    from cortex.tokenizer.cortex_tokenizer import load_tokenizer

    return load_tokenizer(str(tok_dir / "tokenizer.json"))


@pytest.fixture(scope="module")
def recs():
    return [
        {"name": f"fn{i}", "module": "m", "kind": "function",
         "instruction": f"Implement fn{i} that sums a list.\n",
         "code": f"def fn{i}(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"}
        for i in range(6)
    ]


def test_audit_alignment_all_pass(recs, tok):
    res = audit_alignment(recs, tok, seq_len=256)
    assert res["n_records"] == len(recs)
    assert res["all_pass"], res["checks"]
    assert {c["check"] for c in res["checks"]} == {
        "shift: y[t] == x[t+1] over real tokens",
        "prompt labels fully masked (-100)",
        "completion region labels = actual next tokens",
        "final unmasked target is EOT",
        "padded labels masked (-100)",
    }


def test_audit_alignment_fails_when_records_do_not_fit(recs, tok):
    res = audit_alignment(recs, tok, seq_len=8)
    assert not res["all_pass"]


def test_audit_loss_and_padding_all_pass(recs, tok):
    res = audit_loss_and_padding("cortex-demo", tok, recs[:3], seq_len=256)
    assert res["loss_matches_token_mean_over_unmasked"]
    assert res["pad_keys_excluded"]
    assert res["unmasked_targets"] > 0
    assert res["all_pass"]


def test_audit_gradient_flow_all_pass():
    res = audit_gradient_flow("cortex-demo")
    assert res["loss_finite"]
    assert res["params_total"] > 0
    assert res["params_without_grad"] == []
    assert res["params_nonfinite_grad"] == []
    assert res["params_zero_grad"] == []
    assert res["all_pass"]


def test_pad_mask_output_invariant_to_pad_identity():
    torch.manual_seed(4242)
    model = CortexTransformer(get_config("cortex-demo"))
    model.eval()
    B, T, V = 2, 12, model.cfg.vocab_size
    real_len = 7
    base = torch.randint(0, V, (B, real_len))
    pad = model.cfg.vocab_size - 1

    def padded(pad_id):
        x = torch.full((B, T), pad_id, dtype=torch.long)
        x[:, :real_len] = base
        return x

    x1 = padded(pad)
    x2 = padded((pad + 5) % V)
    with torch.no_grad():
        l1, _ = model(x1, pad_mask=x1 == pad)
        l2, _ = model(x2, pad_mask=x2 == pad)
        l3, _ = model(x1, pad_mask=None)
    assert torch.allclose(l1[:, :real_len], l2[:, :real_len], atol=1e-4)
    assert torch.all(torch.isfinite(l1))
    assert torch.allclose(l1[:, :real_len], l3[:, :real_len], atol=1e-4)


def test_pad_tokens_never_receive_loss(recs, tok):
    x, y = make_batch(recs, tok, 256)
    pad = tok.pad_id
    assert int(((y != -100) & (x == pad)).sum()) == 0
    real = (x != pad).nonzero()
    assert real.numel() > 0
    for b in range(x.shape[0]):
        pos = (x[b] != pad).nonzero().flatten()
        assert int(y[b, int(pos[-1]) - 1].item()) == tok.eot_id


def test_overfit_probe_memorizes_verified_batch(recs, tok, tok_dir, tmp_path):
    fit = tmp_path / "fit.jsonl"
    with open(fit, "w", encoding="utf-8") as fh:
        for r in recs[:3]:
            fh.write(json.dumps(r) + "\n")
    res = overfit_probe("cortex-demo", str(tok_dir / "tokenizer.json"),
                        str(fit), n_records=3, steps=150, batch_size=3,
                        lr=1e-3, target_loss=1.0)
    assert res["steps_executed"] >= 1
    assert res["reached_target"], res
    assert res["loss_first"] > 2.0
    assert res["greedy_next_token_accuracy_on_memorized_examples"] >= 0.5
    assert res["free_running_sample"] != ""
    assert res["peak_rss_mb"] > 0


def test_cluster_plans_2b_matrix():
    cfg = CORTEX_2B
    plans = cluster_plans(cfg, batch_size=8)
    assert plans["params"] == 2056012800
    assert plans["params"] == cfg.theoretical_param_count()
    by_world = {p["world_size"]: p for p in plans["plans"]}
    assert set(by_world) == {2, 4, 8}
    assert not by_world[2]["fits_40gb_gpu"]
    assert not by_world[2]["fits_80gb_gpu"]
    assert not by_world[4]["fits_40gb_gpu"]
    assert by_world[4]["fits_80gb_gpu"]
    assert by_world[8]["fits_40gb_gpu"]
    assert by_world[8]["per_gpu_total_gb_est"] <= 40.0
    assert by_world[8]["micro_batch"] == 1
    totals = [by_world[w]["per_gpu_state_gb"] * w for w in (2, 4, 8)]
    assert max(totals) - min(totals) < 0.5


def test_ddp_validation_real_processes(tmp_path):
    ids = np.random.default_rng(1).integers(256, 500, size=2000).astype(np.int32)
    bin_path = tmp_path / "tokens.npy"
    np.save(bin_path, ids)
    res = spawn_ddp_validation("cortex-demo", str(bin_path),
                               str(tmp_path / "out"), world_size=2,
                               steps=3, batch_size=2)
    assert res["backend"] == "gloo"
    assert res["world_size"] == 2
    assert res["checkpoint_reloaded"]
    assert res["max_param_spread_across_ranks"] <= 1e-6
    for r in res["per_rank"]:
        assert r["losses_finite"]
    assert res["loss_first"] == pytest.approx(res["per_rank"][0]["loss_first"])
