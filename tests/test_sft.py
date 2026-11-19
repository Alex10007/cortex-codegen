import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from cortex.tokenizer.cortex_tokenizer import load_tokenizer, train_tokenizer
from cortex.training.sft import (
    build_dataset,
    contamination_overlap,
    evaluate_heldout,
    fine_tune,
    load_split,
    make_batch,
)

STD_SUBSET = "/usr/local/lib/python3.12/json"


def _pair(code, name="fn", license_="Python-2.0 / PSF", instruction="Implement something"):
    return {
        "name": name,
        "module": "m",
        "instruction": instruction,
        "code": code,
        "provenance": {"license": license_, "source_uri": "file:///x",
                       "version": "cpython-3.12"},
    }


def test_contamination_filter_flags_eval_tasks():
    from cortex.evaluation.tasks import TASKS

    leaked = _pair(TASKS[0].reference_solution,
                   instruction=TASKS[0].instruction)
    scores = contamination_overlap(leaked)
    assert max(scores.values()) > 0.5
    clean = _pair("def zip_noop(a):\n    return a\n",
                  instruction="Implement the Python function `zip_noop`.")
    assert max(contamination_overlap(clean).values()) < 0.5


def test_build_dataset_gates_and_splits(tmp_path):
    a = "def alpha():\n    return 1\n"
    b = "def beta():\n    return 2\n"
    pairs = [
        _pair(a, name="alpha"),
        _pair(a, name="alpha"),
        _pair(b, name="beta", license_="proprietary-eula"),
        _pair("def gamma(x):\n    return x + 1\n", name="gamma"),
    ]
    out = tmp_path / "ds.jsonl"
    report = build_dataset(pairs, str(out), seed=7)
    assert report["stats"]["deduped"] == 1
    assert report["stats"]["license_rejected"] == 1
    assert report["train"] + report["heldout"] == 2
    lines = out.read_text().splitlines()
    roles = {json.loads(l)["split"] for l in lines}
    assert roles == {"train", "heldout"}
    report2 = build_dataset(pairs, str(tmp_path / "ds2.jsonl"), seed=7)
    assert report2["train"] == report["train"]


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sft")
    tok_path = train_tokenizer([STD_SUBSET], str(tmp / "tok"), vocab_size=1024,
                               max_bytes=300_000)
    tok = load_tokenizer(tok_path)
    pairs = [
        _pair("def alpha():\n    return 1\n", name="alpha",
              instruction="Implement the Python function `alpha` returning one.\nMake alpha return int 1."),
        _pair("def beta(x):\n    return x * 2\n", name="beta",
              instruction="Implement the Python function `beta` doubling input.\nMake beta return x doubled."),
        _pair("def gamma(s):\n    return len(s)\n", name="gamma",
              instruction="Implement the Python function `gamma` measuring length.\nMake gamma return the length."),
        _pair("def delta(items):\n    return sorted(items)\n", name="delta",
              instruction="Implement the Python function `delta` sorting items.\nMake delta return sorted list."),
        _pair("def eps(a, b):\n    return a - b\n", name="eps",
              instruction="Implement the Python function `eps` subtracting.\nMake eps return a minus b."),
        _pair("def zeta(w):\n    return w.upper()\n", name="zeta",
              instruction="Implement the Python function `zeta` uppercasing.\nMake zeta return w in caps."),
    ]
    ds = build_dataset(pairs, str(tmp / "ds.jsonl"), held_out_fraction=0.2, seed=3)
    return tok_path, tok, str(tmp / "ds.jsonl"), ds


def test_make_batch_masks_prompt_tokens(tiny_setup):
    tok_path, tok, ds_path, ds = tiny_setup
    train = load_split(ds_path, "train")
    x, y = make_batch(train[:2], tok, 256)
    assert x is not None
    assert (x[:, 0] != tok.pad_id).all()
    assert (y[:, 0] == -100).all()
    assert (y != -100).any()


def test_heldout_loss_finite(tiny_setup):
    tok_path, tok, ds_path, ds = tiny_setup
    from cortex.model.configs import get_config
    from cortex.model.transformer import CortexTransformer

    model = CortexTransformer(get_config("cortex-demo"))
    heldout = load_split(ds_path, "heldout")
    loss = evaluate_heldout(model, heldout, tok,
                            get_config("cortex-demo").max_seq_len)
    assert loss is not None and loss == loss


def test_fine_tune_from_verified_outcomes(tiny_setup, tmp_path):
    tok_path, tok, ds_path, ds = tiny_setup
    from cortex.model.configs import MODEL_STATUS

    summary = fine_tune("cortex-demo", tok_path, ds_path,
                        init_checkpoint=None, out_dir=str(tmp_path / "sft"),
                        epochs=2, batch_size=2, lr=3e-4, log_every=1)
    assert Path(summary["checkpoint"]).exists()
    assert summary["train_examples"] == ds["train"]
    assert summary["heldout_loss_first_epoch"] is not None
    assert summary["train_loss_last"] == summary["train_loss_last"]
    sft_entries = MODEL_STATUS["cortex-demo"].get("sft", [])
    assert sft_entries and sft_entries[-1]["checkpoint"] == summary["checkpoint"]
    metrics = (tmp_path / "sft" / "metrics.jsonl").read_text().splitlines()
    assert metrics
