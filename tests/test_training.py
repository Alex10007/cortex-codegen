import json
import math
import sysconfig
from pathlib import Path

import pytest

pytest.importorskip("torch")

from cortex.model.configs import MODEL_STATUS, get_config
from cortex.tokenizer.cortex_tokenizer import load_tokenizer, train_tokenizer
from cortex.training.engine import pack_corpus, train

STD_SUBSET = str(Path(sysconfig.get_paths()["stdlib"]) / "json")


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    tok_path = train_tokenizer([STD_SUBSET], str(tmp / "tok"), vocab_size=1024,
                               max_bytes=300_000)
    data_bin = str(tmp / "tokens.npy")
    path, n_tokens, n_files = pack_corpus(tok_path, [STD_SUBSET], data_bin,
                                          max_bytes=300_000)
    assert n_tokens > 1000 and n_files >= 3
    return tok_path, data_bin


def test_train_produces_checkpoint_and_loss_drop(tiny_corpus, tmp_path):
    tok_path, data_bin = tiny_corpus
    summary = train("cortex-demo", tok_path, data_bin, str(tmp_path / "run"),
                    steps=8, batch_size=8, log_every=2, keep_checkpoints=2)
    assert Path(summary["checkpoint"]).exists()
    assert summary["param_count"] == get_config("cortex-demo").theoretical_param_count()
    assert summary["final_loss"] is not None
    assert summary["final_loss"] < math.log(get_config("cortex-demo").vocab_size)
    assert MODEL_STATUS["cortex-demo"]["status"] == "trained"
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    metrics = [json.loads(l) for l in
               (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 8


def test_train_resume_from_checkpoint(tiny_corpus, tmp_path):
    tok_path, data_bin = tiny_corpus
    out = str(tmp_path / "run")
    train("cortex-demo", tok_path, data_bin, out, steps=4, batch_size=8,
          log_every=2, resume=False)
    summary = train("cortex-demo", tok_path, data_bin, out, steps=6, batch_size=8,
                    log_every=2, resume=True)
    assert summary["steps"] == 6
    assert Path(summary["checkpoint"]).exists()
