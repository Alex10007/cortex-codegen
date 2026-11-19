import pytest

tokenizers = pytest.importorskip("tokenizers")

from cortex.tokenizer.cortex_tokenizer import (
    SPECIAL_TOKENS,
    CortexTokenizer,
    corpus_provenance,
    load_tokenizer,
    train_tokenizer,
)

STD_SUBSET = "/usr/local/lib/python3.12/json"
CODE_SNIPPET = (
    "def foo(items=None):\n"
    "    if not items:\n"
    "        return \"ok\"\n"
    "    return [x for x in items if x is not None]\n"
)


def test_train_and_roundtrip(tmp_path):
    tok_path = train_tokenizer([STD_SUBSET], str(tmp_path), vocab_size=2048,
                               max_bytes=400_000)
    tok = load_tokenizer(tok_path)
    assert tok.vocab_size <= 2048
    assert tok.roundtrip_ok(CODE_SNIPPET)
    with open("/usr/local/lib/python3.12/json/encoder.py", encoding="utf-8") as fh:
        std_text = fh.read()
    assert tok.roundtrip_ok(std_text)
    assert all(s in tok._tok.get_vocab() for s in SPECIAL_TOKENS)
