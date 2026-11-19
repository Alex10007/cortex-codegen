"""Code-specialized BPE tokenizer training and loading.

Trained on the CPython 3.12 standard library (PSF-licensed), byte-level BPE with
byte fallback, so encoding/decoding is lossless by construction and verified by
test. Indentation/newline groups survive pre-tokenization for code structure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from cortex.interfaces import Provenance


SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>"]


def _iter_py_files(roots: list[str], max_bytes: int, max_file_bytes: int) -> list[Path]:
    files: list[tuple[int, Path]] = []
    seen: set[bytes] = set()
    total = 0
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                path = Path(dirpath) / fn
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                if len(data) > max_file_bytes or len(data) < 16:
                    continue
                digest = hashlib.sha256(data).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                files.append((len(data), path))
    files.sort(key=lambda t: t[0], reverse=True)
    selected: list[Path] = []
    for size, path in files:
        if total + size <= max_bytes:
            selected.append(path)
            total += size
    return selected


def corpus_provenance(roots: list[str], version: str = "3.12") -> Provenance:
    return Provenance(
        source_uri=",".join(sorted(set(roots))),
        license="Python-2.0 / PSF",
        version=version,
        content_sha256="per-file sha256 stored per knowledge item",
        collected_at=_utcnow_iso(),
        collector="cortex.tokenizer.cortex_tokenizer.collect_code_corpus",
    )


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def train_tokenizer(roots: list[str], out_dir: str, vocab_size: int = 32768,
                    max_bytes: int = 16_000_000) -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    files = _iter_py_files(roots, max_bytes=max_bytes, max_file_bytes=300_000)
    if not files:
        raise RuntimeError("no corpus files found")
    files_str = [str(p) for p in files]

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train(files_str, trainer)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path / "tokenizer.json"))
    meta = {
        "vocab_size": vocab_size,
        "corpus_files": len(files),
        "corpus_bytes": sum(p.stat().st_size for p in files),
        "corpus_roots": roots,
        "provenance": corpus_provenance(roots).__dict__,
    }
    (out_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(out_path / "tokenizer.json")


def load_tokenizer(path: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    tok.no_padding()
    return CortexTokenizer(tok)


class CortexTokenizer:
    """Thin wrapper exposing train/pad-free encode/decode + special tokens."""

    PAD = SPECIAL_TOKENS[0]
    EOT = SPECIAL_TOKENS[1]

    def __init__(self, tok):
        self._tok = tok

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        tid = self._tok.token_to_id(self.PAD)
        assert tid is not None, "tokenizer is missing the pad special token"
        return tid

    @property
    def eot_id(self) -> int:
        tid = self._tok.token_to_id(self.EOT)
        assert tid is not None, "tokenizer is missing the EOT special token"
        return tid

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def roundtrip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text
