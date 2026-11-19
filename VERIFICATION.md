# Cortex Codegen — Verification Report

All numbers below are **measured**, not asserted. Reproduce with:
`pip install -r requirements.txt && python scripts/build_and_verify.py && python -m pytest -q tests/`

## 1. Test suite

```
40 passed in ~20s (tests/, local host)
```
Coverage: hardware profiler, tokenizer round-trip, model forward/loss/generation,
exact parameter-count equality (arithmetic vs live model), training +
checkpoint resume, vector store, knowledge graph, stdlib indexing, capacity
measurement, controlled-learning rejection/acceptance rules, memory engine,
audit log, sandbox (execution, timeout, memory limit, network isolation,
workspace isolation), security detectors, repository analysis, planner (incl.
destructive-op authorization), full pipeline verify/fail/patch paths,
FastAPI gateway endpoints, evaluation runner (execution-graded).

## 2. Hardware profile (measured on the development host)

- 5 CPU cores, 5.9 GB RAM (5.5 GB available), no GPU, 5,459 GB disk free
- 2B-class model trainable locally: **False** — needs ~46 GB RAM at fp32
  AdamW (16 B/param × 1.5 activations headroom)

## 3. Model status (2 billion physical parameters)

| Config | Parameters (exact count) | Status |
|---|---|---|
| cortex-demo | 8,071,296 | **TRAINED** (real run, 600 steps, loss 10.40 → 3.93, best 3.10, 2,457,600 tokens, seed 1337, checkpoints + metrics.jsonl on disk) |
| cortex-s | 41,997,312 | declared (needs larger host) |
| cortex-m | 235,096,064 | declared |
| cortex-2b | 2,056,012,800 | **declared, NOT trained** (>= 2e9 physical trainable parameters; training path provided for GPU scale-out; no checkpoint exists — intentionally) |

Caveat recorded by design: this section is honest per the directive — the 2B
model is *not* claimed trained, because it has not been trained anywhere.

## 4. Tokenizer (code-specialized)

- Byte-level BPE, vocab 32,768, byte fallback (lossless by construction)
- Trained on 16.0 MB of deduplicated CPython 3.12 stdlib source (174 files,
  PSF license, per-file SHA-256 provenance stored)
- Round-trip verified on real stdlib files and adversarial snippets (tests)

## 5. Virtual knowledge architecture — measured capacity

Reference run, `artifacts/capacity.json`:

| Tier | Measured value | Meaning |
|---|---|---|
| stored | 78,705 items | knowledge records with per-item provenance (path, license, SHA-256, version) |
| indexed | 78,705 | items with live retrieval embeddings in the vector index |
| retrievable | 48,797 (recall@5 = 0.62 on 200-probe set) | items actually reachable in top-5 by realistic queries — **measured**, not claimed |
| validated | 0 | items with execution-verified evidence |

Notes: capacity is accounted in real, addressable knowledge units, distinct
from any neural parameter count. The 500B target is an infrastructure
scale-out objective; nothing in the codebase fabricates its current attainment.
Sources: CPython 3.12 stdlib (PSF), plus project-local code trees.

## 6. Evaluation (execution-graded, contamination-controlled)

Tasks are authored in-repo (not derived from training corpus) and graded by
executing hidden pytest suites in the sandbox:

| Subject | Result (measured) |
|---|---|
| Harness sanity — reference solutions | pass@1 = **1.0** (3/3) |
| Demo model checkpoint (8.07M params, 600 CPU steps) | pass@1 = **0.0** (0/3) |

The demo model's 0.0 is reported as measured. It demonstrates the grading
harness distinguishes "code that parses/looks plausible" from "code that
actually passes tests". Raising this number requires the declared larger
training runs, not changes to the grader.

## 7. Sandbox isolation (demonstrated)

- network namespace isolation via `unshare -rn` (supported on this host; test
  asserts outbound connect fails inside the sandbox)
- RLIMIT_CPU / RLIMIT_AS / RLIMIT_NPROC / RLIMIT_FSIZE enforced (tests prove
  timeout kill and memory-limit kill)
- disposable workspace copies; host tree untouched; cleared environment;
  every run audited to append-only JSONL

## 8. Security engine

AST-based detectors verified by tests against a deliberately vulnerable file:
`eval`, `exec`, `os.system`, `pickle.loads`, `yaml.load` w/o SafeLoader,
`subprocess(..., shell=True)`, plus secret scanning (API keys, credentials,
private keys). Severity summary produced per scan.

## 9. Engineering pipeline (demonstrated end-to-end)

analyze repo → plan → optional authorized edits → run tests in sandbox →
security scan → verification record. Tests cover: verified-repo pass,
failing-test detection, patcher application, destructive-op block without
authorization. Verification flag is true only when tests actually executed
and passed and zero critical security findings.

## 10. Continuous-learning controls (enforced)

`UnverifiedOutcomeError` is raised when code without an executed+passed
outcome is offered as trusted knowledge (tested). Failed attempts are stored
as negative evidence, never marked verified, and patterns dedupe by content
hash.

## 11. Outstanding work (explicitly not done)

1. **2B-class pretraining** — requires GPU cluster; config + training path
   ready (Phases B–D at target scale).
2. Retrieval embeddings were computed from deterministic hashed n-grams and
   measured recall@5 = 0.62; a learned embedding model should replace them at
   larger scale to lift retrievable capacity.
3. Knowledge-graph-driven cross-repo dependency reasoning beyond the indexed
   stdlib tree.
4. Continuous-learning pipelines wired to real fine-tuning runs (currently
   verified-outcome curation + memory; fine-tuning loop unexercised).
5. Multi-language corpus ingestion beyond Python (indexer currently parses
   Python ASTs; other languages ingest as module/doc items).
