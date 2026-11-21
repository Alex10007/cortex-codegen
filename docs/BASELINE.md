# Cortex Codegen — Implementation Baseline (audited before any implementation)

## 1. Existing repository state (as found)

| Artifact | State |
|---|---|
| `app.py` | Gradio `ChatInterface` that proxies **OpenAI `gpt-oss-20b`** via the Hugging Face Inference API. This is a third-party model wrapper, not owned intelligence. It violates the Fibonera ownership directive (core must not depend on external model APIs). |
| `README.md` | HF Spaces template placeholder ("An example chatbot"). |
| `requirements.txt` | Generic web deps (gradio, fastapi, uvicorn, jose, bcrypt). No ML deps. |
| `.github/workflows/main.yml` | CI that imports `cortex.core.CortexAgent`, `cortex.modules.CodeGenerator/ImageGenerator/VideoGenerator` and `app.HuggingFaceGradioDeployment` — **none of these exist in the repo**. CI is fabricated relative to reality and would fail today. It also force-deploys to HF Spaces on push to main. |
| `.gitattributes` / `README` frontmatter | HF Space metadata (sdk: gradio). |
| Model weights, tokenizer, training code, knowledge infra, evals | **Absent entirely.** |

## 2. Reusable components identified

- FastAPI/uvicorn dependency set → reusable for the API Gateway.
- Gradio UI → retained behind the Cortex API as an optional interface, no longer the core.
- GitHub Actions harness → retained, but rewritten to test the real modules instead of nonexistent ones.
- Nothing else was implementable: no core model, tokenizer, knowledge, execution, testing, security, or evaluation code exists.

## 3. Hardware profile (measured, not assumed)

| Property | Measured value |
|---|---|
| CPU | 5 cores |
| RAM | ~6 GB |
| GPU | none (CPU-only sandbox) |
| Disk | ~5.4 TB free |

Consequence: a full 2B-parameter training run cannot be executed on this machine.
Per directive, the 2B configuration is defined and parameter-counted exactly, but is
marked **declared, not trained**. The training pipeline is validated end-to-end at a
verified smaller scale (config `cortex-demo`), with all metrics measured from real runs.

## 4. License-clean corpus available offline

- CPython 3.12 standard library: 6,202 `.py` files, ~59 MB, Python Software Foundation
  License. Used as: tokenizer training corpus, pretraining/fine-tuning corpus, source of
  verified knowledge items (provenance = file path + version + PSF license).
- Sandbox test repos of the session itself for evaluation tasks (execution-verified).

## 5. Implementation gap list (from the directive)

1. Model Core (transformer backbone) — MISSING
2. Code-specialized tokenizer — MISSING
3. Training engine (loop, checkpoints, fault recovery, hardware profiling) — MISSING
4. Knowledge Engine (vector store, knowledge graph, provenance, versioning, capacity
   measurement distinguishing stored/indexed/retrievable/validated) — MISSING
5. Persistent engineering memory — MISSING
6. Planning engine — MISSING
7. Execution engine + sandbox isolation — MISSING
8. Testing engine — MISSING
9. Security engine (static analysis, secret scanning) — MISSING
10. Evaluation framework (execution-based, contamination-controlled) — MISSING
11. API Gateway + monitoring — MISSING
12. Interface routines: repository analysis before modification — MISSING
13. Continuous learning controls (verified-outcomes-only ingestion) — MISSING

## 6. Acceptance criteria for this milestone

- Every engine is a real, importable, tested component. No mock objects, no
  simulated results, no placeholder intelligence.
- The tokenizer is actually trained on the corpus and verified by lossless round-trips.
- The transformer's parameter count is computed by construction exact arithmetic and
  verified by instantiating an actual model and counting trainable tensors.
- The demo config is actually trained; a checkpoint, loss curve and evaluation record
  exist. The 2B config is marked *declared, not trained* everywhere it appears.
- The knowledge engine ingests real corpus items with provenance, and its capacity
  report prints only measured counts (stored / indexed / retrievable / validated).
- The sandbox demonstrably blocks network access when the kernel allows it, and
  enforces CPU/RAM/time limits on all runs.
- The security analyzer demonstrably flags real vulnerability patterns via AST.
- The evaluation runner grades by executing code and records pass rates from those runs.
- Full `pytest` suite passes; CI rewritten to test real modules only.
