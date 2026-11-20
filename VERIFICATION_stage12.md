# Stage 12 Verification — Autoregressive Failure Analysis, Structural Learning and Controlled Training Experiments

Project: Cortex Codegen · Owner: Fibonera · Stage: 12
Parent commit: `082c5eb` (Stage 11) · Branch: `vibe/cortex-core-b71b2c` (PR #1 head)
Execution mode: real repository implementation, real model execution, controlled CPU experiments at demo scale. Zero cloud spend, zero GPU hours.

---

## 0. Headline result

**NOT PROMOTED.** Pre-registered promotion rule (fixed before execution):
`PROMOTE iff fixed-decoding pass@1 > 0.0 on v1 or v3 AND held-out loss <= 3.3015 (stage-10 measured parent value); a syntax-valid but functionally incorrect generation does NOT count.`

All five evaluated checkpoints (dense-control parent + 4 stage-12 training arms) produced **pass@1 = 0.0** on the immutable v1 and v3 suites under identical fixed decoding. The rule was applied unchanged; the negative result is preserved.

The stage produced the first **mechanistic** explanation of the 0.0:

> **Root cause (A-workstream, 45 instrumented task-subject generations):** 43/45 free-running failures are *directly-introduced persistent syntax defects* — a single emitted token makes an incrementally valid prefix permanently parse-invalid. The dominant category is **incorrect indentation transitions (38/45 overall)**, not premature stopping, not truncation, and not the training/eval prompt mismatch. The model emits EOT itself in the great majority of runs (stopped=`eot`: 38/45), so "never terminates" is also ruled out as the binding constraint.

Implications measured in this stage:

- The EOT-in-context mismatch is **not** the binding constraint: an eval-aligned training arm (training prompts byte-identical to benchmark prompts, EOT at context start, 0 vs 184 EOT-before-supervised-target occurrences) reached **eval-aligned held-out 2.9915 vs 2.9498 for the control — 0.0417 worse** — and parse/pass rates did not improve (F: 0.0667 vs 0.0 syntax-valid rate, pass@1 0.0 both).
- Filtering structurally-incomplete chunked windows (49.84% of chunked windows end inside an open syntactic construct) made training **worse** (eval-aligned final 3.0267 vs 2.9498 at the same 2000-step horizon; the filter also trains fewer effective steps: 1910/1892 of 2000). Structural window filtering is rejected at demo scale for now.
- Teacher-forced vs free-running gap persists (train 0.9322 / held-out 0.6214 token accuracy; free-running parse rate 0.0). Reference-token probability probes show the model assigns mean ref-token logprob −3.65 to −5.77 to reference completions under both prompt styles — reasonable probabilities that still do not survive autoregressive drift. This is consistent with exposure bias plus weak structural modeling at 8,071,296-parameter demo scale.
- Constrained (EOT-gated) decoding lifted syntax-valid rate from 0.0 to 0.0667 (1/15) as an exploratory arm only and did not produce a passing program — decoding mechanics are not concealing a working model.

## 1. Workstream acceptance matrix

| Workstream | Scope | Status | Evidence |
|---|---|---|---|
| A — token-by-token autoregressive failure analysis | incremental first-invalid-token analyzer, self-tested; 45 trajectories; A/B reference-token probability | **PASS (executed)** | `stage12_autoregressive_diagnostics.json` |
| B — syntax & program-structure learning | dataset + chunk-boundary structural audit | **PASS (executed)** | `stage12_structure_learning.json` |
| C — memorization vs generalization | overlap, contamination, perturbation probes, TF accuracy | **PASS (executed)** | `stage12_generalization.json` |
| D — controlled training experiments | 4 single/combination-variable arms, 2000 steps, same parent+seed | **PASS (executed); all arms NOT PROMOTED** | `stage12_training_experiments.json` |
| E — free-running generation reliability | tokenizer round-trips, prompt A/B, EOT-gated exploratory arm | **PASS (executed)** | `stage12_generation_reliability.json` |
| F — demo-scale capability increment | immutable v1+v3, fixed decoding, strict gates | **PASS (executed); pass@1 = 0.0 → NOT PROMOTED** | `stage12_capability_evaluation.json` |
| G — model capacity & GPU readiness | demo capacity measurement, 2B gate | **BLOCKED (no CUDA; ~46 GB needed vs 5.44 GB available)** | `stage12_capacity_readiness.json` |
| H — controlled promotion & regression | gates re-derived from per-task rows; rollback preserved | **PASS (executed); decision NOT PROMOTED** | `stage12_promotion_audit.json` |
| I — independent audit & reproducibility | independent re-parse, re-hash, re-derivation | **PASS (10/10 checks PASS)** | `stage12_independent_audit.json` |

Overall: the stage is **engineering-complete and fully executed at CPU demo scale**; the capability objective (first pass@1 success) was **not achieved**, honestly recorded, and stage-11 behavior is preserved.

## 2. Exact measured results

### A — autoregressive diagnostics (fixed decoding, seed 20261212, 15 tasks × 3 subjects)

| Subject | parse rate | pass@1 | first-invalid class | directly-introduced share | stopped |
|---|---|---|---|---|---|
| dense-control | 0.0 | 0.0 | 13 direct / 2 incomplete | 0.8667 | 10 EOT, 5 max-tokens |
| stage11-legacy | 0.0 | 0.0 | 15 direct | 1.0 | 15 EOT |
| stage11-chunked | 0.0 | 0.0 | 15 direct | 1.0 | 13 EOT, 2 max-tokens |

First-invalid categories (aggregate 45 rows): `incorrect_indentation` 38 · `other_parser_confirmed_syntax_error` 5 · `premature_stop_open_construct` 2.

EOT-in-context accounting (re-derived independently by the audit with the supervision-relevant definition "EOT tokens occurring before the first supervised target in actual padded batches"):
- legacy rows: **0** occurrences (EOT never conditions a supervised token),
- eval_aligned rows: **184** EOT occurrences ahead of supervised targets across the 241-record stream — while benchmark prompts condition on EOT. The training/eval surface mismatch was real, and correcting it did not fix generation (D/E), so it is a second-order effect.

Analyzer self-test (valid programs, hard-error persistence, transient-typing tolerance, incomplete-open discrimination): PASS.

### B — structure learning

- 49.84% of chunked stream windows (legacy style) end inside an open syntactic construct; the `complete_windows` arm trains only on the remaining 50.16%.
- Dataset is single-function dominated: class-definition rate 0.0; record-level complete-program rate 0.988 (records end cleanly; the *chunked training windows* do not).

### C — memorization vs generalization

- Exact duplicate records: 0; repeated function bodies: 2; train/val/heldout near-duplicate pairs (≥0.8 jaccard): 22.
- Benchmark contamination: max overlap 0.2472 (`t-wordcount` vs `repo::cortexpkg_textools`) — consistent with stage 10; benchmarks remain immutable.
- Teacher-forced token accuracy: train subset 0.9322, held-out subset 0.6214.
- 14 execution-validated perturbation probes (renamed functions, banned original test names; 1 probe rejected: no def/class): parse rate 0.0, pass@1 0.0 — the model does not transfer to renamed/isomorphic variants.
- Memorization probes vs own code: mean word-jaccard 0.3261 — free-running output is not a memorized copy of training code either; it is illegitimate novel text.

### D — controlled arms (2000 steps, EVAL_EVERY 400, AdamW lr 5e-5 b(0.9,0.98) wd 0.01, batch 4, warm-6 parent, seed 20260101, identical batch cycling)

| Arm | single variable | eval-aligned held-out final | Δ vs control | v1 pass@1 | v3 pass@1 | effective steps |
|---|---|---|---|---|---|---|
| chunked_control | none (stage-11 config) | **2.9498** | — | 0.0 | 0.0 | 2000/2000 |
| eval_aligned | prompt surface only | 2.9915 | +0.0417 (worse) | 0.0 | 0.0 | 2000/2000 |
| complete_windows | complete windows only | 3.0267 | +0.0769 (worse) | 0.0 | 0.0 | 1910/2000 |
| eval_aligned_complete | declared combination | 3.0308 | +0.0810 (worse) | 0.0 | 0.0 | 1892/2000 |

Held-out at start (parent): legacy-style 3.3015, eval-aligned 3.3026 — the control arm improves −0.3517 over the parent at this horizon; all three remediation variants are worse. Champion by the pre-registered rule (max pass@1, tie-break min held-out): `chunked_control`.

### E — generation reliability

- Tokenizer round-trips (incl. EOT/pad-bearing probes): 4/4 re-encode-identical.
- Prompt-style A/B at fixed decoding (3 tasks × 2 subjects): parse rate 0.0 in **both** styles for both subjects — changing the prompt surface to training form does not unlock parsing.
- EOT-gated constrained decoding (EXPLORATORY, labeled as such in the artifact): parse 0.0667 vs 0.0 unconstrained at the same seed; complete-program 0.0; pass@1 0.0. Constraints neither hide nor reveal a working generator.

### F/G/H — capability, capacity, promotion

- F: pass@1 = 0.0 for all 5 checkpoints; syntax_valid_rate ≤ 0.0667; complete_program_rate 0.0; execution attempted 1.0. **NOT PROMOTED.**
- G: demo model measures exactly 8,071,296 parameters (matches config); CORTEX_2B config verifies 2,056,012,800; measured avg legacy prompt 146.5 tokens at seq 256; local 2B gate: infeasible (~46.0 GB needed vs 5.44 GB available; 0 CUDA devices). **BLOCKED.**
- H: pass@1-gate FAIL (0.0), no-regression-gate PASS, held-out-loss-gate PASS (2.9498 ≤ 3.3015) → **NOT PROMOTED**; warm-6 baseline preserved; stage-12 arm checkpoints are additive (`artifacts/runs/cortex-sft-stage10/cortex-sft-stage12-*.pt`) with recorded sha256.

## 3. Independent audit

`stage12_independent_audit.json` — 10 PASS / 0 FAIL / 0 BLOCKED, including: fingerprints v1/v2/v3 recomputed and matched immutable values; benchmark files byte-identical vs `082c5eb`; all arithmetic re-derived from per-task rows; stored generation text re-parsed with the AST (rates within tolerance); checkpoint sha256 recomputed from disk; authoritative 502B store recount slots_updated = 35,292 (unchanged, 216,834,048 unique trained virtual scalars); analyzer self-test re-executed.

## 4. Integrity notes

- Two audit-side defects were found and fixed during verification (both are test/audit bugs, not benchmark or data changes): (1) v3 fingerprint re-derivation must use the 5-field `tasks_v3_tuples` (module_name included); (2) the EOT-in-context claim must be evaluated with the supervision-relevant definition — padded batch rows contain the row-final EOT before a masked target, which conditions nothing.
- `torch.multinomial` CPU draws were observed to differ across process launches under identical seeds (stable within a process). Sampling-sequence equality is therefore NOT used as a correctness gate; instrumentation equivalence is verified deterministically (greedy top-5 vs raw forward math). Sampling-based benchmark comparisons used fixed decoding with in-process seeding, exactly as in stages 10/11.
- Pre-existing Stage 7–11 artifacts, datasets, fingerprints, checkpoints and the authoritative 502B virtual store were not modified. `git status` shows only stage-12 additions/extensions. Spend: $0.00; GPU hours: 0.0.

## 5. Direct answers to the stage-12 questions

1. **What converts teacher-forced learning into free-running failure?** Directly-introduced persistent syntax defects at indentation transitions (38/45 rows), plus autoregressive drift: the model holds plausible per-token probabilities (mean ref-token logprob −3.65…−5.77) but its own generated context compounds small structural errors immediately and permanently.
2. **Is the prompt/EOT mismatch the cause?** No — correcting it (eval-aligned arm) worsened held-out loss (+0.0417) and did not improve parsing; the A/B confirms 0.0 parse rate under both prompt styles.
3. **Do structurally clean training windows help?** No at this scale — the complete-windows arm trained 50.16% of windows and was worse on held-out loss (+0.0769).
4. **Is it memorization?** Partially excluded: no exact duplicates, output jaccard vs own-code averages only 0.3261, and semantically perturbed probes are all failures, so the model transfers neither memorized code nor isomorphic patterns.
5. **Is pass@1 = 0 a decoding artifact?** No — fixed decoding was confirmed mechanically sound (round-trips, prompt A/B, EOT-gating), constrained decoding reaches only 1/15 parses and 0 passes.
6. **What is the next measured lever?** Indentation-aware structural supervision (whitespace-token diagnostics), larger training horizon/scale, and per-position curriculum on indentation transitions; and above all GPU qualification for 2B-scale training, which remains hard-BLOCKED (46 GB vs 5.44 GB).

## 6. Test results

- Full suite: **182 passed, 0 failed** (`python -m pytest tests/ -q`), including 11 new stage-12 tests (analyzer semantics, greedy instrumentation equivalence vs raw forward math, prompt-style record-ids, EOT-before-supervised-target invariant, complete-window filter, probe determinism, artifact/promotion integrity).
- Pyflakes clean on all touched files.

## 7. Artifacts produced

`stage12_report.json` · `stage12_autoregressive_diagnostics.json` · `stage12_structure_learning.json` · `stage12_generalization.json` · `stage12_training_experiments.json` · `stage12_generation_reliability.json` · `stage12_capability_evaluation.json` · `stage12_capacity_readiness.json` · `stage12_promotion_audit.json` · `stage12_independent_audit.json` · `stage12_failure_taxonomy.json` · `tests/test_stage12.py` · `scripts/stage12_run.py` (+ `--finalize-from-artifacts` recovery path) · `scripts/stage12_audit.py`

Ownership: Fibonera retains full ownership and control of Cortex Codegen, its source, architecture, checkpoints, datasets and resulting IP.
