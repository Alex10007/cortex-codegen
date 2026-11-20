# Stage 13 Verification — Structural Code Generation, Syntax Reliability, Functional Correctness and Generalization

Project: Cortex Codegen · Owner: Fibonera · Stage: 13
Parent commit: `402de46` (Stage 12) · Branch: `vibe/cortex-core-b71b2c` (PR #1 head)
Execution mode: real repository implementation, real model execution, controlled CPU experiments at demo scale. Zero cloud spend, zero GPU hours, no simulated results.

---

## 0. Headline result

**NOT PROMOTED.** The pre-registered 5-gate promotion rule (registered in `arm13` configs before any stage-13 execution; never adjusted) required Gate 5 — at least one independently verified pass@1 on the immutable v1/v3 benchmarks — plus held-out ≤ 3.3015. All four stage-13 training arms scored **pass@1 = 0.0**; the capability milestone was not reached and the stage-12 champion (`cortex-sft-stage12-chunked_control`) is retained. `stage13_promotion_audit.json` records **gates passed: 0/5** and `promotion_decision: NOT PROMOTED`.

What the stage did establish, from executed runs:

> **Training-data → generation causal chain confirmed and narrowed.** Stage-12's indirect chunk statistics (49.84% of chunked windows end inside an open construct) were reproduced and then *directly repaired* in the training stream: a new opt-in structural-boundary segmentation (`chunk_record_structural`, `make_batch_chunked(..., segmentation="structural")`) cuts the mid-construct window share from **49.84% → 4.38% clean = 95.62%** (`stage13_training_stream.json`, `chunk_boundary_audit_train`) while the exactly-once completion-supervision guarantee holds **identically** in both modes on every split (train/val/heldout/full; independently re-derived by the audit). The stage-12 measurement was therefore not an artifact — and fixing it, while strictly beneficial at the stream level, did **not** translate into a held-out loss or pass@1 gain at the 2000-step demo horizon (arm `structural_segmentation`: held-out 2.9754 vs 2.9498 control).

Additional executed findings:

- **A (new token-level structural taxonomy):** over 30 fresh instrumented trajectories (2 subjects × 15 immutable tasks), the stage-12 result reproduces: parse rate 0.0 (parent) / 0.0667 (stage-12 champion), pass@1 0.0, indentation-related failure share 0.80 / 0.60, and **every persistent structural failure is directly introduced by the model's own next-token choice** (directly-introduced share 1.0 / 0.9286). Self-emitted EOT remains frequent (10/15 and 9/15) and never coincides with a complete program.
- **C (stream repair, opt-in):** `structural` segmentation removes mid-construct window ends (clean 0.9562 vs 0.5016 fixed) with no supervised-token loss and no prompt splitting; fallback hard cuts are documented. The fixed default is retained until a promotion passes.
- **D (indentation-weighted objective):** upweighting whitespace/indentation targets (weight 3.0) measurably improves held-out structural token accuracy (+0.39 pp whitespace vs control; combined arm +0.78 pp → 0.9094) but costs held-out loss (+0.0751 / +0.0452). Accuracy alone never promotes; the negative loss direction is preserved.
- **F (execution feedback):** all 60 generations classify through the *existing* `grade_single` + `diagnose_candidate` path — 29 syntax_generation_failure, 1 runs_but_tests_failing execution cases (`stage13_execution_feedback.json`). An optimizer-updated feedback-training experiment is pre-registered as NOT TESTED, deferred to a registered stage-14 run.
- **G (generalization):** 14/15 execution-validated perturbed probes (seed_base 20261300): pass@1 0.0, syntax 0.0714, complete 0.0; contamination 0.2472 unchanged; teacher-forced train 0.6816 vs held-out 0.5736 (champion, held-out-first-16 protocol).
- **J (subsystems):** MoE probe on real train batches: mean entropy 3.4624 bits near log2(12)=3.585 uniform across 4 layers with joint-checkpoint JS 0.0007–0.0014 — near-uniform routing persists; divergent reasoning (existing engine): pass@1 0.0, pass@k 0.0, repair recovery 0.0, mean convergence 1.0 on 2 tasks (all hypotheses fail identically).
- **I (infrastructure):** 2B training remains **BLOCKED** — 0 CUDA devices, ~46 GB RAM needed vs ~4.8–5.9 GB available; authoritative 500B virtual store untouched (slots_updated = 35292 recount).

The independent audit (`stage13_independent_audit.json`) reports **10 PASS / 0 FAIL**: fingerprints immutable, benchmarks untouched vs `402de46`, checkpoints sha-verified, promotion decision re-derived from per-task rows, store recounted, stage-7..12 artifacts intact.

## 1. Workstream acceptance matrix

| Workstream | Scope | Status | Evidence |
|---|---|---|---|
| A — structural root cause | per-step lexical trajectories, incremental first-invalid analysis, 13-category taxonomy | **PASS (executed)** | `stage13_structural_root_cause.json` |
| B — structural training-data audit | capability coverage matrix (3 splits), cut stats, overlap/contamination | **PASS (executed)** | `stage13_structural_data_audit.json` |
| C — training-stream integrity | structural segmentation, exactly-once coverage verification, boundary audit | **PASS (executed); opt-in, default unchanged** | `stage13_training_stream.json` |
| D — structural learning measurement | baseline + per-arm per-class held-out token accuracy | **PASS (executed); supporting evidence only** | `stage13_structural_learning.json` |
| E — generation control (EOT) | EOT semantics documented, stop classification, exploratory EOT-gated arm | **PASS (executed)** | `stage13_generation_control.json` |
| F — execution feedback | existing-engine classification of all generations; feedback run NOT TESTED | **PASS (executed classification); training arm NOT TESTED** | `stage13_execution_feedback.json` |
| G — generalization | perturbation probes, overlap/contamination, TF train-vs-heldout | **PASS (executed); pass@1 = 0.0** | `stage13_generalization.json` |
| H — controlled experiment matrix | 4 arms × 2000 steps, one variable each, identical seeds/horizon | **PASS (executed); all arms ≤ control; champion = control_recheck** | `stage13_experiment_matrix.json` |
| I — capacity & infrastructure | 2B gate, virtual store recount | **BLOCKED (2B; no CUDA)** | `stage13_capacity_and_infrastructure.json` |
| J — MoE & divergent reasoning | real-batch MoE probe; execution-verified divergent eval | **PASS (executed); negative verdicts preserved** | `stage13_intelligence_subsystems.json` |
| K — promotion & regression gates | pre-registered 5-gate hierarchy | **PASS (executed); decision NOT PROMOTED (0/5)** | `stage13_promotion_audit.json` |
| L — independent audit | re-parse, re-hash, re-derive, recount | **PASS (10/10 checks PASS)** | `stage13_independent_audit.json` |

Overall: the stage is **engineering-complete and fully executed at CPU demo scale**; the capability objective (first pass@1 success) was **not achieved** and the negative result is preserved with the required distinctions intact.

## 2. Exact measured results

### A — structural root cause (fixed decoding, seed 20261214, 15 tasks × 2 subjects)

| Subject | parse rate | pass@1 | complete rate | indentation share | directly-introduced | stopped |
|---|---|---|---|---|---|---|
| dense-warm6-parent | 0.0 | 0.0 | 0.0 | 0.8 | 1.0 | 10 EOT / 5 max-tokens |
| stage12-chunked_control | 0.0667 | 0.0 | 0.0 | 0.6 | 0.9286 | 9 EOT / 6 max-tokens |

Taxonomies: warm6 `unexpected_indentation_transition` 12, `other_parser_confirmed_syntax_error` 2, `premature_eot_open_construct` 1. Champion: `unexpected_indentation_transition` 9, `other` 4, `missing_opening_delimiter` 1, `complete_program` 1 (a parses-clean output that still fails tests).

Analyzer self-test: 10/10 categories agree (valid, open-only-incomplete, hard-error persistence incl. missing colon).

### B — structural data audit (fingerprinted dataset, 332 records)

- complete-program rate (train): **0.9834**; classes: 0.0%; nested functions: 0.0% — the stage-12 single-function bias re-measures identically on every split.
- duplicates: 0; contamination: 0.2472 (unchanged from stage-12).

### C — training stream

| segmentation | window-end clean | exactly-once completion supervision |
|---|---|---|
| fixed (stage-11 default) | 0.5016 | `complete=True`, 35,857 targets (train) |
| structural (new, opt-in) | **0.9562** | `complete=True`, **identical 35,857 targets (train); identical totals per split** |

`completion_coverage` is recorded for fixed and structural on train/val/heldout/all with `coverage_identical: True` everywhere; the audit re-derives both from a fresh reconstruction. Prompts are never split; fallback cuts documented. **Default behavior is unchanged** until a promotion passes.

### D — structural token accuracy (held-out, teacher-forced)

| model | whitespace | structure keywords | delimiters | overall | held-out (eval-aligned) |
|---|---|---|---|---|---|
| warm6 baseline (train-subset probe) | 0.9375 | — | — | — | 3.3026 (start of every curve) |
| control_recheck | 0.9016 | 0.678 | 0.6667 | 0.5414 | 2.9498 |
| structural_segmentation | 0.8976 | **0.7288** | 0.6667 | 0.5477 | 2.9754 |
| indent_weighted | **0.9055** | 0.6102 | 0.6458 | 0.5227 | 3.0249 |
| structural_indent_combined | **0.9094** | 0.661 | 0.6562 | 0.5352 | 2.9950 |

From `stage13_structural_learning.json` `post_by_arm`. Indentation weighting moves whitespace accuracy up; held-out loss moves the other way. Note the segmentation arm's structural-keyword gain (+0.0508 vs control) with near-neutral whitespace. Champion by pre-registered rule: `control_recheck` (lowest held-out). No arm promoted on the pre-registered rule.

### E — generation control

- EOT semantics (documented, not redefined): EOT id 200 is the trained completion terminator (`record_ids` supervises `code + "\n" + EOT`, exactly once per record) and `generate`/`trajectory_generate` stop on it.
- Stop classification (30 Workstream-A generations): every self-emitted EOT occurred with `taxonomy != complete_program` (0 of 19 EOT stops were complete programs) — consistent with the model treating EOT as a possible continuation inside open constructs.
- Exploratory EOT-gated run (stage-12 champion, 3 v1 tasks): parse 0.0, complete 0.0 — the exploratory decoding policy alone does not recover capability at this scale.

### F — execution feedback (existing engine only)

`stage13_execution_feedback.json`: 30 generations/subject classified via `grade_single` + `diagnose_candidate` (same code path as benchmark grading); totals — `syntax_generation_failure` 29, `runs_or_tests_failing` 1, `passed` 0. Feedback-derived training is **NOT TESTED** here and pre-registered for a stage-14 run with its own checkpoint lineage.

### G — generalization

- 14 execution-valid perturbation probes: pass@1 **0.0**, syntax-valid 0.0714 (1/14), complete 0.0.
- Contamination 0.2472 (unchanged); exact duplicates 0.
- Teacher-forced accuracy champion: train-subset 0.6816, held-out 0.5736 (16-record subsets, identical protocol).

### H — controlled experiment matrix (all arms: warm-6 parent, seed 20260101, 2000 steps, batches identical)

| arm | single change | held-out native | eval-aligned | syntax rate | complete | pass@1 v1/v3 |
|---|---|---|---|---|---|---|
| control_recheck | none | 2.9435 | 2.9498 | 0.0 | 0.0 | 0.0 / 0.0 |
| structural_segmentation | structural cuts | 2.9622 | 2.9754 | 0.0667 | 0.0 | 0.0 / 0.0 |
| indent_weighted | whitespace weight 3.0 | 3.0100 | 3.0249 | 0.0 | 0.0 | 0.0 / 0.0 |
| structural_indent_combined | both | 2.9918 | 2.9950 | 0.0667 | 0.0 | 0.0 / 0.0 |

Champion by pre-registered rule: `control_recheck`. Both structural-cut arms produced the only parse-clean generations (2/30), and the combined arm the best whitespace accuracy — genuine but sub-promotion signals, recorded without promotion.

### I — capacity and infrastructure

- 2B gate: `feasible=False — needs ~46.0 GB RAM (16 B/param × 1.5 activations headroom), available 4.8 GB`; 0 CUDA devices; `$0.00 spent, 0.0 GPU hours`.
- Virtual allocation recount: `slots_updated = 35292` (unchanged); allocated 500,000,000,000 virtual scalars remain architectural capacity, NOT trained parameters.

### J — subsystems (MoE + divergent reasoning, existing implementations only)

- MoE routing (12 real train batches, warm-6 dense-adapted model): mean entropy 3.4624 bits (uniform ≈ 3.585); JS first-vs-last batch joint-physical 0.0007–0.0014 near-uniform → routing remains input-dependent but near-uniform; no specialization claim.
- Divergent reasoning (2 v1 tasks, 3 hypotheses + repair, execution-verified): pass@1 0.0, pass@k 0.0, repair-recovered 0.0, mean convergence 1.0 (all candidates fail identically), 1440 tokens, 14.4 s. Plain fixed-decoding pass@1 0.0 on the same tasks.

### K — promotion gates (fixed before execution)

| gate | result |
|---|---|
| 1 structural improvement (syntax_valid_rate champion > control) | False — champion IS the control |
| 2 complete generation | False |
| 3 functional correctness (any pass@1 > 0, all arms) | False |
| 4 generalization (perturbed pass@1 > 0) | False |
| 5 capability promotion | False |
| held-out ≤ 3.3015 | True (control) — insufficient without Gate 5 |

**Decision: NOT PROMOTED**; rollback record points at `cortex-sft-stage12-chunked_control.pt`; stage-13 checkpoints are additive (separate files, gitignored directory), no stage-12 checkpoint was overwritten.

### L — independent audit (10 PASS / 0 FAIL)

Checks: A re-parse of all 60 stored generations matches reported parse rates; C exactly-once supervision re-derived for both segmentations; checkpoint sha256s recomputed from disk (4 stage-13 arms + rollback target); promotion decision re-derived from per-arm per-task rows (agrees); analyzer self-test re-executed; authoritative store recounted (35292); v1/v2/v3 fingerprints re-derived from raw tuples; benchmark files unchanged vs `402de46`; stage-10..12 artifacts and staged-6/12 checkpoints intact; all 13 stage-13 artifacts present and Fibonera-owned.

## 3. Teacher-forced vs free-running (before → after stage 13)

| metric | stage-12 measured | stage-13 measured |
|---|---|---|
| held-out loss (champion lineage) | 2.9498 | 2.9498 (control_recheck re-derived at stage-13 seed/horizon) |
| free-running parse rate | 0.0 (30 diagnostics) | 0.0–0.0667 across 2 subjects (30 diagnostics) |
| free-running pass@1 | 0.0 | 0.0 |
| chunked windows ending mid-construct | 49.84% | 49.84% fixed / **4.38% structural (repaired stream, opt-in)** |
| exactly-once completion supervision | stage-11 guarantee (fixed) | extended to structural segmentation, audit-verified |

## 4. Direct answers to the stage-13 questions

1. **Where do structural generations fail?** At specific model-emitted tokens (directly-introduced share 1.0/0.9286), dominated by indentation-transition failures (stage-12 share reproduces: 0.8/0.6).
2. **Do the training windows cut constructs?** 49.84% in fixed mode — reproduced; repaired to 4.38% with the opt-in structural segmentation, up to documented fallback cuts, with identical exactly-once supervision.
3. **Is structural knowledge learnable?** Partially: whitespace-class accuracy responds to objective weighting (+0.39…+0.78 pp) but held-out free-running validity does not yet improve enough to promote and held-out loss regresses in the weighted arms.
4. **Did the training stream change help?** Stream-level quality is better (clean boundaries 0.9562, guaranteed supervision); end capability unchanged (pass@1 = 0.0) — single-variable evidence at 2000 steps, honestly recorded, stage-11/chunked default retained.
5. **Implementation complete? executed? tested?** All workstreams A–L executed, not just implemented; 211 tests pass in the full suite; the independent audit reconstructs the major claims from persisted artifacts.
6. **pass@1 after stage 13?** **0.0 on every evaluated checkpoint** on the immutable v1/v3 suites with fixed decoding — unchanged from stage 12 and reported without adjustment.
7. **2B physical model trained?** No — hardware-BLOCKED (0 CUDA devices; ~46 GB required vs ~4.8 GB available; $0.00 spent, 0.0 GPU hours).
8. **What is the exact remaining gap?** The binding constraint remains that the 8.07M-parameter demo model cannot yet emit indentation-consistent complete programs from weak supervision (332 single-function records), and the 2B physical model remains untrained pending Fibonera-approved GPU infrastructure. The next measurable experiments are pre-registered in the artifacts: stage-14 feedback-training run (F), longer-horizon structural arm (C/H), and 2B training immediately upon GPU qualification (I).

## 5. Negative results preserved (do not silently drop)

- structural segmentation arm: held-out regression (+0.0256 eval-aligned) despite better stream;
- indentation weighting: held-out regression (+0.0751), minor whitespace gain;
- combined arm: held-out regression (+0.0452);
- EOT-gated exploratory decoding: 0/3 parses at stage-13 settings;
- divergent reasoning: pass@k 0.0 with convergence 1.0 (hypotheses do not diversify into distinct failures yet);
- MoE routing: near-uniform posteriors, JS 0.0007–0.0014 — no specialization claim.

## 6. Artifacts and provenance

All stage-13 artifacts under `artifacts/` are generated by `scripts/stage13_run.py` from the executed run at parent commit `402de46` and independently checked by `scripts/stage13_audit.py` (writes `stage13_independent_audit.json`). Every experiment block carries experiment_id, parent commit, dataset sha256, hardware description, pre-registered hypothesis, pre-registered acceptance criteria, and a decision. Checkpoints: `artifacts/runs/cortex-sft-stage10/cortex-sft-stage13-<arm>.pt` (additive; sha256 recorded in `stage13_experiment_matrix.json`).

## 7. Test execution

`python -m pytest -q` after all changes: **211 passed, 1 warning, 0 failed** (includes the new `tests/test_stage13.py`: 29 unit/integration checks with executed real-model cases and artifact-verification tests). `python -m pyflakes scripts/stage13_run.py scripts/stage13_audit.py`: clean.
