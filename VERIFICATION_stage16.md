# VERIFICATION — Stage 16: Autoregressive Structural Learning, Complete-Program Generation, Execution-Grounded Correction and Generalizable Coding Capability

Owner: **Fibonera** · Repository: `Alex10007/cortex-codegen` · Branch: `vibe/cortex-core-b71b2c` (PR #1 head)
Parent commit (pre-registration): **`799a217`** (Stage 15 head) · All workstreams executed at this commit.

---

## 1. Status legend used throughout

Every claim below is tagged with exactly one status:

- **IMPLEMENTED** — source code exists in the repository.
- **EXECUTED** — the code was actually run in this environment.
- **TESTED** — covered by automated tests in `tests/test_stage16.py`.
- **TRAINED** — real optimizer updates were performed and checkpoints persisted.
- **EVALUATED** — measured under the fixed evaluation protocols.
- **INDEPENDENTLY VERIFIED** — re-derived by `scripts/stage16_audit.py` from persisted raw artifacts.
- **PROMOTED / NOT PROMOTED** — outcome of the Phase-11 gates.

## 2. Phase 1 — Baseline and failure reproduction (EXECUTED, INDEPENDENTLY VERIFIED)

Deliverable: `artifacts/stage16_baseline.json` (decision **PASS**).

| Checkpoint | sha256 (16) | held-out native | held-out eval-aligned | indent-transition acc. |
|---|---|---|---|---|
| stage15_structural_champion (`cortex-sft-stage14-structural_h6000`) | `c87907616e9f84fe` | 2.9074 | **2.9124** | 0.9608 |
| stage15_control (`cortex-sft-stage15-control_rep2`) | `e3a719da9d3dcbc8` | 2.9074 | 2.9250 | 0.9608 |
| stage12_champion (`cortex-sft-stage12-chunked_control`) | `9d8382efcd03c542` | 2.9435 | 2.9498 | 0.9412 |

All three match the Stage-15 reference values exactly. Free-running failure reproduction on the 8 Stage-15 structural probes under the exact original protocol:

- parse rate **0.0/8**, complete-program rate **0.0/8** (reproduces Stage 15's 0/8);
- **8/8** first failures at the first indentation transition;
- 6/8 probes self-emitted EOT (mean ~9–38 tokens generated);
- training-fragment continuation overlap **0.0** for every probe (generations are not reproduced training fragments — they are genuine free-running failures);
- per-probe token-level evidence persisted (first invalid index, lexer state, taxonomy `unexpected_indentation_transition`).

**Answer to report question 1 (why teacher-forced accuracy fails free-running):** the teacher-forced condition always supplies correct model-context prefixes; the free-running condition supplies the model's own (wrong) prefixes. Instrumented trajectories show the corruption begins at the very first context/item boundary the model itself creates (first divergence mean 1.12 tokens, max 2; first-invalid index 3–4 in probe trajectories).

## 3. Phase 2 — Autoregressive exposure investigation (EXECUTED, TRAINED, EVALUATED, INDEPENDENTLY VERIFIED)

Deliverable: `artifacts/stage16_autoregressive_exposure.json`. Pre-registered design: exposure rows built once from the shared warm-6 init at temperature 0.2 / top-k 50 / max 48 tokens; aligned rows require an **exact token-prefix of the verified reference completion**; diverged prefixes are skipped for aligned arms and truncated at first divergence for the error-conditioned arm; context masked −100; only verified continuation tokens supervised; per-row provenance persisted.

**Exposure dataset (real data, real measurements):**

- attempted 96 records → aligned rows **0**, early (error-conditioned) rows **24**;
- skips: 71 `does_not_fit_seq256`, 24 `prefix_diverged_truncated_to_early_row`, 1 `prefix_covers_full_completion`;
- mean first divergence among diverged prefixes: **1.12 tokens** (max 2) — the shared warm-6 init never reproduces a reference completion prefix; every attempt diverges within its first 3 tokens;
- mean prefix length 44.6 tokens.

**Arms (all 3000 steps, batch 4, lr 5e-5, AdamW betas (0.9, 0.98), warm-6 init, real optimizer updates and persisted checkpoints):**

| Arm | seed | exposure fraction | parse rate (8 probes) | held-out eval-aligned | grad-norm mean |
|---|---|---|---|---|---|
| control (rate zero) | 20261811 | 0.0 | 0.125 | 2.9038 | 4.7817 |
| exposure_const (0.5, aligned) | 20261812 | 0.0 | 0.125 | 2.9038 | 4.7817 |
| exposure_sched (0.25→0.75, PRIMARY) | 20261813 | 0.0 | 0.125 | 2.9038 | 4.7817 |
| exposure_sched_rep2 | 20261891 | 0.0 | 0.125 | 2.9038 | 4.7817 |
| exposure_early (const, error-conditioned) | 20261814 | **0.5123** | **0.0** | **2.7140** | 3.2709 |

Honest findings (negative results preserved):

1. **Aligned exposure had zero eligible data**: warm-6 never generated an exact reference prefix, so the three aligned arms degenerated to exact replicas of control (deterministic dataset iteration + fixed init + no rng consumption ⇒ bit-identical metrics, `exposure_step_frac=0.0`). This is a *measured* consequence of the pre-registered design, not a simulation.
2. **Error-conditioned early exposure changed optimization materially** (51% exposure steps, held-out 2.9038 → 2.7140, grad-norm 4.78 → 3.27) but **reduced free-running parse rate to 0.0**.
3. **Phase-2 acceptance FAIL**: best parse-rate delta over control = 0.0 (required ≥ +0.25). No exposure condition demonstrated the acceptance improvement.

## 4. Phase 3 — Structural transition learning (EXECUTED, EVALUATED)

Deliverable: `artifacts/stage16_structural_learning.json`.

- Teacher-forced indent-transition accuracy: control 0.9216, primary 0.9216 (identical — its arm had no exposure signal).
- First-transition divergence on 16 held-out examples: 3/16 contain a transition error for both; mean first-error position 46.7 tokens (max 89).
- Structural token accuracy (1280-token audit): overall 0.5742 (whitespace 0.8661, structure keyword 0.7288, delimiter 0.6667).
- Free-running: control 0.125, primary 0.125 ⇒ **`free_running_changed=false`**; per pre-registration, a teacher-forced-only change is **not accepted**.

Plain 3000-step training (any arm) moved free-running parse from baseline 0.0/8 to 1/8 — but no exposure variant beat the matched control.

## 5. Phase 4 — Complete-program training and context capacity (EXECUTED, TRAINED, EVALUATED)

Deliverable: `artifacts/stage16_complete_program_training.json`.

- Record-length distribution (306 train+heldout records): mean 464.7 tokens, max 2046; **29.4 % retained at SEQ 256, 71.9 % at 512, 93.5 % at 1024** — the 256-token context is the binding constraint.
- Structure-preserving segmentation (`structural_cut_summary`): 111/241 train records need cuts, 360 cuts total, **242 clean structural cuts vs 118 fallback hard cuts**.
- Complete-program curriculum arm (real 3000 steps; phase 1 = complete fitting records, switch at step 1500 to structural segmentation): held-out improved below control (−0.2676) but free-running parse rate **0.0** (−0.125 vs control 0.125). Curriculum leads to lower loss on lower-diversity data without free-running gains — negative result preserved.
- **CORTEX_S feasibility probe (real forward/backward, no simulation)**: live 41,997,312 params = declared `theoretical_param_count()`, seq width 511 × batch 2, 0.86 s/step, loss converging. A 42M-parameter model with max_seq_len 1024 is feasible on this host; the training arms that used SEQ 256 could be re-run at SEQ 512 on CORTEX_S-scale models in future stages.

## 6. Phase 5 — Objective and optimization ablation (EXECUTED, TRAINED, EVALUATED)

Deliverable: `artifacts/stage16_objective_ablation.json`.

- Gradient interference (real gradient vectors at the exposure_const endpoint on the first 8 exposure rows): loss values lm 1.0728 / structural-weighted 8.4082 / combined 4.7405; **cos(lm, structural-weighted) = −0.4395** — the structural weighting actively opposes the LM gradient; cos(lm, combined) = −0.232.
- Full 3000-step `exposure_weight` arm (indent-weight 3.0 real training): **diverged** — held-out 3.3015 → 7.9511, parse rate 0.0 (−0.125 vs exposure_const). Structural-token weighting at this scale is destabilizing. Negative result preserved.

## 7. Phase 6 — Execution-grounded training (EXECUTED, TRAINED, EVALUATED)

Deliverable: `artifacts/stage16_execution_feedback.json`.

- 12 real candidate generations from the primary arm against 12 execution-verified training records; verdict evidence (stdout/stderr/exit code/doctests) persisted per row. **1/12 passed** actual verification (`Context.copy_abs`, a training-pattern reproduction — classified honestly, rejected as contaminated/feedback material), **11/12 syntax_failure** (contamination count 0).
- 11 compact-evidence feedback rows built (provenance per row: instruction sha256, generated-source sha256, source-checkpoint sha256, error class, contamination flag 0, per-row `feedback_coverage` audit — coverage audit all target-bearing).
- Real 600-step feedback arm warm-started from the exposure_sched checkpoint: **target-bearing steps 600/600** (the Stage-14 coverage defect is fixed) — **but held-out eval-aligned 3.8992 = +0.3035 regression vs the Stage-15 feedback reference 3.5957**. Full target coverage alone does not make execution-feedback SFT a capability win at this scale. Stage-14's 102/600 negative control remains in history; the Stage-15 reference is cited, not modified.

## 8. Phase 7 — Autonomous generation and self-correction (EXECUTED, EVALUATED)

Deliverable: `artifacts/stage16_self_correction.json`.

- 11 tasks evaluated (v1 + t16); model-only direct successes **0**; 11 repaired with evidence-conditioned prompts, **22/22 revision rounds produced differing revisions**, regression rate 0.0, but **repair successes 0**. No repaired program passed its original tests.
- Model-only generation and tool-assisted repair are reported separately; neither has pass@1 > 0 at demo scale.

## 9. Phase 8 — Unseen generalization (EXECUTED, TESTED, EVALUATED)

Deliverable: `artifacts/stage16_generalization.json`.

- **t16 unseen suite (new, pre-registered, authored in this stage)**: 8 tasks covering functions, classes (+`pytest.raises`), data structures, algorithms, recursion+memoization, matrix rotation. Fingerprint `eb496a4a5ec6c341…`; contamination 0/8 vs train/v1/v3/t15.
- Harness sanity: all 8 reference solutions pass their own pytest suites in the real sandbox (**8/8, verified twice: pre-run and by the audit**).
- pass@1: v1 0.0 (n=3), v3 0.0 (n=12), t15-unseen 0.0 (n=8), t16-unseen 0.0 (n=8); perturbation suite 0/14 (seed base 20261900).
- pass@k (k=3 samples × temperatures 0.2/0.7/1.0 on the 8 t16 tasks): tasks with ≥1 pass = **0/8**.

## 10. Phase 9 — MoE and divergent reasoning (EXECUTED, EVALUATED)

Deliverable: `artifacts/stage16_moe_divergent.json` (negative results preserved).

- MoE routing: entropy 3.5015 bits, mean max prob 0.1371, JS(first vs last batch routing) ≤ 0.0017, expert weight cross-cosines ≈ 0.001–0.002 ⇒ routing is near-uniform and nearly static over batches; **no measurable specialization**.
- Divergent reasoning (2 v1 tasks × 3 hypotheses, 1 repair round): 6/6 round-1 failures (all runtime), pass@1 0.0, pass@n 0.0, pass-via-repair 0.0; distinct-candidate rate 0.875 but mean convergence 1.0. Verdict recorded in-artifact: diversity without execution-verified success is not capability. **Neither subsystem contributed measurable coding capability.**

## 11. Phase 10 — Physical and virtual parameter accounting (EXECUTED, INDEPENDENTLY VERIFIED)

Deliverable: `artifacts/stage16_capacity_audit.json`.

- Virtual store recount (real `persisted_optimizer_scan`): **slots_updated = 35292, unchanged** (must equal 35292 — asserted in `main()`), independently re-derived by the audit.
- 2B gate: 0 CUDA devices, 5.92 GB RAM (5.28 available); needs ~46 GB RAM ⇒ **BLOCKED — unchanged since Stage 9**. 2B architecture untouched; no demo-scale result reported as full-scale training.
- Checkpoint recovery: **9/9 real `torch.load` + `load_state_dict` restores** (8 stage-16 checkpoints + the stage-15 reference `cortex-sft-stage14-structural_h6000.pt`), independently re-executed by the audit.
- CORTEX_S reference: declared 41,997,312 params, max_seq_len 1024, verified live.
- spend: **$0.00**, **0.0 GPU hours**.

## 12. Phase 11 — Promotion gates (EXECUTED, INDEPENDENTLY VERIFIED)

Deliverable: `artifacts/stage16_promotion.json`. Candidate: `cortex-sft-stage16-exposure_sched.pt` (sha256 `04f2f38add2ed305…`). Gates were pre-registered in the script docstring before execution and re-derived by the audit.

| Gate | Criterion | Result |
|---|---|---|
| A — Reproducibility | primary delta ≥ +0.25 over control at primary seed AND rep2 | **FAIL** (delta 0.0 both) |
| B — Free-running structure | primary parse > 0 AND ≥ control + 0.25 | **FAIL** (0.125, delta 0.0) |
| C — Complete-program generation | primary capability complete-program rate > 0 | **FAIL** (0.0) |
| D — Functional correctness | v1/v3/t15 pass@1 > 0 (execution-verified) | **FAIL** (0.0) |
| E — Generalization | t16 pass@1 > 0 AND perturbed pass@1 > 0 | **FAIL** (0.0) |
| F — Self-correction | ≥ 1 repair success passing original tests | **FAIL** (0) |
| G — Regression control | held-out ≤ 3.3015 and ≤ stage-15 reference + 0.05 | **PASS** (2.8998 / 2.9038) |
| H — Independent audit | audit all PASS | **PASS** (13 PASS / 0 FAIL) |

**Decision: NOT PROMOTED (2/8 gates).** The Stage-12 champion `cortex-sft-stage12-chunked_control.pt` remains the production champion; rollback record persisted (baseline + Stage-15 reference checkpoints). No acceptance threshold was changed after seeing results.

## 13. Phase 12 — Independent audit (EXECUTED, PASSED)

Deliverable: `artifacts/stage16_independent_audit.json` — **13 PASS / 0 FAIL**, `all_pass=true` (consumed by Gate H). The auditor re-derives from raw persisted artifacts, not from runner summaries:

1. `fingerprints_immutable` — v1 `74cc760709f5cec0…`, v2 `64ea2c020004af92…`, v3 `c70feb13b061a35b…`, t15 `0d0bff384bc234b4…`, t16 `eb496a4a5ec6c341…` all recomputed from dataclass/source data.
2. `benchmark_files_untouched` — `git diff 799a217..HEAD` on benchmark files + clean working tree.
3. `prior_stage_artifacts_untouched` — no pre-Stage-16 artifact modified.
4. `experiment_registration` — all 11 experiment artifacts carry parent commit `799a217`, pre-registered hypothesis and acceptance criteria.
5. `exposure_rows_rederived` — all 24 early rows match exactly one same-name train record; masked context and supervised tokens re-verified token-by-token against the real dataset (a duplicate-record-name ambiguity in the first audit pass was found, root-caused and fixed in the audit — see §15).
6. `baseline_rederived` — per-probe parse re-derivation + checkpoint hash recomputation.
7. `capability_rederived` — per-suite rates recomputed from per-task rows; harness sanity re-verified.
8. `self_correction_rederived`, 9. `execution_feedback_rederived` (verdicts, contamination 0, lineage hashes, target-bearing 600).
10. `checkpoint_recovery_reexecuted` — 9/9 real reloads.
11. `authoritative_store_untouched` — 35292.
12. `stage_lineage_intact` — prior reports owned by Fibonera; Stage 13–15 reference checkpoints present.
13. `promotion_decision_rederived` — 8 gates re-derived independently; agreement with the reported NOT PROMOTED (2/8).

## 14. Answers to the directive's 12 report questions

1. **Why did teacher-forced accuracy fail to translate?** Supervision is on correct-context next tokens; free-running context is model-generated and diverges from reference context within 1–3 tokens of the first generated item (data: §3), so structurally-correct teacher-forced predictions are never exercised in the model's own context state.
2. **Did autoregressive exposure improve the first structural transition?** No. Aligned exposure had no eligible data (0 rows); error-conditioned exposure changed held-out (2.7140) but not the first-transition outcome (parse 0.0). Gate A/B delta = 0.0.
3. **Can Cortex generate complete programs?** Not yet: complete-program rate 0.0 on all 4 suites and all 8 probes.
4. **Can those programs execute successfully?** 1/12 feedback-pool candidates executed with correct results — but it is a training-pattern reproduction near `Context.copy_abs`, so it is not counted as unseen capability; pass@1 on all suites = 0.
5. **Does execution-grounded learning improve coding capability?** Coverage is fixed (600/600) but held-out regressed +0.3035 vs the Stage-15 reference. Negative result preserved.
6. **Can Cortex correct its own failures?** It produces differing revisions (22/22) but 0 pass the original tests.
7. **Does the improvement generalize to unseen tasks?** No improvement exists to generalize: t16 pass@1 = 0.0, perturbed pass@1 = 0.0, pass@3 (3 temperatures) 0/8 tasks.
8. **Did MoE or divergent reasoning improve outcomes?** No (§10); negative results preserved.
9. **Did virtual-parameter training contribute?** Store unchanged at 35292 independently verified scalars; no virtual updates were claimed this stage.
10. **Is full-scale 2B/502B training qualified?** No — 2B gate BLOCKED (~46 GB RAM needed vs 5.28 GB available, 0 CUDA). Unchanged since Stage 9.
11. **Did any candidate pass every gate?** No. Best candidate 2/8 (G, H); **NOT PROMOTED**.
12. **What exact bottleneck remains?** At 8.07M demo scale the model's distribution degrades on its own generated context: the 256-token training context excludes 70.6 % of records, structural-token weighting destabilizes training (gradient cosine −0.44), and verified-reference-prefix exposure is unachievable because the model never emits a verified-correct prefix. The evidence-backed next experiment is a CORTEX_S-scale (41.99 M params, max_seq_len 1024) rerun at SEQ 512 — feasibility was probed and demonstrated this stage (0.86 s/step real forward/backward).

## 15. Defects found and fixed during Stage 16 (narrow, tested)

1. `stage16_run.stage_10_capacity` composed the stage-15 reference recovery path with a doubled prefix (`cortex-sft-stage14-stage14-…`); fixed to `cortex-sft-<arm>` and re-executed (9/9 restores).
2. First audit pass exposed an ambiguous record-name resolution for 3 exposure rows (duplicate `name` entries in the train split with different instructions); the audit now resolves rows against **all** same-name train records by exact token-identity and reports ambiguity explicitly. Re-derivation: 24/24 rows valid.
3. Neither defect affects Stage 1–15 evidence; all prior artifacts, fingerprints and checkpoints verified untouched (`prior_stage_artifacts_untouched` PASS).

## 16. Test results (EXECUTED)

- `tests/test_stage16.py`: **28 passed / 0 failed** (real tokenizer, real pytest sandbox for the 8 t16 harness checks, real exposure-alignment property checks; artifact checks asserted against produced artifacts rather than skipped).
- Full repository suite: **291 passed / 0 failed** (includes all Stage 1–15 tests — no regression).
- `python -m py_compile` + `python -m pyflakes`: clean on `scripts/stage16_run.py`, `scripts/stage16_audit.py`, `tests/test_stage16.py`.

## 17. Delivery

- Single commit on `vibe/cortex-core-b71b2c` containing `scripts/stage16_run.py`, `scripts/stage16_audit.py`, `tests/test_stage16.py`, `VERIFICATION_stage16.md` and the 13 Stage-16 artifacts.
- Stage-16 checkpoints (`artifacts/runs/cortex-sft-stage10/cortex-sft-stage16-*.pt`) are gitignored per existing convention and recoverable from the recovery audit; nothing previous was deleted or overwritten.
- Compute: CPU-only, ~2 h real training/evaluation wall time, **$0.00**, **0.0 GPU hours**. 2B training remains BLOCKED.
- **Status: Stage 16 executed and independently audited; engineering investigation successful (root causes identified and tested); coding-capability improvement NOT PROMOTED (2/8 gates) — negative results preserved for the next stage.**


## Supplemental metadata-compliance audit (integration follow-up)

A separate, non-destructive audit was added as `scripts/stage16_metadata_audit.py`.
It checks the 11 workstream artifacts against the directive's per-artifact
metadata contract and writes `artifacts/stage16_metadata_compliance.json`.
It does not rewrite historical experiment artifacts or manufacture missing
seeds, configurations, checkpoint hashes, timestamps, or provenance.

The Stage 16 artifacts already contain experiment identifiers, parent/source
commit fields, dataset checksums, timestamps, decisions and raw workstream
records. However, metadata is not uniformly complete: some workstreams do not
persist a random seed, a full configuration, or a checkpoint hash. This is a
historical evidence gap, not something that can be safely repaired by guessing.
The supplemental report must be executed on the repository checkout to record
the exact per-artifact missing fields. The original Stage 16 report and
independent-audit artifact remain unchanged.

Stage 16 remains **NOT PROMOTED**. Stage 12 remains the retained champion.
