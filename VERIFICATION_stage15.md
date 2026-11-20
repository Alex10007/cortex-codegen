# VERIFICATION — Stage 15

**Project:** Cortex Codegen (Fibonera) · **Stage:** 15 · **Branch:** `vibe/cortex-core-b71b2c` (PR #1 head)
**Parent commit:** `318e601` (stage 14, asserted at run start) · **Source commit:** see `source_commit` in each artifact header (`_experiment_header15`, corrected local helper)
**Execution mode:** all workstreams below were actually executed on the real model, data, tokenizer, optimizer and the existing sandbox/evaluation subsystems. Nothing was simulated. $0.00 spent, 0 GPU-hours used.
**Result classification keys:** Implemented / Executed / Trained (really optimizer-updated) / Blocked by infrastructure / Rejected by evaluation.

---

## 1. Pre-registered hypotheses and status

Experiments were pre-registered in `scripts/stage15_run.py` headers **before** execution (`_experiment_header15` records exact IDs, seeds, decode settings and the pre-registered 6-gate promotion rule):

| # | Experiment | Pre-registered hypothesis | Status | Verdict |
|---|---|---|---|---|
| A | Structural-segmentation reproducibility: fresh 6000-step arm pair (fixed vs structural), seed `SEED_A_REP2`, all else identical, warm-start warm-6 | The stage-14 held-out improvement (structural 2.9124 vs control 2.9250) reproduces at the fresh seed | Executed, Trained | **PASS** — reproduces: rep-2 delta **−0.0126** (identical to rep-1); structural ≤ control in native AND eval-aligned; −0.0374 vs stage-13 champion |
| B | Structural consolidation: teacher-forced per-class structural accuracy, indentation-transition accuracy, 8 free-running structural probes | Held-out improvement corresponds to better structural prediction; free-running produces structural constructs | Executed | **Mixed** — teacher-forced: indent-transition accuracy **0.9608**, structural token accuracy 0.6188 overall / 0.7458 keyword / 0.7188 delimiter (matches stage-14 champion exactly); free-running: **0/8 probes parse, classes 0, nested defs 0**, persistent error `unexpected_indentation_transition` in all 8 — structural learning does NOT transfer to free-running generation |
| C | Feedback-SFT redesign: compact evidence (error class + last source-relevant error line, 160-char cap), per-row coverage audit BEFORE training, 600 steps warm-started from `stage14-structural_h6000` | Compact evidence restores 100% target-bearing coverage and removes the stage-14 held-out regression | Executed, Trained | **Coverage PASS / loss FAIL** — 600/600 target-bearing steps (stage-14: 102/600), 231,716 supervised tokens, training loss 0.7496 → 0.0038; **held-out regressed to 3.5862 native / 3.5957 aligned (+0.3783 vs the stage-14 negative control 3.2174)** — the stage-14 regression was NOT a coverage artifact |
| D | Execution-verified data generation: 3 independent samples/record at temperatures 0.2/0.7/1.0, real sandbox verification, stage-15 error taxonomy | Verified positive/negative execution evidence with zero contamination | Executed | **0/36 successes** (36× syntax_failure); contamination 0/36 |
| E | Generalization: 8 authored unseen tasks (execution-validated references, fingerprint `0d0bff38…`), immutable v1/v3 + unseen + 14 perturbation probes + unseen pass@k(3) | Capability on train-sample tasks transfers to unseen tasks | Executed | **0.0 everywhere** — unseen pass@1 0.0 (0/8), pass@3 0.0 (0 tasks with any pass), perturbed pass@1 0.0 (0/14), v1 0.0/3, v3 0.0/12; harness sanity 8/8 references real-verified |
| F | Self-correction: direct generation → real error evidence → model-repaired regeneration; each transition measured | Model-conditioned repair beats direct generation | Executed | **0/11 repairs** (0/11 direct as well); all 11 repairs re-executed, 0 passed; 11/11 revised candidates differed from the original |
| G | Optimization audit: REAL backward passes on identical model states for control data, compacted feedback data, structurally-weighted loss | Different data/objective conditions produce measurably different gradient signals | Executed | **Measured** — grad norms 10.8007 (control) / 13.2696 (feedback) / 30.6344 (indent-weighted 3.0); completion-only masking verified in all 3 conditions; no training arms run |
| H | Capacity + virtual accounting: hardware profile, 2B gate, store recount, checkpoint recovery (real torch.load + state_dict) | 2B gate opens only on qualified CUDA infrastructure | Executed | **BLOCKED** — 0 CUDA, needs ~46 GB vs 5.1 available; store recount `slots_updated = 35292` unchanged; all 3 stage-15 checkpoints restore |
| I | MoE + divergent-reasoning contribution: real `moe_probe` + ablated divergent eval | Diversity without execution-verified success is not capability | Executed | Measured — MoE routing near-uniform (mean entropy 3.456 bits, 4 layers) tied to untrained router; divergent eval 0.0 pass at index 0, pass@n 0.0, pass-via-repair 0.0 |
| J | Promotion gates (6 pre-registered, unchanged) | pass@1 = 0 ⇒ NOT PROMOTED | Executed | **NOT PROMOTED — 1/6 gates** (only gate 1: reproducible structural improvement); stage-12 champion retained, rollback intact |
| K | Independent audit (`scripts/stage15_audit.py`) | Audit re-derives results from persisted artifacts without trusting the orchestrator | Executed | **14 PASS / 0 FAIL** |

## 2. Acceptance matrix

| Workstream | What was done | Classification | Key measured result |
|---|---|---|---|
| A — Champion reproducibility | Two new 6000-step arms (1742 s + 2308 s real CPU training), seed `SEED_A_REP2`, identical config, held-out curves in both prompt styles every 1000 steps | Executed, Trained | structural 2.9124 vs control 2.9250 eval-aligned (delta −0.0126 = rep-1 delta); native equal (2.9074); **reproduces = true** |
| B — Structural consolidation | Teacher-forced measurements on the real held-out split + 8 trajectory-instrumented free-running probes classified by the 13-category taxonomy, no repair | Executed | indent transitions 0.9608; stops: 5× `eot`, 3× `max_new_tokens`; structural constructs 0/8; the model predicts structure well teacher-forced but generates none of it free-running |
| C — Feedback training redesign | 32 seeded records → 32 verified candidates (31 syntax_failure, 1 failed_reference_doctest) → compacted rows → per-row coverage audit → 600 optimizer steps → checkpoint `cortex-sft-stage15-feedback.pt` (sha256 recorded) → full eval | Executed, Trained | rows_fitting_seq 9/32 (compaction also improves raw fit 9 vs stage-14-style 7); all 9 fitting rows supervise the EXACT full solution; 600/600 target-bearing steps; **held-out 3.5957 (+0.3783 vs stage-14 negative control)** — regression refuted as coverage artifact; v1/v3 pass@1 0.0/0.0 |
| D — Execution data | 36 candidates (12 records × 3 temperatures), real sandbox verdicts, taxonomy classification, contamination tracking | Executed | 0 successes, 36 syntax_failure, **contamination 0/36**, records with any success 0/12 |
| E — Generalization | 8 authored unseen tasks with execution-validated references (real pytest on every reference), fixed-decoding eval across v1+v3+unseen, perturbation probes (14 execution-validated), pass@k(3) at 3 temperatures | Executed | model: unseen pass@1 0.0, unseen pass@3 0.0, perturbed 0.0/14, syntax 0.0 all suites |
| F — Self-correction | 11 tasks (3 v1 + 8 unseen): direct gen → real evidence → repair-gen → re-verification | Executed | repair_changed_candidate 11/11 (the model modifies output given its own errors), revised executions 11, **revised passes 0** — repair mechanics work, corrected output does not |
| G — Optimization quality | Real gradient diagnostics per data condition + masking invariants | Executed | see §1 row G; no promotion claim from gradients |
| H — Capacity/infrastructure | Profile + 2B gate + store recount + real reload of all 3 stage-15 checkpoints | Executed (blocked) | 5 cores, 5.92 GB RAM (5.09 available), 0 CUDA; 2B needs ~46 GB → BLOCKED; checkpoints `control_rep2`, `structural_rep2`, `feedback` all restore |
| I — Subsystems | Existing `moe_probe` on the real MoE subject; existing `divergent_eval` ablation vs plain generation | Executed | see §1 row I |
| J — Promotion | 6 pre-registered gates evaluated from measured per-task rows | Executed | 1/6, **NOT PROMOTED**, negative result preserved |
| K — Independent audit | Fingerprints (v1/v2/v3 from raw tuples, unseen from raw tuples), benchmark diff vs `318e601`, checkpoint re-hash (stage-15 + stage-14 byte-identity), real reload recovery, coverage/promotion/capability/execution/repair re-derivation, store recount | Executed | **14 PASS / 0 FAIL** (`artifacts/stage15_independent_audit.json`) |

## 3. Direct answers to the stage's five questions

1. **Is the Stage-14 structural-learning improvement reproducible?** YES. Two independent repetitions (stage-14 seed, stage-15 pre-registered seed) both give held-out delta exactly −0.0126 (2.9124 vs 2.9250), and structural ≤ control in both native and eval-aligned. Structural segmentation at a 6000-step horizon is a stable, reproducible improvement — the smoothest-verified learning result of the project.
2. **Can execution-feedback training improve the model when target coverage is correct?** NO — this is the stage's most important REFUTATION. With coverage fixed (600/600 target-bearing steps, exact full-solution supervision proven per row, 231,716 supervised tokens, training loss → 0.0038), held-out loss regressed MORE than the stage-14 under-covered arm (3.5957 vs 3.2174 vs champion 2.9124). The stage-14 regression was therefore never a coverage artifact: error-evidence-conditioned SFT data itself moves the model away from the held-out distribution. Negative result preserved as the reference for any future feedback experiment.
3. **Can Cortex generate more complete and executable programs?** No. Syntax-valid/complete/executable rates remain 0.0 across all suites and free-running structural probes; 68 fresh candidates (32 in C + 36 in D) produced exactly 0 executions passing verification (66 syntax failures, 2 doctest failures).
4. **Can Cortex correct its own execution failures?** The repair MECHANISM runs end-to-end (evidence built from the model's own real stderr; the model's revisions differed in 11/11 cases and were re-executed 11/11), but corrected outcomes: **0/11**. The stage-14 0/15 result reproduces under the redesigned evidence conditioning.
5. **Do the improvements transfer to previously unseen tasks?** No. On 8 execution-validated unseen tasks (plus 14 perturbation probes, plus 3-temperature pass@k): 0 successes, 0 syntax-valid outputs. Teacher-forced structural knowledge (0.9608 indent transitions) does not manifest in free-running generation.

## 4. Key mechanistic finding (carried forward)

The strongest structural signal of Stage 15: the model predicts indentation transitions at **0.9608** teacher-forced accuracy, and per-class structural accuracy equals the stage-14 champion exactly — yet ALL 8 free-running probes fail at the FIRST indent transition with `unexpected_indentation_transition`. The deficit is specifically in autoregressive continuation of learned structure (exposure/compounding of per-step predictions across long horizons), not in whether the structure was learned. Next measurable levers, in evidence order: (1) complete-program window training at SEQ=256 (only 9/32 records fit — the window truncates most records mid-construct), (2) scheduled-sampling-style exposure-curve experiments, (3) longer-context training infrastructure (blocked with 2B on GPU).

## 5. Verification evidence

- Run: `python scripts/stage15_run.py` from parent `318e601` (asserted), store invariant asserted, clean tree; resumable via persisted artifacts (`_resume` helper reused completed workstreams after an interrupted process).
- Auditor: `python scripts/stage15_audit.py` → `artifacts/stage15_independent_audit.json`, **14 PASS / 0 FAIL**.
- Tests: `python -m pytest tests/ -q` → **262 passed, 0 failed** (21 new stage-15 tests: classifier categories incl. real-sandbox syntax/doctest evidence, compaction caps and fallbacks, row surface form/provenance, coverage exact-match on real batches, unseen-suite reference execution 8/8, leakage = 0, registered constants).
- Lint: `python -m pyflakes` clean on `scripts/stage15_run.py`, `scripts/stage15_audit.py`, `tests/test_stage15.py`.
- Artifacts (all `owned_by: Fibonera`): `stage15_report.json`, `stage15_reproducibility.json`, `stage15_structural_consolidation.json`, `stage15_feedback_training.json`, `stage15_execution_data.json`, `stage15_generalization.json`, `stage15_self_correction.json`, `stage15_optimization_audit.json`, `stage15_capacity_audit.json`, `stage15_subsystem_contribution.json`, `stage15_promotion_audit.json`, `stage15_independent_audit.json`, `stage15_failure_taxonomy.json`.
- Checkpoints (additive, `artifacts/runs/cortex-sft-stage10/`): `cortex-sft-stage15-control_rep2.pt`, `cortex-sft-stage15-structural_rep2.pt`, `cortex-sft-stage15-feedback.pt` — sha256 recorded and re-verified by real reload; stage-14 checkpoints re-hashed byte-identical; rollback target `cortex-sft-stage14-structural_h6000.pt` intact.
- Immutable benchmarks: v1/v2/v3 fingerprints match immutable constants; benchmark files diff-empty vs `318e601`; unseen-suite fingerprint `0d0bff38…` recorded and recomputed from raw tuples; $0.00 spent, 0 GPU-hours.

## 6. Outstanding

- 2B physical training (~46 GB GPU-class infrastructure) — **BLOCKED**, unchanged since stage 9.
- First immutable-benchmark pass@1 > 0 — NOT PROMOTED (1/6 gates: only the reproducible-structural-learning gate passes).
- Prioritized next levers (from evidence): complete-program window training shapes at SEQ=256, exposure-sampling experiments aimed at the free-running indent-transition gap, and continued search for zero-regression feedback conditioning (stage-14 and stage-15 arms both regressed).
