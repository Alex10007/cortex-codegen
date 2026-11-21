# VERIFICATION — Stage 14

**Project:** Cortex Codegen (Fibonera) · **Stage:** 14 · **Branch:** `vibe/cortex-core-b71b2c` (PR #1 head)
**Parent commit:** `c30ceb5` (stage 13, asserted at run start) · **Source commit:** see `source_commit` in each artifact header (`_experiment_header14`, corrected local helper)
**Execution mode:** all workstreams below were actually executed on the real model, data, tokenizer, optimizer and existing sandbox/evaluation subsystems. Nothing was simulated. $0.00 spent, 0 GPU-hours used.
**Result classification keys:** Implemented (code exists) / Executed (really ran) / Trained (really optimizer-updated) / Blocked by infrastructure / Rejected by evaluation.

---

## 1. Pre-registered hypotheses and status

Both experiments were pre-registered in `scripts/stage14_run.py` headers **before** execution (exact IDs, seeds, decode settings, and the unchanged 5-gate promotion rule imported from `scripts/stage13_run.py`):

| # | Experiment | Pre-registered hypothesis | Status | Verdict |
|---|---|---|---|---|
| B | Execution-feedback-derived SFT (48 train records, 48→1 passing candidates, 47 real-evidence rows, 600 steps warm-started from stage-12 champion `chunked_control`) | Conditioning SFT prompts on the model's OWN sandbox/doctest evidence from TRAIN-split records improves unconditional fixed-decoding pass@1 on immutable v1/v3 vs the stage-12 champion | Executed, Trained | **FAIL** — pass@1 0.0/0.0; held-out regressed +0.2676 |
| C | Longer-horizon single-variable arms (6000 steps each, control `fixed` vs `structural` segmentation, identical seeds/optimizer/batching, warm-start from warm-6) | The stage-13 structural-segmentation held-out regression (+0.0256 at 2000 steps) is a horizon artifact that will not persist at 6000 steps | Executed, Trained | **PASS** — regression does not persist (`regression_persists_at_6000: false`); held-out −0.0126 vs same-horizon control, −0.0374 vs stage-13 champion |
| D | Promotion attempt (5 gates unchanged from stage 13, champion selected across the 3 stage-14 candidates by pre-registered rule; perturbation probes rebuilt at seed_base 20261400) | pass@1 = 0 on every candidate implies NOT PROMOTED regardless of loss or syntax improvements | Executed | **NOT PROMOTED** — 0/5 gates |
| A | GPU re-qualification | 2B gate opens only on genuine qualified CUDA infrastructure | Executed | **BLOCKED** — 0 CUDA devices |
| E | Independent audit | audit must not trust the orchestrator's own numbers | Executed | **10 PASS / 0 FAIL** |

## 2. Acceptance matrix (follow-ups B–E from stage 13; A is the blocked-GPU standing re-check)

| Workstream | What was done | Classification | Key measured result |
|---|---|---|---|
| A — GPU re-qualification | Hardware profile, 2B feasibility gate, store recount | Executed; Blocked by infrastructure | 0 CUDA devices, 5 cores, 5.92 GB RAM (5.42 available); 2B gate needs ~46.0 GB → `two_b_gate_feasible_locally.feasible = false`; store recount `slots_updated = 35292` unchanged; decision **BLOCKED** |
| B — Execution-feedback training | 48 seeded train records → fixed-decoding generation → real sandbox execution + stdlib doctest verification against each record's reference doctests in the harvested module namespace → 47 evidence rows → 600 optimizer steps (AdamW lr 5e-5, betas 0.9/0.98, wd 0.01, clip 1.0) → checkpoint `cortex-sft-stage14-feedback.pt` → full eval | Executed, Trained | 47/48 candidates fail sandbox execution (executed_rate 0.0208); **1 candidate (`read_stringnl_noescape_pair`, 180 tokens) passed real execution + 2/2 reference doctests**; 47 feedback rows built, zero contamination (`benchmark_instruction_contamination_hits = 0`); training loss 2.3496 → 0.1084; **held-out worsened to 3.2048 native / 3.2174 eval-aligned (+0.2676 vs stage-13 champion 2.9498)**; pass@1 kept at 0.0/0.0 |
| B — repair-conditioned regeneration | Re-generate on 15 still-failing records with their own failure evidence prepended | Executed | Repair success rate **0.0** (0/15); evidence conditioning did not produce passing candidates |
| C — Horizon arms (6000 steps) | Two arms differing ONLY in `segmentation` (fixed vs structural); held-out curve at every 1000 steps in both prompt styles; capability eval + per-class structural accuracy | Executed, Trained | control: held-out 2.9074 native / 2.9250 aligned; structural: 2.9074 native / **2.9124 aligned**; findings: aligned delta −0.0126, native delta 0.0000, vs stage-13 champion −0.0374; training loss 3.8434→1.4083 (control) / 3.8783→1.4374 (structural); per-arm runtime 1768 s / 2287 s CPU |
| C — structural token accuracy (held-out, real tokenizer/data) | whitespace / structure-keyword / delimiter / other classes | Executed | feedback arm 0.5133 overall (whitespace 0.8780); control_h6000 **0.6047** (whitespace 0.8937, keyword 0.6780, delimiter 0.6771); structural_h6000 **0.6188** (whitespace 0.9055, keyword 0.7458, delimiter 0.7188) — the structural arm learns structural-token classes measurably better |
| D — Promotion audit | 3 candidates (feedback, control_h6000, structural_h6000); champion = structural_h6000 by pre-registered ranking; perturbation suite rebuilt (14 execution-validated probes); rollback record | Executed | Gate 1 (syntax 0.0 vs 0.0) FAIL; Gate 2 (complete 0.0 vs 0.0) FAIL; Gate 3 (any pass@1 > 0) **FAIL — pass@1 = 0.0/0.0 on all three candidates**; Gate 4 (perturbed pass@1 = 0.0 on 14 probes) FAIL; Gate 5 = Gate 3 FAIL; held-out gate ok (both ≤ 3.3015) but irrelevant without pass@1 → **NOT PROMOTED, stage-12 champion retained** |
| E — Independent audit | `scripts/stage14_audit.py`: fingerprints recomputed from raw tuples; benchmarks diffed vs `c30ceb5`; per-record feedback evidence sanity; capability pass@1/syntax/complete re-derived from per-task rows; all 3 stage-14 checkpoints + rollback target re-hashed; store recounted; analyzer selftest re-executed; stage-10..13 artifacts + warm-6/stage-12/stage-13 checkpoints verified; promotion decision re-derived | Executed | **10 PASS / 0 FAIL** (`artifacts/stage14_independent_audit.json`) |

## 3. Direct answers

1. **Did execution-feedback-derived training lift pass@1?** No. Trained (`cortex-sft-stage14-feedback.pt`, sha256 recorded), evaluated — v1 0.0, v3 0.0, syntax 0.0, complete 0.0. Held-out regressed +0.2676; 600 steps on 47 evidence rows is strongly over-fit (train loss 0.1084). Only 102/600 steps bore targets (long evidence prompts + solution can exceed the 256-token window) — a measured data limitation of feedback conditioning at SEQ=256.
2. **Does the stage-13 structural-segmentation regression survive a longer horizon?** No — the pre-registered hypothesis is CONFIRMED: at 6000 steps the structural arm does not regress against the same-horizon control (−0.0126 aligned, 0.0000 native) and is 0.0374 better than the stage-13 champion. Structural segmentation is now the best-known held-out config for a 6000-step horizon, and it also achieves the best per-class structural token accuracy (keyword 0.7458 vs 0.6780 control). This does **not** change the promotion decision: gates 3–5 require pass@1 > 0.
3. **Is there any first-ever verified pass@1?** One train-sample-scale success exists (1/48: `read_stringnl_noescape_pair` executed and passed 2/2 reference doctests under harvest-context verification) but zero on immutable v1/v3 and zero on 14 perturbed probes. Capability promotion remains **NOT PROMOTED**.
4. **Is the 2B model trained?** No. Still **BLOCKED**: 0 CUDA devices, ~46 GB needed vs 5.4 available, no procurement authorized, $0.00 spent, 0 GPU-hours.
5. **Virtual store integrity?** Authoritative store recounted twice (run start and report): `slots_updated = 35292`, unchanged; stage 14 training used updated dense-model copies only, never the authoritative store.
6. **Are results independently verified?** Yes — audit re-derives every headline metric from persisted rows and re-hashes every checkpoint; 10 PASS / 0 FAIL.

## 4. Known provenance note (preserved, not silently altered)

Stage-13 experiment-artifact headers record `parent_commit: a8c4e98` because the shared `_experiment_header` helper reads a stage-10 module global; their `source_commit` fields are correct. Per the immutable-artifact rule, stage-13 artifacts were NOT retroactively modified; stage-14 uses a corrected local helper (`_experiment_header14`) and the independent audit documents this discrepancy explicitly.

## 5. Verification evidence

- Run: `python scripts/stage14_run.py` (workstreams A→B→C→D, assertions checked parent commit and store invariant before starting).
- Auditor: `python scripts/stage14_audit.py` → `artifacts/stage14_independent_audit.json`, 10 PASS / 0 FAIL.
- Tests: `python -m pytest -q` → **229 passed** (18 new stage-14 tests: verifier correctness incl. real execution + doctests, wrong-algorithm rejection, harvest module-namespace reproduction on the real `functools.once` record, feedback-row surface form/target integrity/evidence cap, exact-string benchmark leakage = 0 on the train split, deterministic sampling, corrected header commit, artifact ownership/integrity).
- Lint: `python -m pyflakes` clean on all three new files.
- Artifacts (all `owned_by: Fibonera`): `stage14_gpu_qualification.json`, `stage14_feedback_training.json`, `stage14_horizon_arms.json`, `stage14_promotion_audit.json`, `stage14_independent_audit.json`, `stage14_report.json`; checkpoints additive in `artifacts/runs/cortex-sft-stage10/` (`cortex-sft-stage14-feedback.pt`, `cortex-sft-stage14-control_h6000.pt`, `cortex-sft-stage14-structural_h6000.pt`; all sha256 recorded in artifacts and re-verified by the audit).
- Immutable benchmarks: fingerprint checks v1/v2/v3 all match; benchmark files diff-empty vs `c30ceb5`; dataset sha256 `8c37f0db…` unchanged; negative results (feedback-arm regression, 0 repair successes, NOT PROMOTED 0/5) preserved.

## 6. Outstanding

- 2B physical training (46 GB GPU-class infrastructure) — BLOCKED, unchanged since stage 9.
- First immutable-benchmark pass@1 > 0 — not achieved; next measurable lever suggested by evidence: the 102/600 target-bearing step loss at SEQ=256 limits feedback conditioning (longer context or evidence compaction), and the horizon finding makes structural segmentation at 6000+ steps the reference configuration for future arms.
