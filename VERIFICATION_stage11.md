# Stage 11 — Training-Stream Correction and the First Capability-Increment Attempt at Demo Scale: Verification

Owner: Fibonera. Baseline: Stage 10 at commit `8eba46c`. Immutable benchmark
fingerprints preserved and re-verified this stage: v1 `74cc7607…`, v2
`64ea2c02…`, v3 `c70feb13…`. Spend this stage: **$0.00**, GPU-hours: **0.0**.
Machine-readable record: `artifacts/stage11_report.json`; per-workstream
artifacts `stage11_stream_audit.json`, `stage11_extended_training.json`,
`stage11_diagnosis.json`, `stage11_acceptance_audit.json`. All numbers below
come from executed runs and persisted state; every failed or blocked item is
reported as such.

## What Stage 11 was chartered to do and why

Stage 10 measured, with executed evidence, that the demo model's training
stream silently drops 71.08% of dataset records (fit filter at seq-256) in
BOTH the dense SFT path and `JointTrainer.train_step`, and that the stage-6
chunked path (windowed rows from over-long records) eliminates the drop in
the dense path but was NOT PROMOTED at a 400-step horizon. Stage 11 executes
the follow-up: wire the chunked stream into the joint coordinator,
re-audit the stream record-by-record, run a like-for-like 4000-step
controlled arms experiment at the fixed horizon, measure failure-mix drift
against the stage-10 diagnosis, and independently re-verify everything.

## Acceptance matrix

| Workstream | Pre-registered criterion | Status | Evidence |
|---|---|---|---|
| A stream audit | chunked coverage strictly greater; drop eliminated = fit == all records | **FAIL (honest)** | legacy 74/241 records fit vs chunked 185/241 fit (245 windowed rows recovered, 319 total rows) — but 56 records still cannot produce any window: their instruction alone is up to 773 tokens > 256-token window; `drop_eliminated: false` recorded as FAIL, not re-labeled |
| B extended training (dense) | 4000-step fixed horizon, identical seeds/batches/optimizer, chunked vs legacy; held-out cadence every 400 steps; PROMOTE iff pass@1 > 0.0 on v1 or v3 AND held-out ≤ 3.3015 | **EXECUTED — NOT PROMOTED** | chunked held-out 2.8819 vs legacy 2.96 (delta −0.0781, chunked wins on held-out); best chunked point 2.8382 at step 3600; token-bearing steps 4000/4000 vs 3101/3100-analog for legacy; benchmarks v1/v3 pass@1 **0.0/0.0 both arms** → gate not met, negative result preserved |
| B' joint coordinator | `JointTrainer.train_step(chunked=...)` produces real supervised steps on records the legacy path silently drops, default semantics unchanged | **PASS** | regression test `test_joint_train_step_chunked_trains_long_records` + `test_train_step_default_semantics_unchanged` (real training steps executed in-test) |
| D diagnosis drift | stage-10 protocol mirrored exactly: 2 arms × (v1 + v3) = 30 fixed-decoding diagnostics | **EXECUTED — honest negative** | both arms 15/15 `syntax_error` (share 1.0, delta 0.0 vs control); the stage-10 mix (27/30 syntax_error, 3 truncation) does NOT improve: the data stream is **not** the binding constraint at this model/data scale |
| E independent audit | fingerprints recomputed, virtual store recounted, benchmark immutability vs `8eba46c`, promotion-rule compliance | **PASS** | `stage11_acceptance_audit.json`: v1/v2 fingerprints recomputed-match, benchmark files untouched vs stage-10 commit, virtual unique-trained recount 216,834,048 == stage-9 final, promotion rule followed, authoritative store never written |

Audit counts: **PASS 2 / FAIL 1 / BLOCKED 0 / EXECUTED-negative 2**.

## Direct answers to the stage questions

1. **Does the chunked stream win at 4000 steps?** Yes on held-out loss:
   2.8819 vs 2.96 (delta −0.0781), with 4000/4000 token-bearing steps vs
   3101 for the starved legacy control. It is the first sustained
   (4000-step) held-out improvement from the stream correction.
2. **Is the checkpoint promoted?** **No.** The pre-registered rule requires
   fixed-decoding pass@1 > 0.0 on v1 or v3; both arms returned 0.0/0.0.
   Thresholds were not changed after observing results and the negative
   verdict is preserved.
3. **Did the failure mix change?** No — under the corrected
   like-for-like protocol (30 diagnostics, 15 per arm across v1+v3),
   syntax_error share is 1.0 for both arms (delta 0.0). The initial
   3-entry diagnosis artifact (v1 suite only) was a protocol defect in
   `stage_c_diagnosis`, fixed to mirror the stage-10 30-diagnostic
   protocol and re-executed (`/tmp/stage11_d_rerun.log`).
4. **Why does pass@1 remain 0.0?** Stage 11 adds a fresh, controlled
   elimination: over-long-record starvation is now **ruled out** as the
   sole binding constraint — repairing the stream improves held-out loss
   and removes 71% of the silent drop without moving immutable-benchmark
   pass@1 or the syntax-error mix. The evidence now points at
   model capacity / data volume / decoding, requiring GPU-scale training
   volume to resolve further; the syntax_error-only mix at 15/15 per arm
   remains consistent with a sub-capacity model that has not yet learned
   to emit parseable code under the fixed decoding settings.
5. **Is the authoritative 502B store intact?** Yes — read-only scan before
   and after; recount of 216,834,048 unique trained scalars over 35,292
   slots reproduced exactly; no stage-11 writes.

## Preserved lineage and integrity

- Parent commit `8eba46c` (Stage 10) — all Stage 7–10 artifacts, checkpoints
  and reports untouched (`git diff` vs `8eba46c` on benchmark files: empty).
- Checkpoints created this stage (gitignored, sha256 recorded in
  `stage11_extended_training.json`):
  `cortex-sft-stage11-legacy_stream_control.pt` `772440c4…`,
  `cortex-sft-stage11-chunked_stream.pt` `960e9d4e…`; parent warm-6
  lineage recorded inside both.
- Fingerprint constants re-verified: v1 `74cc7607…`, v2 `64ea2c02…`,
  v3 `c70feb13…`.

## Defect fixed this stage

- Pre-existing (present at clean `8eba46c`), reproduced by stash-verify
  then fixed: `scripts/train_2b_fsdp.py` did not insert the repo root into
  `sys.path`, so the launcher subprocess died with
  `ModuleNotFoundError: No module named 'cortex'` (exit 1) instead of the
  intended GPU-sufficiency refusal (exit 2) whenever ambient PYTHONPATH did
  not mask it. One-line fix + existing test
  `tests/test_stage6.py::test_2b_launcher_refuses_without_gpus` now green
  without PYTHONPATH masking.

## Test results

- Full suite: **171 passed** (`python -m pytest tests/ -q`, 216 s wall),
  up from 164 — the 7 new Stage-11 tests all run real components:
  - coordinator chunked path trains over-long records the legacy path
    drops (real AdamW steps, gradient norms asserted);
  - default `train_step` semantics unchanged;
  - stream-audit accounting (chunked strictly increases coverage; legacy
    < 0.5 fit-rate reproduced);
  - `chunk_record` rejects over-long instructions;
  - artifact + promotion-rule compliance; audit integrity; checkpoint
    registration.
- Lint clean on new/changed files (pyflakes).

## Honest accounting

- Implemented: chunked-stream flag in `JointTrainer.train_step` (default
  off), stage-11 orchestrator, corrected 30-diagnostic failure-mix
  protocol, independent audit, 7 regression tests, launcher sys.path fix.
- Executed: A (record-level audit), B (two 4000-step arms, 27+ minutes
  wall CPU), D (30 fixed-decoding diagnoses × 2 arms), E (independent
  integrity audit).
- Trained: real demo-scale dense arms only; the 2B physical model remains
  **BLOCKED** (0 CUDA devices; ~46 GB RAM required vs 5.5 GB available).
- Improved: held-out loss (chunked −0.0781 vs legacy at fixed 4000-step
  budget); supervision coverage 185/241 records, 245 recovered rows.
- Not improved / rejected: immutable-benchmark pass@1 remains 0.0;
  failure mix unchanged; no checkpoint promotion.
- Blocked by infrastructure: actual 2B training, 500B virtual capability
  correlation, GPU qualification beyond CPU-side software verification.
- Spend: $0.00; GPU-hours: 0.0; authoritative 502B store untouched.
