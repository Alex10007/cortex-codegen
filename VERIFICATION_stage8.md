# Stage 8 — Virtual Parameter Training Expansion + GPU Qualification: Verification

Owner: Fibonera. Baseline: Stage 7 at commit `26e484c` (lineage step 655,
`tx-stage7-joint`, immutable fingerprints v1 `74cc7607…`, v2 `64ea2c02…`).
Machine-readable record: `artifacts/stage8_report.json`. All numbers below
are reproduce from real persisted state — nothing is inferred from
allocation counts.

## Stage 8 validation matrix

| # | Requirement (per directive) | Method | Executed evidence | Result |
|---|---|---|---|---|
| 1 | Audit complete Stage-7 implementation | lineage/fingerprint/report checks (`stage_a_audit`) | stage-7 lineage step 655 ✓, v1+v2 fingerprints ✓, stage-7 report present ✓ → `preserved: true` | VERIFIED |
| 2 | Verify 500B allocations + storage semantics | per-file exact sizes, `st_blocks` physical bytes, manifest cross-check | 5,000 bank files, logical 2,000,000,000,000 B, physical blocks **143,032,320 B → sparse_file_ratio 0.9997**, committed rows 20,236, initialized 1,011,800,000 scalars | VERIFIED |
| 3 | Logical vs initialized vs physical blocks vs persisted values | `stage_b_storage_semantics` + digest sampling + npz scan | digests of sampled committed rows all match (500/500 banks sample); **386/386 sampled rows fully nonzero on disk**; `sampled_rows_nonzero == 1.0` fraction | VERIFIED |
| 4 | Expand initialization + real optimizer updates | expansion sweep: rows 1–3 in all 5,000 banks, 1 step per 8-bank group (1,875 steps executed, committed every 250) | initialized grew 250,000,000 → 1,011,800,000 scalars; 20,236 distinct slots carry persisted AdamW state; **104,485 slot-optimizer steps total (mean 5.16, max 220 per slot)** | EXECUTED |
| 5 | Scalar-level coverage, gradients, updates, distributions | `cortex/virtual/coverage.py` + participation log + `adam-*.npz` ground truth (npz exists iff VirtualAdamW ever stepped the slot = real gradients) | see strict accounting below; `invariants_hold: true` | VERIFIED |
| 6 | Optimizer-state consistency + recovery at larger coverage | commit → drop cache → reload; cross-process restart parity | per-slot steps/m/v restored from disk ✓ (all 3 probes); npz files: 20,236; cross-process restart: **bit-exact ref-batch parity 0.09663676470518112**, `virtual_ahead_of_checkpoint: 684` (honest: later verified commits), integrity OK | VERIFIED |
| 7 | Joint training vs identical-batch dense control | 120-step arms, same batches/lr/warm-start; dense control = untouched `CortexTransformer` | dense 1.6062 vs joint-random 1.9462 (fresh-expert cost **0.3400**); both joint arms on **virgin slots (`slots_fresh_fraction: 1.0`)** | EXECUTED |
| 8 | Fresh-expert cost / SVD init experiment | arm A virgin rows 1998 (seeded noise) vs arm B virgin rows 1997 (rank-16 SVD of trained attn-proj) | SVD-start loss 7.626 vs noise-start 7.287 (SVD did **not** reduce early loss at this scale); heldout after 120 steps: SVD **3.3975 vs noise 3.4422 (+0.045 better)** — mixed, honest result | EXECUTED |
| 9 | MoE routing/specialization, no weakening | JS divergence of mean router posteriors (algo vs text records), utilization, collapse checks | JS = 0.0 on all 4 layers → **no input-conditional specialization at demo scale** (routers near-uniform); utilization [1.0, 0.83, 0.5, 0.58]; no collapse | EXECUTED (honest null) |
| 10 | Divergent reasoning + execution-based repair | rewritten `cortex/evaluation/divergent.py`: 3 temps + repair round conditioned on real pytest stderr; every candidate sandbox-graded | 3 tasks × (3 divergent + repair): pass@1 0.0, pass@n 0.0, pass_via_repair 0.0, distinct-candidate rate 0.917, convergence 1.0 — demo model remains **execution-verified incapable**, no inflation | EXECUTED |
| 11 | Preserve Stage-5 corrected pipeline + immutable sets | fingerprint checks + no training-code changes to causal objective | v1/v2 fingerprints unchanged; completion-only masking/EOT untouched | PRESERVED |
| 12 | GPU qualification on real infra | FSDP2 + DDP spawns require actual CUDA devices; storage bandwidth measured on real bank files | measured `cuda=False, gpus=0` → **REFUSED-CPU-ONLY**; measured write BW 281.6 MB/s, 12802 MB/s read; estimated 478.9 s per 2B optimizer-step checkpoint tier (16 bytes/param × 2.056B) | GATED (honest) |

## Strict accounting (final, from disk truth)

| Class | Value |
|---|---|
| Allocated virtual parameters | 500,000,000,000 (exactly) |
| Initialized & persisted (committed rows × 50,000) | 1,011,800,000 (0.202% of capacity) |
| Activated (paged + persisted optimizer state) | 124,329,984 |
| Received gradients (npz-ground-truth ∪ participation) | 124,329,984 |
| Optimizer-updated (persisted `step_count > 0`) | 124,329,984 (20,236 slots) |
| Demonstrably trained | 124,329,984 |
| Slot optimizer steps total | 104,485 (mean 5.16 / slot, max 220) |
| Partitions with committed row | 5,000 / 5,000 (100%) |
| Rows per bank (this lineage) | up to 7 |
| Allocated − trained | 499,875,670,016 (**99.975% remains untrained**) |

`received_gradients == optimizer_updated == demonstrably_trained` is
inherent: `VirtualAdamW.step` increments `step_count` only after a real
gradient pass, and the persisted npz is the proof on disk. Value-change
magnitudes are quantified per step by the participation log
(update-distribution percentiles + histogram).

## Physical model status

- Demo-scale joint prototype (`CortexMoETransformer`, 11.6M params): trained
  here, checkpoints at `artifacts/runs/joint-502b-stage8/checkpoints`
  (`tx-stage8-expansion`, lineage step 1,875 … frontier 2,559).
- CORTEX_2B (2,056,012,800 params): **NOT trained** — CPU-only host. Full-scale
  start remains gated on measured GPU qualification (≥4 devices passing the
  FSDP2 + sharded-optimizer + recovery path).

## Defects found and fixed during Stage 8 (each regression-tested in `tests/test_stage8.py`)

1. `svd_slot_init` docstring claimed determinism but calibrated α via the
   global RNG → now uses a fixed local generator.
2. `js_divergence` returned NaN for zero-bearing probability vectors → masked
   zero bins.
3. `coverage.update_distribution` percentile dict had int-vs-string key
   mismatch (`KeyError: '5'`).
4. `VirtualParameterStore.commit` let `committed_step` regress when a later
   trainer committed at a lower step counter → watermark is now monotonic;
   trainers resume their step counter from the store watermark.
5. `restart_from_checkpoint` refused checkpoints when the store legitimately
   advanced past them; refusal for BEHIND (partial-commit) pairs kept strict,
   AHEAD pairs now accepted only via opt-in `allow_ahead` with the parity
   check intact (stage-7 tamper-refusal test still passes).
6. Divergent-repair round added to `cortex/evaluation/divergent.py`
   (execution-feedback conditioning), with unit-tested mechanics.
7. An earlier aborted run corrupted `committed_step` (3 vs 1875);
   `repair_manifest_watermark` restores it **only** after full digest + sha256
   verification, and the repair event itself is recorded in the report.

## Test results

- `tests/test_stage8.py`: 13/13 pass.
- Full suite: **125/125 pass** (was 112 before Stage 8).

## Honest limitations

- 0.202% of the 500B capacity initialized; 0.025% demonstrably trained.
  Allocation ≠ training; the numbers above are the measured truth.
- Stage-7 field `js_divergence` specialization is unproven at demo scale.
- Divergent pass rates remain 0.0 (demo-model capacity; repair round executes
  but cannot compensate).
- GPU qualification is REFUSED on this host; no claim of 2B or 502B-scale
  training is made.
- On marker-resumed runs the in-memory sweep loss trajectory is not retained;
  the trained state itself is fully persisted and measured by the npz scan,
  and loss trends reported for arms that executed in-run.
