# Stage 7 — Unified 502B Trainable-Parameter Architecture: Verification

Owner: Fibonera. See `docs/OBJECTIVE_502B.md` for the mathematical
specification. Executed: see `artifacts/stage7_report.json` for the
machine-readable record of the full run (stages A–I).

## Validation matrix

| # | Requirement (per directive) | Method | Executed evidence | Result |
|---|---|---|---|---|
| 1 | Exact physical parameter count | `ModelConfig.theoretical_param_count` + live `actual_param_count` | dense 2B target: **2,056,012,800** (architecture; not trained on CPU host); demo dense: 8,071,296; demo MoE: 11,630,976 | VERIFIED |
| 2 | Exact virtual parameter count | `exact_virtual_count() == 5,000 banks × [2,000 × 50,000]` | `file_bytes == 100,000,000 × 4 × 5,000` exact, `allocated == 500,000,000,000`, sparse on disk (`st_blocks << logical`) | VERIFIED (allocated) |
| 3 | Numerical tensor integrity | sha256 per committed row, `verify_integrity()` | committed rows digest-checked OK; deliberate corruption detected (`corrupt_row_detected_by_digests: true`) | VERIFIED |
| 4 | Virtual parameter gradient correctness | real autograd through `slot.forward` + switch gate; AdamW step independently recomputed | `test_virtual_adamw_matches_independent_recomputation`: update == `-lr·g/(|g|+eps)` at step 1, `atol=1e-6` | VERIFIED |
| 5 | Actual optimizer-driven parameter changes | before/after comparison in a joint step | `test_joint_step_changes_both_systems`: physical params AND slot values change after one step | VERIFIED |
| 6 | Coordinated physical+virtual updates | one joint loss backward; separate optimizers over the same graph | participation log per step lists `virtual_slots_updated` + `virtual_grad_norms` > 0 | VERIFIED |
| 7 | Checkpoint save/reload integrity | physical sha256 + virtual digests + step cross-check + ref-batch loss parity | **exact** cross-process parity: `ref_batch_loss_after_restart == before` (bit-identical) | VERIFIED |
| 8 | Distributed training correctness | real 2-rank gloo probe (`scripts/virtual_shard_probe.py`) | `exit_codes [0, 0]`, all banks covered, **no ownership overlap**, digests exchanged | VERIFIED at probe scale |
| 9 | MoE routing and expert utilization | per-layer routing diagnostics | utilization [1.0, 0.83, 0.92, 0.83], `collapsed=false`, aux>0, z>0 | VERIFIED (demo scale) |
| 10 | Divergent reasoning evaluation | 3 hypotheses/task at temps (0.2, 0.7, 1.0), **each graded by hidden pytest execution in sandbox** | executed; current scores equal honest model baseline (see limitations) | EXECUTED |
| 11 | End-to-end coding benchmark results | immutable v1/v2 suites, free-running generation | fingerprints unchanged: v1 `74cc7607…`, v2 `64ea2c02…` | PRESERVED |
| 12 | Training coverage across virtual partitions | deterministic round-robin coverage sweep, 8 routes/step | full sweep targets all 5,000 partitions (625 complete groups); executed counts in report (`accounting.partitions_touched`) | EXECUTED |

## Strict accounting (from `virtual-manifest.json` + participation logs)

- `allocated` = 500,000,000,000
- `initialized` = committed rows × 50,000
- `activated` = paged into compute (cumulative distinct slots × slot size)
- `received_gradients` = activated slots with non-zero gradient
- `optimizer_updated` = slots with ≥1 VirtualAdamW step
- `demonstrably_trained` = `optimizer_updated` (never inflated)

**Allocated capacity is never reported as trained capacity.** Full-scale
500B training remains hardware-gated (Stage-I gate in the report).

## Tests

13 new tests in `tests/test_stage7.py`; full suite `pytest tests/`: **112 passed**.

## Key executed results (full run)

- Joint warm phase (30 steps, stage-6 warm start): train loss 6.46 → ~4.5
- Coverage sweep: 625 complete groups target 5,000/5,000 partitions; each
  targeted slot takes ≥1 real AdamW step (see `partitions_touched`)
- Controlled experiment: identical batches; dense control vs joint arm;
  early joint-arm loss higher because 8 fresh virtual-slot experts start
  from noise while the dense control reuses an already-SFT'd head
- Infrastructure: storage bw ~320 MB/s write / ~10 GB/s read on bank rows;
  paging cold-open ~0.5 ms; sparse slot AdamW step ~0.3 ms; 2-rank gloo
  sharding probe clean; SIGKILL recovery probe clean
- Stage-I gate: **REFUSED** full-scale 2B+500B training on this host
  (measured `cuda=false, gpus=0`); no simulated full-scale results

## Honest limitations

1. Physical 2B model: architecture exact and count-verified; NOT trained at
   2B scale on this host (no CUDA). FSDP2 path validated at smaller scale in
   Stage 6; full-scale training requires the qualified GPU pool.
2. Virtual 500B system: allocation, sparse paging, sharding, integrity,
   recovery, and joint optimization executed and verified at demonstrable
   scales (row/slot/group level, 2-rank distributed probe); 502B-scale
   distributed runs pending GPU infrastructure.
3. Divergent-reasoning pass rates equal the underlying demo-model baseline
   (currently 0.0 pass@1 on v1 tasks); the mechanism is executed and
   measured honestly, will only improve as the model improves.
4. Kernel mmap write-back can persist out-of-protocol unflushed mutations;
   the guarantee is therefore **detection** via digest verification (demonstrated),
   not crash-innocence of unflushed writes.
