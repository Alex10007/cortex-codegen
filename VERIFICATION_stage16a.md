# Stage 16A — Architecture Correctness Remediation

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Baseline:** `799a21758b1a8b43386b08bc9216f4a33ca987a8`  
**Remediation branch:** `fibonera/cortex-remediation-stage16a`  
**Scope:** Correctness fixes to virtual optimizer ownership, MoE routing, virtual slot accounting, and training-capacity qualification.

## Changes implemented

### 1. Shared virtual-slot optimizer lifecycle

- The same virtual `SlotState` is shared by a route across MoE layers.
- Gradients are now cleared once before the forward pass.
- Contributions from all layers accumulate into the shared leaf parameters.
- Each unique `(bank,row)` slot receives at most one `VirtualAdamW.step` per coordinated training step.
- Gradients are cleared after the virtual optimizer step.
- Conflicting distinct objects claiming the same persisted slot identity are rejected.

### 2. MoE gate correctness

- Expert outputs now use the renormalized selected top-k gate weights rather than the unnormalized softmax probabilities.
- Virtual parameter counting includes both LoRA factors, A and B.
- Repeated references to the same slot across layers are counted once.
- MoE accounting now reports virtual parameters and physical-plus-virtual totals separately.

### 3. Virtual slot compatibility and accounting

- Default virtual rank is capped by the exact row capacity: `floor(BANK_COLS / (2 * dim))`.
- Explicit ranks that exceed row capacity fail closed.
- Reusing a cached `(bank,row)` with incompatible A/B shapes now raises an error.
- Store commits persist each initialized slot's exact dimension, rank and scalar count.
- Legacy rows with optimizer state are backfilled from the persisted A/B optimizer tensor shapes.
- Legacy rows without shape metadata or optimizer state are reported as unknown instead of being counted as 50,000 initialized scalars.
- The persisted optimizer scan now reports the exact virtual allocation count.

### 4. Training-capacity qualification

- The memory estimator now includes a conservative quadratic attention-matrix estimate for the current explicit attention implementation.
- Hardware profiling records per-device total and currently free VRAM.
- The FSDP launcher enforces the validated world-size matrix, visible GPU count and per-device free-memory estimate.
- Unsupported world-size overrides are refused.
- If no configured cluster plan fits the current attention implementation, the launcher refuses before dataset packing or training.

## Regression tests added

`tests/test_stage16a_remediation.py` adds tests for:

- Renormalized MoE top-k routing.
- 2B virtual-rank capacity.
- Cached slot-shape mismatch rejection.
- A+B virtual scalar counting and deduplication.
- One optimizer update per shared virtual slot and gradient clearing.
- Quadratic attention memory estimation.
- Exact initialized-slot accounting.
- Honest unknown accounting for legacy rows without shape evidence.
- Refusal when the 2B model does not fit the current validated cluster matrix.

Existing Stage 8 coverage assertions were updated to expect exact LoRA slot scalars rather than full 50,000-scalar bank rows.

## Execution and verification status

- Source changes and regression tests: **implemented and committed on the remediation branch**.
- Local test execution: **not performed in this remote GitHub-only session**.
- GitHub Actions: workflow run [#18](https://github.com/Alex10007/cortex-codegen/actions/runs/44804499278) concluded failure, but the test job reported `runner_id=0` and an empty steps list; the job never exposed test execution or logs. This is an infrastructure/runner blockage, not evidence that the tests passed or failed.
- Therefore the new tests are **not yet verified as executed**.
- 2B training remains **BLOCKED**. The updated conservative estimate for batch 8 / sequence 8192 is approximately 2.29 TB aggregate activation-plus-state memory, or approximately 286 GB per device across 8-way sharding under the current explicit attention implementation. This estimate is not a measured run. An efficient attention implementation, smaller sequence/microbatch plan, or both are required before qualification.
- No training run was launched.
- No model checkpoint, benchmark, dataset, historical experiment artifact or virtual store was modified.
- Stage 12 champion remains unchanged.
- No PR was merged.

## Remaining remediation work

1. Harden sandbox isolation; default untrusted execution must not fall back to the host user/filesystem boundary.
2. Add API authentication, authorization, path confinement and request-level resource limits.
3. Integrate model-backed planning/patching into the engineering pipeline.
4. Establish a single checkpoint-aware model-loading/inference path for the selected dense or MoE architecture.
5. Complete exact parameter ledgers for dense and MoE physical architectures.
6. Integrate persistent memory and verified knowledge retrieval into generation context, with a separately measured learning path.
7. Correct knowledge-graph symbol identity for nested and same-named definitions.
8. Clarify Fibonera ownership and licensing metadata.
9. Restore executable CI and run the full test suite before declaring Stage 16A verified.
10. Continue with Stage 16 autoregressive structural learning only after these correctness and safety gates are independently verified.

**Decision:** Stage 16A source remediation is implemented. It is not yet test-verified and must not be treated as a promoted or production-ready build.
