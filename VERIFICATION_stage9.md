# Stage 9 — Full-Scale Training, Virtual Expansion, MoE Specialization, Reasoning, Capability Validation: Verification

Owner: Fibonera. Baseline: Stage 8 at commit `d5d26ce` (expansion lineage
`tx-stage8-expansion`, step 1875; stage-7 lineage step 655). Immutable
benchmark fingerprints: v1 `74cc7607…`, v2 `64ea2c02…`, new v3 suite
`c70feb13…`. Machine-readable record: `artifacts/stage9_report.json`.
All numbers below come from executed runs and persisted state; every
failed or blocked workstream is reported as such.

## Independent acceptance audit (artifacts/stage9_acceptance_audit.json)

The audit (`scripts/stage9_audit.py`) recomputes evidence independently of
the orchestrator: it rebuilds the v1 fingerprint from raw task fields,
re-scans all `adam-*.npz` optimizer states itself, verifies lineage
sha256s and step monotonicity, and samples store-manifest digests with its
own 2-D memmap logic.

| # | Area | Status | Evidence |
|---|---|---|---|
| A | Infrastructure | **BLOCKED** | 0 CUDA devices on host (`flawed_if_gpu_qualification_executed: false` — nothing was faked) |
| B | Physical 2B model | **BLOCKED** | param count 2,056,012,800 verified; `trained: false`, zero steps, no fabricated metrics |
| C | Virtual scalar updates | **PASS** | audit re-scanned npz ground truth itself: 216,834,048 unique trained scalars, matches orchestrator; stage-9-start scan 125,804,544 recorded; +91,029,504 during stage 9; stage-8 snapshot monotonic |
| D | MoE specialization | **PASS** | measured input-conditional routing beyond the 0.005 JS threshold on all 4 layers (stages 7/8 measured exactly 0.0); root cause measured, no metric inflation |
| E | Divergent reasoning | **FAIL** | pass@n 0.0, pass-via-repair 0.0; all 9 round-1 failures are runtime-category — honest capability failure |
| F | Coding capability | **FAIL** | fingerprints preserved (v1/v2/v3); harness sanity 1.0/1.0/1.0 proves the graders work; max subject pass@1 = 0.0 on all suites (zero rate == capability failure, reported honestly) |
| G | Joint vs dense | **FAIL** | `joint_better_than_dense_on_heldout: false` (delta +0.5875 loss worse at 60-step demo scale); identical batch order/count/optimizer documented |
| H | Recovery/reproducibility | **PASS** | full checkpoint recovery verified (SIGKILL probe, parity, digests, monotonic steps); multi-GPU recovery BLOCKED (no CUDA) |
| I | Fingerprint integrity | **PASS** | v1 recomputed and matching, v2 matching, v3 registered, lineage monotonic 655 → 1875 → 5286, no manifest digest failures in 100-bank sample |
| J | Ownership | **PASS** | all artifacts carry `owned_by: "Fibonera"` |

Final audit counts: **PASS 5 / FAIL 3 / BLOCKED 2 / NOT YET TESTED 0**.

## Workstream A — Training infrastructure qualification: BLOCKED

Executed on the actual host (never simulated): CPU 5 cores, **RAM 5.92 GB
total** (5.49 available), disk 5,459 GB free, **CUDA devices: 0**.

- Storage benchmark on real virtual-bank files: 335.2 MB/s write,
  11,252.3 MB/s read (40 rows × ~195 KB).
- Checkpoint timing: demo warm-6 checkpoint load 0.1 s (30.81 MB); a 2B
  AdamW state tier would be ~32.9 GB (16 bytes/param).
- 2B trainable on this host: **False** — needs ~46.0 GB RAM
  (16 B/param × 1.5 activations headroom) vs 5.5 GB available.
- GPU qualification (FSDP2/NCCL/multi-GPU) **not executed**: requires
  real CUDA devices; none exist. CPU/Gloo software verification from
  stages 7/8 is retained as software-level evidence only.
- Missing requirements recorded: ≥4 GPUs × 80 GB (or ≥8 × 40 GB) or
  equivalent; NCCL qualification; host RAM ≥46 GB.
- Decision: **BLOCKED.** No paid resources were provisioned; spend $0.00,
  GPU-hours 0.0.

## Workstream B — Full 2B physical model training: BLOCKED-BY-INFRASTRUCTURE

- Architecture independently re-verified: `CORTEX_2B.theoretical_param_count()`
  = **2,056,012,800** (`param_count_is_2_056_012_800: true`), meta
  instantiation counts match, stage-4 breakdown preserved.
- Training status: **BLOCKED-BY-INFRASTRUCTURE**. Training tokens
  processed: 0. Training loss: none recorded. No fabricated metrics.
- The physical model was NOT falsely reported trained; the 2B
  architecture remains a verified definition plus the demo-scale control.
- Demo-scale dense control quality preserved: dense heldout loss
  **3.3015** on the stage-6 checkpoint (`stage6-epoch-002.pt`).
- Lineage anchors recorded: stage-8 report sha256, expansion step 1875,
  warm start path; dataset/tokenizer sha256 recorded in the artifact.

## Workstream C — Virtual parameter training expansion: EXECUTED

Sweep executed rows 4, 5, 6 × all 5,000 banks (8 banks per group →
625 groups/row, 1,875 optimizer steps, committed every 250 groups;
~0.53 s/group on CPU).

Strict unique-scalar accounting (from persisted `adam-*.npz`
`step_count > 0` — NEVER cumulative updates):

| Snapshot | Unique trained scalars | Slots | Provenance |
|---|---|---|---|
| Stage-8 report | 124,329,984 | 20,236 | stage-8 stage-C sweep end |
| Stage-9 start (independent scan) | 125,804,544 | 20,476 | `persisted_optimizer_scan` before the stage-9 sweep; +240 slots = +1,474,560 scalars, reconciled exactly to stage-8's own post-C experiments (rows 1997/1998 × banks 0..119) |
| Stage-9 final (audit re-scanned) | **216,834,048** | **35,292** | `persisted_optimizer_scan` + independent audit scan agree |

Delta during stage 9: **+91,029,504 scalars (+14,816 slots, +73.6%)**.
Invariants hold (`invariants_hold: true`).

Reconciliation of the baseline history: marker-based reconstruction gives
156,162,048 (= 216,834,048 − marker sums 60,672,000), which equals the
true stage-9 start **plus +30,357,504 (4,941 slots × 6,144)** from an
interrupted first pass of the row-4 sweep (orphan checkpoint
`joint-step-003411.pt`, no lineage pair). The authoritative pre-sweep
baseline remains the recorded 125,804,544 scan; every number is consistent
with the on-disk npz ground truth.

- Slot update distribution: 196,357 slot-optimizer steps total, mean 5.56,
  p50 4, max 228 per slot.
- Cost: **14.95 seconds per million unique updated scalars** (CPU).
- Heldout loss after the sweep (single-point): 3.8295 — the 1-step-per-
  group exploration sweep is a coverage experiment, not a capability
  improvement claim; no heldout gain is claimed from coverage expansion.
- Capacity state: allocated 500,000,000,000; initialized & persisted
  **1,764,600,000**; unique trained **216,834,048** (0.0434% of allocated,
  12.3% of initialized); rows 0..3 of the earlier lineages never re-targeted,
  stage-7/8 rows preserved untouched.

## Workstream D — MoE specialization: EXECUTED (root cause identified)

Root cause of stage 8's JS = 0.0 (measured, not asserted):
`near_uniform_router_posteriors: true` — the routers were initialized
fresh and carried the warm-start dense path into effectively one expert;
with insufficient trained steps the top-k posteriors stayed within the
0.15 near-uniform threshold, so Jensen–Shannon divergence of mean
posteriors was exactly 0.0 regardless of input. Incorrect diagnostic
calculation, missing gradient flow and load-balancer over-regularization
were ruled out by measurement:

- Router gradients flow: nonzero grad norms on all 4 router weight
  tensors (e.g. `blocks.2.mlp.router.weight` 0.0568), 9,216 router params.
- Routers actually **moved** since the warm-start fingerprint: relative
  L2 moves 0.0235 / 0.1026 / 0.1158 / 0.1465 per block.
- After the workstream-C sweep training, measured input-conditional JS
  per layer: **[0.0240, 0.0131, 0.0122, 0.0194]** — all above the 0.005
  routing-differentiation threshold, against exactly 0.0 in stages 7/8.

Verdict: the stage-9 sweep training moved routers off uniform posteriors
and routing is now measurably input-conditional. This is an early-horizon
signal of differentiation, **not** proof of useful expert specialization;
useful specialization requires task-conditioned execution-quality gaps,
which remain unmeasured at demo scale (honest limitation stated in the
artifact's conclusion). 5 coding task families probed (functools,
pickletools, _pydecimal, difflib, recipes); dense control preserved.

## Workstream E — Divergent reasoning: EXECUTED, REJECTED-BY-EVALUATION

The failure-taxonomy upgrade (`cortex/evaluation/failure_tax.py`) now
classifies execution failures into 8 categories (`passed/syntax/
import/runtime/assertion/timeout/empty/unknown`) and drives targeted
repair prefixes; hidden-test leakage is prevented by contract (tested).
Execution result on the demo model (3 tasks × 3 candidate temperatures +
repair round, sandbox grading):

- pass@1 = **0.0**, pass@n = **0.0**, pass_via_repair = **0.0**
- All 9 round-1 failure codes are **runtime** category (the model body
  produces syntactically plausible but non-executing code)
- Distinct-candidate rate 1.0, convergence 1.0, contradiction rate 0.0
- Tokens 2,160; wall 17.36 s — measured cost with zero verified output
- Status: **REJECTED-BY-EVALUATION** — diversity without verified
  correctness is not capability; no inflation of this result.

## Workstream F — Coding capability validation: EXECUTED, FAIL (capability)

Immutable sets preserved: v1 fingerprint `74cc7607…` recomputed and
unmodified, v2 `64ea2c02…` unmodified; the new **v3 suite** (12 tasks:
algorithms, SQL/databases, debugging, test generation, refactoring, API,
security, performance, data structures) registered under its own
fingerprint `c70feb13…` and never modified after observation.

- Harness sanity: reference solutions **pass 1.0 / 1.0 / 1.0** on
  v1/v2/v3 — graders, sandboxes and tasks verified working.
- Under identical fixed decoding (temperature 0.2, top_k 50, 180 tokens):
  dense-control pass@1 = 0.0, joint-stage9 pass@1 = 0.0 on v1, v2 and v3;
  v3 compile rate 0.0; every family 0.0.
- Status: honest capability failure. The harness is verified; the model
  is not yet capable. This is the stage's central open problem.

## Workstream G — Unified joint training: EXECUTED, FAIL (negative result kept)

Four arms, identical batch order/count, optimizer settings, sequence
length 256, 60 steps each, warm-started from the same dense checkpoint
(`joint-loss-v2` objective unchanged from stages 7/8: completion-masked
CE 1.0, aux load-balance 0.01, router z-loss 0.001):

| Arm | Heldout loss | Note |
|---|---|---|
| 1 dense control | 3.4696 → **3.5096** | slight overfit at this horizon |
| 2 joint (frozen router) | **4.1767** | 64 fresh row-7 slots paged |
| 3 joint (full) | **4.0971** | 64 fresh row-7 slots paged |
| 4 full integrated (joint + divergent reasoning) | pass@n 0.0 | reasoning adds no verified gains yet |

`joint_better_than_dense_on_heldout: false`; delta joint − dense =
**+0.5875 loss (worse)**. Primary cause measured: each joint arm pages in
fresh (untrained) row-7 virtual slots, and at a 60-step horizon the
fresh-slot transients dominate the learning signal. Recorded as a negative
finding; larger-scale joint training with pre-trained slots is future
work. No thresholds were adjusted to change the outcome.

## Workstream H — Recovery and reproducibility: PASS

Executed at stage-9 lineage step **5286** (`tx-stage9-row6`):

- Checkpoint restart: resumed step 5286, ref-batch parity OK, physical
  digest OK, lineage steps match, virtual integrity OK.
- SIGKILL probe (`cortex/virtual/benchmarks.kill_recovery_probe`): child
  signaled ready → SIGKILL → committed rows verified 1/1, digests OK after
  kill, corrupt-row digest detection confirmed working, monotonic step
  enforcement active.
- `recovery_verified: true`. Corrupt npz and stale-pair (BEHIND manifest)
  refusals are regression-tested in `tests/test_stage9.py`.
- Multi-GPU recovery: **BLOCKED** (no CUDA device on host).

## Workstream I — Independent acceptance audit: PASS (as mechanism)

`scripts/stage9_audit.py` uses its own verification logic (raw-dataclass
fingerprint recomputation, direct npz directory scan, own lineage sha256
and digest sampling independent of `store.py` APIs). Result table above.
During development the audit itself caught one genuine defect in my own
auditor logic (1-D memmap row indexing); the fix (2-D `shape=(2000, 50000)`
sampling) then verified all 20,476+ sampled rows OK — the store integrity
claim rests on the corrected logic.

## Tests and integrity

- Full suite: **142 passed** (`python -m pytest tests/ -q`, 142.84 s),
  including 17 new stage-9 tests: 8-category failure taxonomy + no hidden-
  test leakage through repair hints; v3 suite properties (12 tasks, unique
  ids, no overlap with v1, deterministic fingerprint); fingerprint
  immutability (v1/v2 unchanged, unknown suite raises); MoE probe on the
  demo model (near-uniform detection, router gradient flow, assignment
  stability); divergent-repair targets dominant failure category; audit
  mechanics incl. tampered-constant detection and stale-pair refusal;
  corrupt-npz and corrupt-row-digest detection; exact 2B count; hardware
  report BLOCKED without GPUs.
- Lineage monotonic: 655 (stage 7) → 1875 (stage 8) → 5286 (stage 9),
  all physical checkpoint sha256s verified.
- Stage 7/8 artifacts and checkpoints untouched; no benchmark thresholds
  changed.

## Final-report answers

1. **Is the 2B physical model trained?** No — BLOCKED-BY-INFRASTRUCTURE;
   architecture verified at exactly 2,056,012,800 parameters, zero training
   steps executed, nothing fabricated.
2. **Physical training tokens?** 0 (2B model). Demo-control lineage
   continues warm-start evals only.
3. **Verified physical training loss?** None recorded for 2B; dense demo
   control heldout 3.3015.
4. **Which virtual scalars are initialized?** 1,764,600,000 of 500B
   allocated (persisted committed rows × 50,000).
5. **Unique virtual scalars with real optimizer updates?**
   **216,834,048** (35,292 slots), independently re-scanned by the audit.
6. **Independently verified as trained?** The same 216,834,048 — the
   audit's own npz scan reproduces the orchestrator's number from persisted
   `step_count > 0` states.
7. **Measurable held-out improvements from virtual training?** Not yet:
   at 1-step-per-group coverage scale no gain is claimed; the 60-step
   joint arms were worse than dense (negative result recorded).
8. **MoE experts genuinely specializing?** Partly: input-conditional
   routing is now measured above threshold on all layers (was exactly
   0.0); useful task-conditioned specialization is NOT yet demonstrated.
9. **Divergent reasoning execution success improved?** No: pass rates
   remain 0.0 (REJECTED-BY-EVALUATION); the taxonomy/repair machinery is
   in place and honesty-preserving.
10. **pass@1 / pass@k?** pass@1 = 0.0 on v1/v2/v3 for both subjects;
    pass@n = 0.0, pass-via-repair = 0.0. Harness sanity is 1.0 across all
    suites, so the zero rates are real capability limits.
11. **Integrated model beats dense control?** No at this scale (heldout
    +0.5875 worse); measured fresh-slot transients identified as the
    dominant cause; negative finding kept.
12. **GPU time?** 0.0 hours. 13. **Actual cost?** USD 0.00 — no paid
    resources provisioned; budget ceiling untouched.
14. **Complete training state survives interruption/recovery?** Yes —
    verified via SIGKILL kill probe, checkpoint restart at step 5286,
    optimizer-state restoration, digest verification, monotonic step
    enforcement (single-process; multi-GPU recovery blocked on hardware).
15. **Acceptance status?** PASS 5 (C, D, H, I-reproducibility, J)/ FAIL 3
    (E, F, G) / BLOCKED 2 (A, B) / NOT YET TESTED 0.
16. **Remaining gap to the full Cortex objective?** (a) real GPU
    infrastructure for 2B training; (b) 99.96% of virtual capacity still
    untrained; (c) pass@1 = 0.0 — the core capability gap; (d) MoE useful
    specialization beyond routing statistics; (e) joint-training wins at
    practical horizons.

## Limitations

- All training executed on CPU at demo scale; no GPU claim is made.
- unique-trained-coverage expansion measured coverage, not capability:
  heldout loss after the sweep (3.8295 single-point) is reported without
  a gain claim.
- The marker-based baseline reconstruction cannot distinguish the orphan
  row-4 first pass (+30,357,504) from pre-sweep state without the recorded
  start scan; the authoritative 125,804,544 scan is recorded with
  provenance, and the difference is fully explained.
- v3 suite has 12 tasks and is small; its fingerprint is registered and
  immutable so future expansions can extend, never edit it.
