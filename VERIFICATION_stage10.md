# Stage 10 — Capability Breakthrough, Full-Scale Training Qualification, Unified Intelligence Advancement: Verification

Owner: Fibonera. Baseline: Stage 9 at commit `a8c4e98` (lineage 655 → 1875 →
5286). Immutable benchmark fingerprints preserved and re-verified: v1
`74cc7607…`, v2 `64ea2c02…`, v3 `c70feb13…`. Spend this stage: **$0.00**,
GPU-hours: **0.0**. Machine-readable record:
`artifacts/stage10_report.json`; per-workstream artifacts as listed below.
All numbers below come from executed runs and persisted state; every failed
or blocked item is reported as such.

## Independent acceptance audit (artifacts/stage10_acceptance_audit.json)

The audit (`scripts/stage10_audit.py`) re-derives evidence independently of
the orchestrator: it rebuilds the v1 fingerprint from raw task fields,
cross-checks v2/v3 against the module tuple builders AND the immutable
constants, re-scans all `adam-*.npz` optimizer states with its own logic,
verifies lineage sha256s and monotonic step order, verifies via `git diff`
that no benchmark task file changed relative to `a8c4e98`, and cross-checks
the orchestrator's report against its per-workstream artifacts.

| Workstream | Status | Evidence |
|---|---|---|
| A GPU infrastructure | **BLOCKED** | decision BLOCKED; 0 CUDA devices; can-train-2b-locally `feasible: false` (needs ~46.0 GB RAM, 5.5 GB available); nothing fabricated |
| B 2B physical training | **BLOCKED** | `two_b_training_status.trained: false`; param count 2,056,012,800 re-verified; 0 tokens; $0 spent |
| C pass@1 root cause | **PASS** | C1–C5 executed on both subjects over v1+v3 (44 graded generations); C2 dataset audit persisted; two-arm remediation executed with honest NOT-PROMOTED decision |
| D virtual expansion | **PASS** | three arms executed at identical 300-step budget on disposable stores; authoritative store untouched (scan before == after); negative finding preserved |
| E MoE specialization | **FAIL** (honest) | routing differs by family (JS max 0.0219 > 0.005) but per-family loss spread 0.0247 < 0.05 → `specialization_accepted: false` |
| F divergent reasoning | **FAIL** (honest) | repair round delta == 0.0; pass@n 0.0 with and without repair; single-candidate pass@1 0.0 |
| G dense vs unified | **FAIL** (honest) | warm-slot joint arm does NOT beat dense on heldout (3.5792 vs 3.0845); fresh transient 3.4826 re-measured; identical batches/optimizer/seed |
| H recovery | **PASS** | intact restart parity at step 5286 with staged lineage store; truncated checkpoint, stale pairs, corrupt npz all detected; SIGKILL probe recovery_clean |
| I capability maturity | **PASS** | 6 capabilities classified from artifact evidence; max level 5 (checkpoint recovery); no level inflation |
| J reproducibility/integrity | **PASS** | v1/v2/v3 fingerprints recomputed and matching; lineage monotonic 655→1875→5286 with sha256s OK; independent recount 216,834,048 == stage-9 final; benchmark files untouched vs parent commit; all 10 artifacts owned by Fibonera |

Final audit counts: **PASS 5 / FAIL 3 / BLOCKED 2 / NOT YET TESTED 0**.
The three FAILs are capability verdicts on honestly executed experiments,
not harness defects — exactly the accounting this stage was chartered to
produce.

## Workstream A — GPU infrastructure qualification: BLOCKED

- Runtime-measured hardware: CPU 5 cores, RAM 5.92 GB total / 5.49
  available, disk 5,459 GB free, **CUDA devices 0**
  (`stage10_gpu_qualification.json`).
- `can_train_2b_locally`: **not feasible** — 2B (2,056,012,800 params)
  needs ~46.0 GB RAM (16 B/param × 1.5 activation headroom) vs 5.5 GB
  available.
- FSDP2/NCCL/multi-GPU qualification: not executed (no CUDA devices);
  record kept explicit, nothing simulated.
- Procurement recommendation left as ESTIMATE ONLY (4×80 GB ZeRO-3
  recommended minimum; 8×40 GB marginal; 2×80 GB insufficient vs stage-6
  activation measurements). No provider contacted, **no authorization,
  nothing provisioned, $0.00 spent**.
- Decision: **BLOCKED.**

## Workstream B — 2B physical model training: BLOCKED-BY-INFRASTRUCTURE

- Architecture re-verified: `CORTEX_2B.theoretical_param_count()` ==
  2,056,012,800 → `param_count_is_2_056_012_800: true`.
- Training-objective checklist re-executed at demo scale (2 real AdamW
  steps on the corrected causal next-token loop: loss 2.6199 → 2.4913);
  warm-checkpoint sha256 recorded; `scripts/train_2b_fsdp.py` launcher
  present as the GPU path.
- 2B training status: **BLOCKED-BY-INFRASTRUCTURE.** Training tokens
  processed: 0. Training loss: none. The 2B model was NOT reported as
  trained.

## Workstream C — pass@1 root cause: PASS (executed, root causes identified)

### C1 — Failure classification (actual generations, not assumed)

44 fixed-decoding generations (22 tasks × 2 subjects × v1+v3), all graded
by sandbox execution and diagnosed jointly from text + execution evidence
(`cortex/evaluation/failure_analysis.py`):

- **syntax_error: 27 / 30 failures** — dominant failure mode. Output is
  non-empty (non-empty rate 1.0 everywhere) but unparseable Python.
- truncation: 3.
- Zero degenerate repeats, zero undefined-dependency, zero
  signature-mismatch, zero passed.
- Greedy vs sampled decoding, longer horizon (320 tokens) and both prompt
  formats (eval-style header vs train-style) change nothing: parseable
  rate stays 0.0 → **the failure is not decoding or prompt-conditioning
  alone; it is what the model cannot yet generate.**

### C2 — Dataset investigation (`cortex/evaluation/dataset_audit.py`)

- Dataset: 332 records (241 train / 26 val / 65 heldout), splits
  deterministic (structural check + unit tests), 0 exact duplicates,
  2 near-dup pairs in the 120-sample, provenance present on all records.
- Contamination re-measured (not trusted from stored scores): max overlap
  0.2472, none above 0.5 → no leakage above threshold.
- **New quantitative finding: at the training seq_len 256 the fit filter
  admits only 96/332 records (28.92%); 236 records are silently dropped
  per epoch.** Mean instruction+completion length ~456 tokens. This is the
  measured data-side learning ceiling that survives stage-6 chunking
  (which fixed training *targets* for fitting records but never widened
  the active stream buffer).

### C3 — Model-learning investigation

- Teacher-forced token accuracy: dense control train 0.4059 / heldout
  0.4698; joint stage-9 train **0.9979** / heldout 0.6285. The joint
  subject has memorized its training distribution (train≈1.0, heldout
  much lower) — consistent with undertraining data starvation, not an
  inference defect.
- Memorization probe (8 records, 240 steps): train loss 2.6055 → 1.1926,
  but free-running generation reaches only **0.1419 mean reference-token
  overlap and 0.0 parseable rate**. The pipeline can drive loss down
  while generation quality stays broken — evidence that teacher-forced
  loss is not measuring generation competence at this scale.

### C4 — Controlled remediation (two arms, identical everything else)

- Arm `legacy_batching` (stage-6 make_batch, reproduces the 71% drop):
  400 steps, heldout 3.3015 → 3.1081; 80/100 steps in the final eval
  chunk had targets (the rest were silently skipped).
- Arm `chunked_batching` (stage-6 validated `make_batch_chunked` path):
  400 steps, heldout 3.3015 → 3.0881; **100/100 steps had targets** —
  chunking demonstrably eliminates the silent record drop.
- Immutable benchmarks after both arms: v1 pass@1 0.0, v3 pass@1 0.0.
- **Decision: NOT PROMOTED** — no immutable-benchmark improvement over the
  executed baseline; negative result + checkpoints recorded
  (`artifacts/runs/cortex-sft-stage10/cortex-sft-stage10-extended-{legacy,chunked}_batching.pt`,
  sha256s in the artifact).
- Root-cause synthesis (every claim evidence-linked): pass@1 = 0.0 is a
  **model-capacity + data-starvation limitation**. The grader is verified
  (reference-solution pass rate 1.0 on v1 and v3). The tokenizer rounds
  internally (same tokenizer trained and evaluated; TF accuracy 0.4–0.63).
  Inference/decoding are exonerated by the decode ablation. What remains:
  an 8M-parameter model that memorizes (train TF accuracy 0.998) but
  cannot yet produce syntactically valid code, plus a dataset where 71%
  of records never enter the training stream at seq-256. Remediation
  requires the 2B model on real GPU infrastructure and wider training
  data (chunked-stream datasets), not harness changes.

### C5 — Capability acceptance

- First pass@1 > 0 on immutable held-out tasks: **not achieved** (0.0
  before and after remediation). Reported honestly; regression-below-
  same-numbers preserved; no threshold changed.

## Workstream D — Virtual parameter training expansion: PASS (executed, negative capability finding)

All arms at an identical fixed 300-step budget, identical batches, on
disposable stores only (`stage10_virtual_coverage.json`):

| Arm | Held-out loss | Unique slots | Unique scalars | Steps/slot | Time |
|---|---|---|---|---|---|
| dense_only | **2.9398** | 0 | 0 | — | 38.1 s |
| joint_breadth (fresh slots, breadth expansion) | 2.9566 | 2,400 | 14,745,600 | 1 | 74.4 s |
| joint_depth (repeat-slot depth training) | 4.0483 | 8 | 49,152 | ~300 | 52.8 s |

- Breadth expansion (13,000× the scalar coverage of depth) **does not
  beat dense; depth-overfitting is sharply counterproductive** on held-out
  loss at demo scale.
- Authoritative store verified untouched (scan before == after:
  35,292 slots / 216,834,048 unique trained scalars preserved exactly).
- Conclusion: at 8M model scale, expansion of *virtual coverage* is not
  the lever for capability; the binding constraint is the physical model
  and its data stream. This preserves the stage-9 G finding under a
  cleaner arm design and extends it from routing statistics to held-out
  behavior.

## Workstream E — MoE specialization: FAIL (task-performance evidence absent)

- Task-conditioned routing measured over 6 stdlib families
  (`stage10_moe_specialization.json`): JS divergence IS input-conditional
  (max 0.0219 > 0.005 threshold, mean 0.0066) — routing differentiates by
  family, confirming and extending stage-9.
- Per-family completion loss: functools 0.0860 … pickletools 0.1107 —
  spread 0.0247 is BELOW the 0.05 quality gate → `quality_differs:
  false` → **`specialization_accepted: false`**.
- Honest note preserved in the artifact: at pass@1 = 0.0, execution-
  quality differences between experts cannot be measured at all; JS
  movement alone is NOT accepted as specialization success.

## Workstream F — Divergent reasoning: FAIL (repair produces no gain)

- Both arms executed with real sandbox grading (`stage10_divergent_reasoning.json`):
  3 hypotheses/task over 3 tasks, with and without the repair round.
- pass@n = 0.0 in both arms; pass-via-repair = 0.0; **repair delta = 0.0**
  — the reasoning extension cannot create correctness the base model
  lacks. Candidate distinctness 1.0 in both arms (diversity exists;
  solved-correctness does not).
- Single-candidate fixed decoding on the same tasks: pass@1 0.0, all 3
  failures runtime-category. Cost recorded: 18.4 s vs 12.5 s (the repair
  round is overhead with zero benefit at this capability level).

## Workstream G — Dense vs unified: FAIL (honest, fair, executed)

- Identical batches, optimizer, seeds, eval harness; joint warm slots are
  the warm-copied row-1 slots with real persisted AdamW state (`joint_warm_beats_dense: false`):
  dense 3.0845 vs warm-joint 3.5792 held-out; fresh-joint 3.4826.
- The joint arm's higher initial loss (7.23 vs 2.61) and slower held-out
  convergence re-measures and confirms the stage-9 transient; at demo
  scale, 200 steps do not amortize the virtual-route cold start.
- Authoritative store untouched (scan before == after). **No unified
  advantage is claimed at this scale; the comparison artifact records the
  negative verdict.**

## Workstream H — Recovery and reproducibility: PASS

- Intact restart: `restart_from_checkpoint` on the stage-9 checkpoint
  pair resumed at step 5286 with parity_ok + sha256 OK + coherent
  (non-stale) virtual state via a new lineage-staging helper
  (`_stage_store_for_lineage`) that builds a legitimate disposable pair
  from read-only authoritative state.
- Fault injections, all on disposable copies: truncated checkpoint
  (detected), stale pair with allow_ahead both ways (refused), corrupt
  optimizer npz (detected), SIGKILL mid-commit (recovery_clean, digests
  verify the committed row and flag the corrupt row).
- Stage-9 lineage re-verified from disk (step 5286, sha256 OK).

## Workstream I — Unified intelligence readiness: PASS (assessment)

Capability maturity matrix from artifact evidence only
(`stage10_capability_report.json`, new
`cortex/evaluation/capability_maturity.py`):

| Capability | Level | Meaning |
|---|---|---|
| checkpoint_recovery | 5 | independently-reproduces |
| virtual_parameter_training | 3 | learns-from-training (updates verified; no held-out gain) |
| moe_expert_specialization | 2 | executes-correctly (routing differs; no quality evidence) |
| code_generation_execution_verified | 2 | executes-correctly (harness verified; pass@1 = 0.0) |
| divergent_reasoning | 2 | executes-correctly (selects/repairs; zero success) |
| full_scale_2b_training | 1 | implemented (BLOCKED by hardware) |

- Gates are cumulative and blocked capabilities keep earned levels without
  unearned ones; the maturity classifier is unit-tested for exactly these
  semantics (including the blocked-does-not-erase rule).
- Roadmap recorded: first pass@1 > 0 (level 4) requires the 2B model + a
  training stream that actually contains the data (chunked windows) —
  both quantified by this stage — then MoE/virtual/astral claims gain
  measurable downstream bases.

## Workstream J — Independent acceptance audit: PASS (of the audit itself)

Executed and persisted as `stage10_acceptance_audit.json` (counts above).
The audit deliberately does not trust orchestrator helpers: fingerprints
recomputed two ways, npz ground truth re-scanned, benchmark files
diff-checked against `a8c4e98`, lineage sha256s and monotonic steps
re-verified, report-vs-artifact consistency checked with **zero issues**.

## Answers to the stage directive's key questions

1. **pass@1 across suites after all remediations:** 0.0 (v1/v2/v3
   fingerprints unchanged; harness sanity 1.0/1.0/1.0). Root cause
   classified with 44 diagnosed generations + dataset fit measurements
   (28.92% fit-rate ceiling) + memorization/TF-accuracy probes.
2. **Does the unified architecture beat dense?** No, at demo scale,
   measured fairly — recorded as a negative, not hidden.
3. **Do MoE experts specialize?** Routing differentiates by task family;
   task-quality differences do not (yet) — specialization NOT accepted.
4. **Does divergent reasoning improve execution success?** No; repair
   delta 0.0 — recorded.
5. **2B physical model trained?** No — hardware gate honest BLOCKED; $0
   spent.
6. **Virtual param coverage:** 216,834,048 unique trained scalars,
   independently recounted and unchanged (35,292 slots); new experiments
   ran 14.7M additional unique scalar updates on disposable stores and
   produced no held-out improvement — coverage growth is not capability.
7. **Infrastructure:** unchanged since stage 8 qualification; needs
   ≥4×80 GB (or 8×40 GB) GPUs + ≥46 GB host RAM.

## Tests and reproducibility

- New tests: `tests/test_stage10.py` (22 tests) — failure-diagnosis
  categories, severity ranking, dataset-audit fit/duplicate/determinism
  (with tamper detection), maturity-gate semantics, chunked-vs-legacy
  batching fact, authoritative-store read-only invariant, lineage-store
  staging + stale-guard coherence, audit tamper detection.
- Full suite: **164 passed** (`python -m pytest tests/ -q`, 228 s).
- Checkpoints produced (gitignored; sha256s persisted in
  `stage10_pass1_root_cause.json`):
  `cortex-sft-stage10-extended-legacy_batching.pt` `57f0048a…`,
  `cortex-sft-stage10-extended-chunked_batching.pt` `ff6ea148…`. Neither
  is promoted into any evaluation record; promotion requires benchmark
  gains, which did not occur.

## What is blocked / outstanding

- Full-scale 2B training + remaining GPU-only workstream items: BLOCKED
  on qualifying GPUs (4×80 GB recommended) — unchanged from stage 8/9,
  without which pass@1 stays 0.0 at 8M scale. The two materially new
  quantitative facts this stage contributes toward that unblock: (a) the
  corrected chunked stream must be default in GPU training configs (it
  excludes zero of the dataset vs 71% silently dropped); (b) the
  pre-registered target for the first capability increment is one pass@1
  > 0 on v1/v3 at fixed decoding.
- MoE specialization, divergent reasoning, and unified-advantage claims
  remain REJECTED-BY-EVALUATION and must be re-established after 2B
  training under the same immutable benchmarks.
