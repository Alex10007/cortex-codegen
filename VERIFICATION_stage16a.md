# Stage 16A — Cortex Architecture-Correctness Remediation

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Baseline:** `799a217` (Stage 15)  
**Remediation branch:** `fibonera/cortex-remediation-stage16a`  
**Scope:** Correctness, model integration, verified context, execution isolation and training-capacity qualification.

## Executive status

Stage 16A has implemented a first remediation pass across the virtual optimizer, MoE routing, virtual store, model loading, project memory, API and hardware qualification.

**This is not yet a verified release.** The regression tests are committed but GitHub Actions has not executed them: the latest job reports `runner_id=0`, an empty `steps` list and no retrievable logs. No local repository runtime was available in this remote GitHub session.

No training run was launched. No historical benchmark, dataset, Stage 12 champion or checkpoint was promoted or intentionally modified. The remediation branch is separate; PR #1 remains unmerged.

## Implemented changes

### 1. Virtual optimizer ownership and update lifecycle

- The same `SlotState` is shared by a virtual route across MoE layers.
- Virtual gradients are cleared once before forward execution.
- Contributions from all layers accumulate into the shared A/B leaf parameters.
- Each unique `(bank,row)` slot receives at most one virtual optimizer step per global training step.
- Gradients are cleared after the virtual optimizer update.
- Distinct objects claiming the same persisted slot identity are rejected.
- The virtual rank is capped by the exact bank-row capacity; explicit oversized ranks fail before slot attachment.

### 2. MoE routing and parameter ledger

- Actual expert output uses renormalized top-k gate weights.
- Unattached virtual routes are masked from routing; the model refuses to run if fewer active experts than top-k remain.
- Load-balancing fractions are normalized by token count times top-k assignment count.
- Virtual parameter accounting includes both LoRA A and B factors and deduplicates shared slots.
- Dense-reference, physical expert, router, physical total and virtual slot parameters are reported separately.
- A static MoE parameter estimator reports the default 2B dense reference separately from the larger physical MoE construction. The default 2B MoE construction is approximately **5.833B physical parameters**, not 2.056B.

### 3. Virtual store integrity and accounting

- Each committed slot records exact dimension, rank and scalar count.
- Cached and persisted slot identities cannot be reinterpreted under a different shape.
- Legacy committed rows may be restored only when their optimizer tensors provide an exact A/B shape witness; ambiguous legacy rows fail closed.
- Orphan optimizer files are not counted as trained slots and are rejected during loading.
- Initialized, activated, gradient-receiving and optimizer-updated counts use unique `(bank,row)` identities and exact tensor shapes.
- Per-slot boolean change masks now record which individual A/B scalars have numerically changed. `demonstrably_trained` is derived from these masks, separately from optimizer-updated capacity.
- Legacy optimizer states without change masks are reported as unknown for demonstrated scalar change; the system no longer assumes every scalar in a slot changed merely because AdamW stepped.
- The coordinator now snapshots virtual values before each step and records exact moved-scalar counts, mean absolute delta and maximum absolute delta.
- Full 50,000-scalar bank rows are no longer counted as initialized LoRA slots.
- Optimizer state files are written to temporary files, flushed and atomically replaced.
- Multi-slot commits publish all row digests through one atomic manifest replacement.
- Committed row digests are verified before a slot is loaded.

### 4. Checkpoint-aware model integration

- `CortexAgent` now reads checkpoint architecture metadata and can load either the dense transformer or the MoE transformer.
- MoE checkpoint loading requires its joint lineage, verifies physical checkpoint hash, virtual-store manifest hash, exact committed-step equality and row integrity, then restores virtual routes.
- Joint physical checkpoints are written atomically; caller-supplied extra metadata cannot override reserved checkpoint hashes, step, virtual routes or Fibonera ownership fields.
- Generation executes under inference mode to avoid building autograd graphs.
- Project-context generation returns whether verified context was actually supplied.
- Knowledge inserts cannot self-assert verification, and replacing a previously verified record resets its trust status until re-verified.

### 5. Verified project memory and knowledge

- Project facts have an explicit `verified` field; new facts default to unverified and are excluded from generation context until explicitly verified.
- Existing fact databases are migrated with a default-unverified column.
- Generation context includes verified project facts, verified events and execution/test-verified knowledge patterns.
- Knowledge indexing uses qualified lexical scopes for nested functions and methods, reducing same-name symbol collisions.
- The API exposes authenticated generation with optional project context.
- This is retrieval-conditioned generation, not automatic model-weight training. Knowledge persistence does not by itself fine-tune Cortex.

### 6. API and execution isolation

- Protected API routes require a configured bearer token, checked with constant-time comparison.
- Repository operations are restricted to explicitly configured allowed roots.
- Request bodies have explicit length/range limits.
- API code-execution routes fail closed unless a working bubblewrap user/pid/network namespace capability probe succeeds.
- The sandbox uses a read-only runtime and disposable writable workspace when bubblewrap is available.
- The sandbox reports degraded isolation separately; API execution does not accept that fallback.
- The pipeline distinguishes baseline test validation from actual implementation: without applied edits it cannot claim the requested objective was implemented. Even after edits and passing repository tests, it requires an explicit task-specific `objective_verifier` callback before setting `verified=true`; the API does not currently supply a model-backed patcher or objective verifier.

This improves isolation, but it is not a certification of production security. Rate limiting, seccomp policy, cgroup-level resource isolation, deployment threat modeling and external penetration testing remain outstanding.

### 7. Training-capacity qualification

- The memory estimator now includes a conservative quadratic attention-matrix estimate for the current explicit `[batch,heads,seq,seq]` attention implementation.
- GPU profiling records total and free VRAM per device.
- The FSDP launcher validates the selected world size, visible device count and per-device free memory.
- World-size overrides cannot under-provision the computed minimum or bypass the supported matrix.
- The launcher refuses when no validated cluster plan fits.
- Cluster plans do not emit zero-sized microbatches.

For the dense 2B configuration at batch 8 and sequence length 8,192, the revised conservative estimate is approximately **2.288 TB aggregate**, or **286 GB per device across 8-way sharding**. This is an estimate, not a measured training run. The current explicit attention implementation therefore does not qualify on the existing 2/4/8-device, 40/80-GB planning matrix. Efficient attention and a measured memory plan are required before 2B training.

## Regression tests added

`tests/test_stage16a_remediation.py` contains 29 new tests covering:

- Normalized MoE routing and load balancing.
- Masking of unattached virtual routes.
- Virtual optimizer update uniqueness and gradient clearing.
- Virtual-rank capacity and slot-shape compatibility.
- Exact A+B virtual scalar counts and per-scalar numerical-change evidence.
- Exact MoE physical parameter accounting.
- Legacy slot restoration and orphan optimizer rejection.
- Atomic multi-slot manifest publication and row-digest checks.
- Quadratic attention memory estimates and GPU-plan refusal.
- MoE checkpoint loading with virtual lineage and rejection of virtual-store state ahead of its physical checkpoint.
- Verified project-memory retrieval.
- Inference-mode generation and accurate context-use metadata.

Existing Stage 8 accounting assertions were updated to use exact slot scalars rather than full bank-row capacity. API and pipeline tests were extended for authenticated, fail-closed execution and truthful implementation status.

## Verification status

- Source changes: **implemented on remediation branch**.
- Regression tests: **written; not yet executed in this session**.
- GitHub Actions: [latest run](https://github.com/Alex10007/cortex-codegen/actions/runs/44809511172) concluded failure before a runner started. The test job has `runner_id=0`, no steps and no logs. This is not evidence of either a passing or failing test suite.
- 2B physical training: **BLOCKED** pending efficient attention and real hardware qualification.
- 500B virtual training: infrastructure is corrected in source, but the corrected optimizer/store path has not yet been re-executed at scale.
- MoE physical model: parameter count is now reported honestly; the default 2B MoE model is larger than the 2.056B dense reference.
- No Stage 16 coding-capability experiment has been run.
- No new candidate is promoted; Stage 12 champion remains the champion.
- **Historical accounting clarification:** Stage 7–15 optimizer-state files predate per-scalar change masks. Their slot/step counts prove optimizer updates, but do not prove that every scalar in each slot changed. The previously reported 216,834,048 virtual scalars should therefore be interpreted as optimizer-updated slot capacity; exact per-scalar numerical-change evidence for those legacy runs is unavailable. The new accounting reports those legacy slots as unknown rather than retroactively rewriting Stage 7–15 artifacts.
- No merge to `main` was performed.

## Remaining gaps

1. Execute all 29 dedicated Stage 16A tests, the four new patch-set regressions, and the full existing suite on a functioning runner; fix any failures.
2. Add efficient attention (e.g. verified SDPA/FlashAttention path with correct causal and padding-mask semantics) and benchmark peak GPU memory.
3. Add explicit device placement for virtual slot parameters and optimizer state; the current `JointTrainer` remains CPU-oriented and is not yet a qualified GPU joint-training path.
4. Build a qualified distributed MoE + virtual training path; the existing 2B FSDP script trains the dense transformer and does not establish full 502B joint distributed training.
5. Integrate a model-backed patcher into the engineering pipeline; current pipeline still accepts an injected patcher and does not autonomously implement changes through the API.
6. Complete production security controls, including multi-host rate limiting and reverse-proxy deployment policy.
7. Improve semantic retrieval beyond hashed n-gram vectors and evaluate retrieval quality.
8. Clarify the repository's Apache-2.0 metadata versus Fibonera's intended IP/licensing policy. No license change was made.
9. Reconcile the default `main` branch/Hugging Face deployment with the Fibonera Cortex implementation only after owner review.
10. Resume Stage 16 autoregressive structural learning only after Stage 16A tests and correctness gates pass.

**Decision:** Stage 16A remediation is implemented in source but remains **NOT VERIFIED**. It must not be treated as a production-ready or promoted Cortex build.


## Stage 16A.2 — patch-set publication safety

A focused pipeline correctness increment was added after the initial Stage 16A remediation:

- Patcher output is now fully preflighted before any repository file is written. Invalid later entries therefore cannot leave earlier files partially modified.
- Absolute paths, parent traversal, protected metadata paths, symlink traversal, duplicate normalized targets, non-regular targets, and oversized/non-text content are rejected before publication.
- Every replacement is staged in its target directory, flushed and fsynced, then published with per-file atomic replacement.
- If a later replacement fails, already-published files are restored from their original bytes (or removed if newly created). Existing file permissions are preserved; newly created files receive mode 0644.
- Four regression tests cover all-entry preflight, symlink rejection, multi-file replacement/mode preservation, and rollback after an injected publication failure.

This is rollback-capable multi-file publication, **not** a filesystem-wide atomic transaction. It does not close a same-user concurrent filesystem race between preflight and publication; stronger dirfd/openat-style path handling remains a hardening opportunity.

The branch head at this report update is expected to include the Stage 16A.2 commits; verify the live ref before treating this report as current. GitHub Actions run [44809511172](https://github.com/Alex10007/cortex-codegen/actions/runs/44809511172) again failed before runner startup (`runner_id=0`, empty steps). None of the new tests have an execution result from that run.

**Stage 16A.2 status: implemented, tests written, NOT VERIFIED.** No model training, benchmark promotion, PR merge, or main-branch change was performed.


## Stage 16A.3 — shared-state API rate limiting

Implemented on the remediation branch:

- Added `cortex/api/rate_limit.py`, a SQLite-backed sliding-window limiter. Each request uses a short `BEGIN IMMEDIATE` transaction, so multiple API worker processes sharing the same local database cannot independently exceed the same bucket budget.
- Database rows store SHA-256 bucket identifiers rather than bearer tokens, client addresses, or raw principal strings. The database file is restricted to mode 0600 where supported; the limiter fails closed with HTTP 503 if its storage cannot be used.
- Added a configurable database path (`CORTEX_RATE_LIMIT_DB`) and configurable per-minute budgets to `create_app`.
- Authenticated requests share a token-principal budget (default 120/minute); failed authentication attempts share a direct peer-address budget (default 20/minute); expensive generation, execution, evaluation, repository-analysis and security-scan routes receive an additional budget (default 20/minute).
- HTTP 429 responses include `Retry-After`. `/health` remains public and exempt. Forwarded client-IP headers are intentionally not trusted; behind a reverse proxy, failed-auth throttling therefore keys on the socket peer unless a trusted-proxy design is added.
- Added 8 rate-limit tests for shared database state, concurrent workers, authenticated budgets, failed-auth throttling, expensive-route budgets, identity hashing, sliding-window expiry and principal isolation.

**Scope limits:** this is process-shared on one host, not a multi-host distributed limiter. A deployment spanning multiple hosts needs a shared backend such as Redis or an equivalent atomic service. The API currently uses one configured bearer token, so authenticated budgets are per configured token rather than per end-user account. The local rate-limit database must be placed on a reliable local filesystem; do not assume SQLite locking semantics on arbitrary network filesystems.

**Verification:** source and tests are committed, but none of these tests have been executed in this environment. GitHub Actions run [44810168757](https://github.com/Alex10007/cortex-codegen/actions/runs/44810168757) again failed before runner startup (`runner_id=0`, no steps). This provides no evidence of a passing or failing test suite.

**Stage 16A.3 status: IMPLEMENTED, NOT VERIFIED.** No model training, benchmark promotion, PR merge, or main-branch change was performed.


## Stage 16A.4 — sandbox workspace and process-tree hardening

Implemented on the remediation branch:

- Replaced path-following workspace copies with descriptor-relative traversal. Source entries are opened with `O_NOFOLLOW`; symlinks and non-regular files (including FIFOs and device nodes) are rejected rather than copied or dereferenced.
- The copy path fails closed on platforms that do not expose the required POSIX no-follow directory primitives. It enforces a 100,000-file ceiling and a 2 GiB aggregate copy ceiling.
- Replaced `subprocess.run` timeout handling with explicit process-group management on POSIX. Each run starts a new session; on wall-clock timeout, the entire process group receives SIGKILL before output collection, preventing ordinary descendants from surviving after the direct child is killed.
- Added adversarial tests for external file symlinks, external directory symlinks, special-file rejection, and descendant termination after timeout.

Security scope and verification limits:

- These changes harden workspace ingestion and ordinary process-tree timeout handling; they do not constitute a complete sandbox certification.
- Bubblewrap namespace/mount behavior, seccomp policy, cgroup-level aggregate resource limits, output-volume limits, and adversarial race testing still require a functioning Linux runner with bubblewrap.
- The descriptor-relative copy implementation deliberately requires POSIX `O_NOFOLLOW`; unsupported platforms reject workspace execution rather than silently falling back to unsafe path traversal.
- The new tests are committed as regression specifications but have not been executed in this remote GitHub editing session. A GitHub Actions job that never obtains a runner is not test evidence.

**Stage 16A.4 status: IMPLEMENTED, NOT VERIFIED.** No model training, benchmark promotion, PR merge, or main-branch change was performed.


## Stage 16A.5 — bounded output and descendant lifecycle

Implemented:

- Replaced unbounded `communicate()` capture with concurrent stdout/stderr drainers.
- Retained output is capped at 4 MiB per stream. Readers continue draining after the cap, preventing pipe backpressure from deadlocking the child while bounding parent-process memory use.
- Execution metadata reports the per-stream cap and whether either stream was truncated. Returned text includes an explicit truncation marker.
- On timeout, the entire POSIX process group is killed. The same cleanup now runs after a normal direct-child exit, preventing ordinary background descendants from continuing or holding inherited output pipes open.
- Added tests for simultaneous multi-megabyte stdout/stderr and background descendants surviving a normal parent exit.

Limitations:

- This is process-group cleanup, not a cgroup/container guarantee against a deliberately daemonized process that escapes its session or namespace. Full bubblewrap qualification and cgroup-level resource enforcement remain outstanding.
- The tests have been added but have not been executed in this remote editing session. GitHub Actions runner allocation has not provided a usable test result.
- No claim is made that bubblewrap mounts, network namespace, seccomp, or host-filesystem denial have been independently validated.

**Stage 16A.5 status: IMPLEMENTED, NOT VERIFIED.** No model training, benchmark promotion, PR merge, or main-branch change was performed.


## Stage 16A.7 — cgroup v2 aggregate resource enforcement

The strict sandbox execution path now requires an administrator-delegated cgroup v2 subtree configured with `CORTEX_CGROUP_ROOT`. Each strict run configures aggregate CPU quota, memory ceiling, zero swap, group-wide OOM behavior, and a process/task ceiling. The launcher is held behind a pipe gate until its PID has been moved into the cgroup; untrusted code is not executed before attachment. Cleanup uses both process-group termination and cgroup-wide `cgroup.kill`, then waits for `cgroup.events populated 0` before deleting the cgroup.

The behavioral qualification now attempts to exceed the aggregate process cap and launches descendants in separate sessions. The cgroup must report empty after cleanup. API workers cache the full qualification result once per process; each execution still gets a fresh cgroup and fail-closed setup.

**Stage 16A.7 status: IMPLEMENTED, NOT RUNTIME-VERIFIED.** Regression tests are committed but have not executed because GitHub Actions jobs continue to fail before runner allocation (`runner_id=0`, empty steps and no logs). Numeric CPU/memory pressure tests, I/O bandwidth/IOPS controls, seccomp, privileged cgroup migration and kernel escape resistance remain outstanding. See [VERIFICATION_stage16a7.md](VERIFICATION_stage16a7.md). No training, benchmark promotion, PR merge or `main` change was performed.


## Stage 16A.8 — Kernel-Evidence Resource Qualification

- Added cgroup snapshots for `cpu.stat`, `memory.events`, `pids.events`, and `memory.current` before cgroup removal; missing/malformed evidence fails closed.
- Strict sandbox CPU quota is configurable for controlled qualification. CPU, memory and process-limit qualification now requires kernel event counters, not merely configuration or failed child-spawn attempts.
- Added controlled CPU-bound and concurrent aggregate-memory workloads to behavioral qualification. These require live delegated Linux cgroup v2 + Bubblewrap and remain **NOT RUNTIME-VERIFIED** until run on that host.
- Seccomp remains **NOT ENABLED / NOT CERTIFIED**; no unverified default-deny policy or speculative dependency was introduced. Disk I/O remains `NOT_CONFIGURED`.


## Stage 16A.9 — CI trigger and queue hygiene

- Corrected the `pull_request.branches` filter to include `vibe/cortex-core-b71b2c`, the actual base branch of PR #2. Previously the workflow only listened for pull requests targeting `main`; PR #2 therefore depended on branch-push events and did not receive the intended pull-request validation trigger.
- Added per-source-branch workflow concurrency with stale-run cancellation so rapid commit bursts do not queue one hosted runner per commit.
- Added explicit read-only repository-token permissions for CI and bounded test/deploy job timeouts.
- Rechecked Actions run history: the latest run and its retry both failed within seconds, the test job had no steps and no retrievable logs. The available evidence places the failure before workflow steps/runner execution; it does **not** establish a Python/test failure and does not identify a definitive GitHub-side cause.
- A runner retry was issued; it again failed before steps. This is an external execution-capacity/service/configuration blocker, not something repository source can conclusively repair. A functioning GitHub-hosted or owner-managed self-hosted runner is still required to execute the test suite.
- No test is marked passed by this increment. Stage 16A remains **NOT VERIFIED** until CI executes. No training, benchmark promotion, checkpoint/data mutation, merge, or `main` update was performed.


## Stage 16A.10 — Strict sandbox launch and evidence correctness

- Static review found that the cgroup-gated bootstrap source encoded literal backslash-n text instead of line breaks. The launcher source is now generated from explicit source lines and has a compile-based regression test.
- Strict sandbox mode previously applied RLIMIT_NPROC at the same threshold as cgroup pids.max. That could make an unprivileged process fail at RLIMIT_NPROC before the kernel cgroup counter recorded pids.events:max. Strict mode now leaves process-count enforcement to the delegated cgroup by raising the soft RLIMIT_NPROC to its existing hard ceiling; degraded mode retains its per-process limit.
- Added focused regression tests for launcher syntax and both strict/degraded process-limit policies.
- **Tests not executed**: GitHub Actions still fails before job steps. Stage 16A remains NOT VERIFIED and live cgroup/Bubblewrap qualification remains outstanding.
- No training, benchmark promotion, checkpoint/data mutation, PR merge, or main update occurred.


## Stage 16A.10 — cgroup input and evidence-schema hardening

**Commit increment:** source and regression-test changes on `fibonera/cortex-remediation-stage16a`.

- Cgroup resource limits now require positive integers; booleans and floats are rejected rather than being accepted through Python's integer coercion rules.
- Kernel counter snapshots now require the specific evidence fields used by qualification: `nr_throttled` and `throttled_usec` in `cpu.stat`, `oom`, `oom_kill`, and `oom_group_kill` in `memory.events`, and `max` in `pids.events`.
- Negative `memory.current` values are rejected.
- Added regression coverage for invalid CPU quota/period values and incomplete CPU counter evidence.

**Verification:** changes are committed, but the new tests have **not been executed**. GitHub Actions still reports completed failed jobs with no step records and no logs (`steps: null`); this does not establish a test failure in the repository or a passing test suite. Live delegated-cgroup qualification remains outstanding.

**Limitations unchanged:** no seccomp certification, no disk I/O enforcement, no kernel escape-resistance claim, and no live CPU/memory/pids stress qualification. No training, benchmark promotion, checkpoint/dataset mutation, merge, or `main` update.


## Stage 16A.11 — cgroup configuration and evidence parsing boundaries

- Enforced the documented cgroup v2 `cpu.max` period range of 1,000–1,000,000 microseconds. Positive-integer validation remains in place for all resource-limit inputs.
- Counter parsing now rejects duplicate keys rather than silently accepting the last value.
- `memory.current` must be a non-negative ASCII decimal integer; signed or otherwise malformed values fail closed.
- Added regression cases for out-of-range CPU periods, duplicate counter keys, and malformed memory-current evidence.

**Verification:** source and test changes committed to the remediation branch. These tests have not been executed; CI jobs still fail before any runner steps or logs are available. This increment therefore makes no test-pass or live-sandbox qualification claim.

**Scope unchanged:** strict sandbox requires real Bubblewrap plus administrator-delegated cgroup v2; seccomp and disk I/O controls remain unconfigured; live CPU, memory and process-limit qualification is outstanding. No model training, benchmark promotion, checkpoint/dataset mutation, merge, or `main` update.


## Stage 16A.12 — kernel-compatible CPU quota validation and fixture correction

- Added the cgroup v2 CFS bandwidth minimum: `cpu_quota_us` must be at least 1,000 microseconds. The existing period range remains 1,000–1,000,000 microseconds. Quotas greater than the period remain permitted because they represent more than one CPU-equivalent of aggregate bandwidth.
- Added a regression test for a sub-millisecond quota.
- Corrected the Stage 16A.10/16A.11 test fixture strings so their embedded counter records contain actual line separators. The previous double-escaped fixture would have tested malformed-line rejection rather than the intended missing-counter/duplicate-counter cases.

**Verification:** committed source and test changes only. No test suite has been executed and no live cgroup host has been qualified. GitHub Actions has previously failed before runner steps/logs; the new commits must still receive an actual executable test run.

**No changes** to training, model weights, checkpoints, benchmark champion, datasets, PR merge status, or `main`.


## Stage 16A.13 — unified sandbox lifecycle cleanup

The strict and degraded execution paths now use one cleanup routine after process launch:

- Process-group termination is attempted on every normal exit, timeout, and exceptional path; non-POSIX execution terminates a still-running direct child.
- A strict run issues cgroup-wide termination, waits for the direct child to be reaped, joins output readers, verifies the cgroup is empty, captures kernel counters, and attempts cgroup removal.
- Cleanup failures are no longer silently discarded. They are surfaced as an execution failure; if another exception already triggered cleanup, the original exception remains primary and receives a cleanup-failure note.
- Output-reader startup failures and unexpected process-wait failures now enter the same cleanup path.
- Cgroup creation was moved until after command/isolation planning, reducing the interval in which a setup exception could leave an unused cgroup behind.
- The launch-failure and cgroup-attachment-failure paths now attempt explicit cgroup cleanup and preserve cleanup errors as exception notes rather than swallowing them.
- Added regressions for successful process/cgroup cleanup, cgroup-kill failure reporting after direct-child reaping, and injected failure starting the second output reader.

**Verification status:** implementation and regression tests are committed to the remediation branch. The tests have not been executed in this session. CI must run them on an actual runner; source inspection is not a test pass. The strict sandbox remains NOT RUNTIME-VERIFIED until delegated cgroup v2 + Bubblewrap qualification passes on a supported Linux host.

**Known limits:** this increment does not certify seccomp, disk I/O limits, host-kernel escape resistance, or deliberate privileged cgroup migration. It also does not replace live CPU/memory/process-limit qualification. No model training, checkpoint/data changes, benchmark promotion, PR merge, or main update occurred.


## Stage 16A.14 — runner retry and pre-launch cgroup cleanup

Follow-up inspection found one remaining resource-lifecycle edge: after a strict-run cgroup was created, failures while allocating the launch gate pipe or preparing the gated bootstrap command could occur before the process-lifecycle cleanup routine was entered. The cgroup could therefore remain allocated despite no child process having started.

Changes:
- Launch gate preparation is now guarded. If pipe allocation or bootstrap-command preparation fails, the code closes any gate descriptors already allocated, kills the new cgroup, waits for it to become empty, and attempts removal.
- Cleanup errors are attached to the primary setup exception rather than silently discarded.
- The earlier Stage 16A.13 cleanup flow remains unchanged for successfully launched processes.

Runner investigation:
- Re-ran the failed GitHub Actions test job for run 44839586322 (job 136644367994). GitHub accepted the retry, but the retried job again completed with failure, no step summaries, and no retrievable logs. The deploy job was skipped.
- The branch's configured workflow uses ubuntu-latest; this connector exposes no repository Actions settings, hosted-runner diagnostics, or runner registration controls. The available evidence establishes only that the job failed before any step was reported. It does not identify the underlying account, billing, policy, or GitHub runner-service cause.
- The latest commit has not yet produced a verifiable Python test execution. No tests are claimed as passed.

**Status:** pre-launch cleanup edge addressed in source; executable verification remains BLOCKED. Strict sandbox remains NOT RUNTIME-VERIFIED. The next useful action requires a functioning GitHub-hosted runner or an administrator-provisioned, registered self-hosted Linux runner. Do not change the workflow to self-hosted until such a runner is online and authorized.

No model training, checkpoint/data changes, benchmark promotion, PR merge, or main-branch update occurred.



## Stage 16A.15 — independent cgroup cleanup attempts

Static lifecycle review found that two pre-launch failure handlers used a single sequential `try` block for `cgroup.kill`, `wait_empty`, and `close`. If the first operation failed, later cleanup operations were skipped. This could leave an otherwise-empty cgroup allocated even when removal was still possible.

Changes:
- Added one shared unlaunched-cgroup cleanup routine that attempts kill, empty-state verification, and removal independently, retaining each failure for the original exception note.
- Reused it for both launch-gate preparation failures and `Popen` launch failures.
- Cgroup creation rollback no longer silently discards a failed `rmdir`; the cleanup failure is attached to the original configuration exception.
- Added failure-injection coverage proving that wait and removal are still attempted after an injected cgroup-kill failure.

Verification:
- Source-level edits and regression test are committed; **the test has not been executed**.
- The latest Actions job still has no step records or logs, so no CI pass/fail conclusion about Python tests is available.
- Strict sandbox remains **NOT RUNTIME-VERIFIED**. Live delegated cgroup v2 + Bubblewrap qualification remains required.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or `main` update occurred.



## Stage 16A.16 — launch-gate descriptor failure containment

Static review of the strict launch sequence found that descriptor closure itself could interrupt cleanup. In particular, an exception while closing the parent's gate-read descriptor after `Popen` could bypass cgroup attachment/termination handling, leaving a blocked launcher; a gate-write close failure after release could escape while the child continued executing.

Changes:
- Added a non-throwing launch-descriptor close helper that returns close failures for explicit handling.
- On parent gate-read close failure, the launcher is terminated through the common process/cgroup cleanup path and the run fails closed.
- Gate attachment, release, and gate-write closure now share one guarded failure path. Any failure triggers process/cgroup cleanup; secondary descriptor-close and cleanup failures are retained as exception notes.
- Process-launch failure now attempts to close both parent descriptors independently before cleaning the unlaunched cgroup.
- Added focused regression coverage for descriptor-close failure reporting and absent descriptors.

Verification:
- Source and regression tests are committed. **No Python test has been executed**; the latest GitHub Actions job failed before emitting steps or logs.
- This is a source-level lifecycle correction, not evidence of live descriptor-race or cgroup behavior.
- Strict sandbox remains **NOT RUNTIME-VERIFIED** pending a functioning Linux runner with delegated cgroup v2 and Bubblewrap.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or `main` update occurred.



## Stage 16A.17 — cgroup lifecycle event integrity

Static review found that `CgroupV2.wait_empty()` previously built a dictionary from whatever event lines happened to parse as two fields. Malformed lines were silently ignored, duplicate keys overwrote earlier values, and a missing `populated` field was indistinguishable from a cgroup that remained populated until timeout.

Changes:
- Added strict `cgroup.events` parsing. Every line must contain exactly a key and a non-negative ASCII decimal value; duplicate keys are rejected.
- Require an explicit `populated` field. Its value, and the documented `frozen` boolean when present, must be 0 or 1.
- `wait_empty()` now returns success only for a structurally valid record with `populated 0`; malformed or incomplete evidence fails immediately.
- Reject non-positive, boolean, non-finite, and non-numeric timeout values before polling.
- Added regression cases for missing/duplicate/malformed fields, invalid boolean values, valid empty evidence, and invalid timeout values.

Verification:
- Source and tests are committed. **No Python tests have been executed**; GitHub Actions continues to fail before producing usable step records/logs.
- These are parser and control-flow regression specifications, not live kernel-cgroup qualification.
- Strict sandbox remains **NOT RUNTIME-VERIFIED** pending an actual Linux runner with delegated cgroup v2 and Bubblewrap.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or `main` update occurred.
