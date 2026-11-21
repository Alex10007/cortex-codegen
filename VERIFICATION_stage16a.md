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


## Stage 16A.18 — delegated-root and cgroup-name ownership safety

Static review examined delegated-root path resolution and exclusive cgroup creation.

Changes:
- Delegated cgroup roots must now be absolute, resolvable existing directories.
- Reject paths that traverse symlink aliases in any component, rather than checking only whether the final path component itself is a symlink.
- Use the canonical resolved root for subsequent child-path construction.
- Added a symlink-root regression test.
- Added a deterministic UUID-collision regression: when the generated cgroup name already exists, creation fails and the pre-existing directory plus its ownership sentinel remain untouched.

Safety boundary:
- Path.mkdir() remains the exclusive creation operation; a collision is not treated as permission to reuse an existing cgroup.
- Setup rollback only attempts to remove the path after this invocation successfully created it.
- This increment does not claim descriptor-relative protection against a privileged administrator concurrently renaming/replacing the delegated mount/root after validation. A dirfd/openat-based implementation and live concurrent stress testing remain potential hardening work.
- It does not certify cgroup kernel behavior or delegated-root mount identity.

Verification:
- Source and regression tests committed; **no Python tests have been executed**. GitHub Actions has not produced usable runner step records/logs.
- Strict sandbox remains **NOT RUNTIME-VERIFIED** pending a functioning Linux runner with delegated cgroup v2 and Bubblewrap.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or main update occurred.


## Stage 16A.19 — consistent delegated-root validation

Static review found that CgroupV2.create() rejected delegated roots reached through symlink aliases in any path component, while probe_cgroup_v2() only rejected a symlink at the final component. This allowed qualification and execution setup to disagree about the same configured root.

Changes:
- Centralized absolute-path, strict resolution, directory, and full-path symlink-alias validation in CgroupV2._resolve_delegated_root().
- Both create() and probe_cgroup_v2() now use the same validator and canonical root.
- The probe now fails closed for unresolved roots and path aliases before reading controller files.
- Added regression coverage for an intermediate-component symlink and a missing root, checking probe/create consistency.

Verification:
- Source and regression tests are committed; **no Python tests have been executed** in this increment.
- This closes a source-level policy inconsistency only. It does not eliminate privileged concurrent mount/path replacement races or qualify live kernel cgroup behavior.
- GitHub Actions has previously failed before emitting usable job steps/logs; no CI pass is claimed.
- Strict sandbox remains **NOT RUNTIME-VERIFIED** pending a functioning Linux runner with delegated cgroup v2 and Bubblewrap.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or main update occurred.


## Stage 16A.20 — cgroup directory identity binding

Static review found that a `CgroupV2` instance retained only a pathname. If that path were renamed and replaced before teardown, a later `kill()`, `wait_empty()`, or `close()` could act on a different cgroup directory. Setup rollback had a similar pathname-reuse hazard.

Changes:
- Capture the created cgroup directory's `(st_dev, st_ino)` identity immediately after exclusive creation.
- Verify the current path still identifies the same non-symlink directory before attach, kill, event polling, snapshot, and removal.
- Re-check identity immediately before `rmdir()` and refuse teardown if the path has changed.
- Setup rollback now checks the captured identity and refuses to remove a replacement directory.
- Added regressions for directory replacement and symlink replacement, asserting replacement/owner data is left untouched.

Safety boundary:
- This is pathname identity-change detection, not a complete defense against a privileged concurrent rename/replacement between the identity check and filesystem operation (TOCTOU). Descriptor-relative `openat`/directory-fd operations remain the stronger follow-on hardening. No kernel cgroup behavior is inferred from these filesystem tests.

Verification:
- Source and regression tests committed; **not executed** in this increment.
- GitHub Actions has repeatedly failed before emitting usable job steps/logs; no CI pass is claimed.
- Strict sandbox remains **NOT RUNTIME-VERIFIED** pending a functioning Linux runner with delegated cgroup v2 and Bubblewrap.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or `main` update occurred.


## Stage 16A.21 — descriptor-bound cgroup control operations

Static review of Stage 16A.20 identified a residual path-resolution window: checking a cgroup pathname's inode and then opening a control file by pathname still allowed the directory entry to be swapped between those operations.

Changes:
- CgroupV2 now opens and retains a file descriptor for the parent directory and the owned execution-cgroup directory.
- The child directory is opened relative to the retained parent descriptor with O_DIRECTORY and O_NOFOLLOW, then its fstat device/inode is checked against the captured identity.
- Runtime control writes (cgroup.procs, cgroup.kill) and event/counter reads (cgroup.events, cpu.stat, memory.events, pids.events, memory.current) use os.open(..., dir_fd=owned_directory_fd) and O_NOFOLLOW.
- Teardown checks the child entry's identity relative to the retained parent descriptor and removes it with descriptor-relative os.rmdir.
- Directory descriptors are closed after successful teardown; failed teardown retains descriptors so cleanup can be retried.
- Added regression tests for symlinked control-file rejection and retained descriptor identity/lifecycle.

Scope and residual risks:
- Initial delegated-root resolution and creation/configuration of the new cgroup directory and its initial limits still use path-based operations. This increment hardens subsequent runtime control I/O and removal; it is not a complete descriptor-relative creation transaction.
- The retained parent descriptor anchors operations to the opened parent directory, but does not prevent a privileged actor from moving/remounting cgroup hierarchy objects or changing kernel delegation.
- The existing pathname identity check remains as an additional fail-closed guard; descriptor-relative access ensures a successful file operation is addressed within the opened directory rather than a newly substituted child pathname.
- Unit fixtures use ordinary filesystem files. They do not prove that the target Linux cgroupfs supports every open mode or that kernel resource enforcement works.

Verification:
- Source and regression tests committed; **not executed** in this increment.
- No CI pass or live cgroup/Bubblewrap qualification is claimed.
- Strict sandbox remains **NOT RUNTIME-VERIFIED**.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or main-branch update occurred.


## Stage 16A.22 — descriptor-relative cgroup creation and initial configuration

Changes:
- The delegated root is opened once as a directory descriptor with no-follow semantics. Its opened device/inode identity is compared with the resolved root identity before any child is created.
- Controller availability and subtree enablement are read relative to the pinned delegated-root descriptor.
- Execution cgroups are created with descriptor-relative mkdir semantics. Existing UUID collisions fail closed; the implementation never reuses an existing directory.
- The newly created child entry is inspected without following symlinks, opened relative to the pinned root, and its device/inode identity is compared with the opened directory descriptor.
- Initial cpu.max, memory.max, pids.max, memory.swap.max, and memory.oom.group configuration is written relative to the opened child descriptor with O_NOFOLLOW.
- Required cgroup.procs, cgroup.kill, and cgroup.events controls are checked relative to that descriptor without following symlinks.
- Setup rollback verifies the child entry identity relative to the pinned parent descriptor and removes only that owned entry. Descriptor closure and rollback errors are retained as failure notes.
- Added tests for descriptor-relative setup failure rollback and rejection of a symlinked initial control file without modifying its external target. Updated create fixtures to emulate cgroupfs controls under descriptor-relative mkdir.
- Corrected a malformed escaped newline in the test module import discovered during branch inspection.

Scope and residual risks:
- The configured root must still be resolved and initially opened by pathname. The opened root is identity-checked and then retained; subsequent controller reads, child creation, child configuration, and rollback are relative to that descriptor.
- This does not defend against a privileged actor that can alter the mounted cgroup hierarchy or change kernel delegation semantics. It is not a kernel-escape or host-isolation certification.
- Unit tests emulate cgroup pseudo-files with regular files and `/proc/self/fd` directory resolution. They cannot establish compatibility of all open modes with a live cgroupfs mount or prove that the kernel enforces the configured limits.

Verification:
- Source, tests, and this report are committed on the remediation branch.
- Tests have **not been executed**. GitHub Actions continues to fail before job steps are reported; no CI pass is claimed.
- Live delegated-cgroup v2 and Bubblewrap qualification remains outstanding. Strict sandbox remains **NOT RUNTIME-VERIFIED**.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or main-branch update occurred.


Stage 16A.22 follow-up verification note:
- The main cgroup creation success fixture now emulates cgroupfs pseudo-file teardown so `close()` can perform its own final `cgroup.events` check before directory removal.
- UUID collision handling is scoped specifically to the descriptor-relative mkdir operation; a later `FileExistsError` during setup now enters the ordinary owned-directory rollback path.
- Latest Actions run `44843634159` failed with the test job reporting `steps: null` and no logs. This remains a pre-step runner/workflow failure; it does not provide a Python test result.


## Stage 16A.23 — component-by-component delegated-root acquisition

Changes:
- Added `CgroupV2._open_delegated_root()`, which starts from an opened filesystem root directory and walks each canonical delegated-root component using directory descriptors.
- Each component is inspected relative to the currently pinned parent with `follow_symlinks=False`, must be a directory, and is then opened with `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`.
- The opened component's `fstat` device/inode identity must match the no-follow directory-entry metadata captured immediately before opening. A replacement or symlink swap during acquisition fails closed.
- The final delegated-root descriptor is returned and used directly for controller reads and child cgroup creation; create no longer reopens the resolved root pathname.
- `probe_cgroup_v2()` now uses the same component-walk acquisition and descriptor-relative controller reads, so probe and create share the same root-acquisition policy.
- Added regression tests for an intermediate symlink alias and a deterministic directory-to-symlink swap between entry inspection and open.

Scope and residual risks:
- The existing canonical-path policy remains: a root whose supplied path resolves through symlink aliases is rejected. The new component walk adds race-resistant acquisition after canonicalization.
- The walk pins each opened directory while advancing; it does not prevent a privileged actor from renaming/mounting ancestors or changing cgroup delegation after acquisition. Kernel hierarchy mutation and privileged adversaries are not certified.
- Tests have been authored but not executed. The race test uses a local filesystem and verifies `O_NOFOLLOW` acquisition behavior; it is not a live cgroupfs test.

Verification:
- Source and 27 regression tests are committed; no test execution or CI pass is claimed.
- Live delegated cgroup v2 and Bubblewrap qualification remains outstanding. Strict sandbox remains **NOT RUNTIME-VERIFIED**.
- Latest Actions run status must be checked after this commit; earlier runs failed before job steps and provided no Python test evidence.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or `main` branch update occurred.


## Stage 16A.24 — cgroup descriptor lifecycle cleanup

Static lifecycle review found that descriptor cleanup could itself short-circuit: the directory-descriptor cleanup helper stopped after the first os.close() error, delegated-root traversal could lose the newly opened descriptor if releasing the previous one failed, and create() finalization could skip closing the parent descriptor after a child-descriptor close error. Some cleanup failures could also replace the primary acquisition/setup exception.

Changes:
- The directory-descriptor cleanup helper now invalidates each owned descriptor exactly once, attempts child and parent closure independently, and returns all close failures. It is safe to call repeatedly; it does not retry a possibly-interrupted close() against a descriptor number that may already have been released and reused.
- Constructor failure records secondary descriptor-cleanup errors on the original exception while still attempting both closes.
- Delegated-root component traversal transfers ownership to the newly opened descriptor before releasing the previous descriptor. If previous-descriptor release fails, the traversal descriptor is still cleaned up; if component inspection fails, cleanup errors are attached to the primary error.
- create() independently attempts child and parent descriptor release in finalization, and preserves cleanup failures as notes on an active setup exception rather than masking it. Owned-directory rollback remains identity-checked and is attempted even if closing the child descriptor reports an error.
- probe_cgroup_v2() now preserves a controller-inspection exception if closing the pinned root descriptor also fails; on an otherwise successful probe, a close failure is surfaced as a failed probe.
- Successful close() marks the cgroup removed before reporting descriptor-close failures. A repeated close is therefore idempotent and will not attempt a second removal.

Regression coverage added:
- First descriptor close fails while the second is still attempted; both descriptor attributes are invalidated.
- Constructor identity failure attempts closure of both transferred descriptors.
- Delegated-root traversal closes the newly opened descriptor when releasing the previous descriptor fails.
- Repeated close() after successful removal performs only one directory removal.

Verification status:
- Source and four new fault-injection tests are committed on this branch.
- The new tests have NOT been executed. No CI pass is claimed; prior GitHub Actions failures before step execution remain a runner/workflow issue, not Python test evidence.
- These filesystem tests do not establish live cgroupfs compatibility or kernel enforcement. Strict sandbox remains NOT RUNTIME-VERIFIED pending full test execution and live delegated cgroup v2 + Bubblewrap qualification.
- No model training, checkpoint/dataset changes, benchmark promotion, PR merge, or main update occurred.


Stage 16A.24 test-fixture correction:
- Review of the newly added idempotent-close test exposed three existing cgroup.events fixtures in this test module that encoded the newline escape literally rather than writing actual line breaks. Corrected all three fixtures, including the new test, so the strict event parser receives valid `populated 0` records.
- This is a source-level fixture correction only; the tests remain unexecuted.

## Stage 16A.25 — pinned-parent ownership checks and serialized teardown

Static review identified two additional lifecycle hazards: ownership validation re-resolved the full cgroup path after retaining directory descriptors, so a replaced ancestor could make path-based checks disagree with the directory actually used by descriptor-relative operations; and two concurrent calls to close() on the same manager could both pass the initial open-state check and race toward removal.

Changes:
- _assert_owned_path() now verifies the child directory entry with os.stat(..., dir_fd=pinned_parent_fd, follow_symlinks=False) and compares it with both the manager's recorded identity and fstat(pinned_child_fd). It no longer re-resolves the ancestor pathname for runtime ownership checks.
- The pinned parent descriptor is also verified as a directory before any owned operation.
- Each manager now has a teardown lock. The complete close sequence (open-state check, empty wait, identity recheck, removal, descriptor closure) is serialized, making concurrent close() calls on the same manager idempotent rather than allowing competing removals.
- The final removal check explicitly requires a real directory entry and matching device/inode identity immediately before descriptor-relative rmdir.
- Added regression tests proving that replacing the parent pathname does not redirect cleanup into a replacement directory, and that concurrent close calls on one manager perform only one wait/removal.
- Corrected the two new race-test cgroup.events fixtures to contain actual newline escapes.

Residual race boundary:
- Linux/POSIX directory removal is still name-based (rmdir/unlinkat); there is no portable inode-conditional rmdir operation. A privileged or otherwise authorized actor that can concurrently mutate the pinned parent directory could theoretically replace the child entry in the interval between the final statat identity check and rmdir. The descriptor-bound checks and per-manager lock reduce accidental and same-instance races but do NOT eliminate this adversarial cross-process TOCTOU window.
- The delegated cgroup parent must therefore remain access-controlled so untrusted workloads cannot rename, replace, or create sibling cgroup entries. This increment does not claim resistance to a privileged actor with write access to the delegated parent.
- A future hardening step should validate and document the delegated-parent ownership/permission model against the actual service deployment, and determine whether the supported cgroupfs/kernel environment provides a safe atomic removal strategy. Do not infer such support from ordinary filesystem unit tests.

Verification:
- Source changes and two new race/lifecycle regression tests are committed on the remediation branch.
- Tests have NOT been executed. No CI pass is claimed and no live cgroupfs behavior is established.
- Strict sandbox remains NOT RUNTIME-VERIFIED pending full test execution and live delegated cgroup v2 + Bubblewrap qualification.
- No model training, checkpoint/dataset changes, benchmark promotion, PR merge, or main-branch update occurred.


## Stage 16A.26 — delegated cgroup filesystem isolation gate

Stage 16A.25 documented that descriptor-bound cleanup still has a name-based removal race if an actor with write access to the delegated parent can replace a sibling entry. The service's untrusted execution path runs through Bubblewrap with `--unshare-all` and does not intentionally bind `/sys` or the host cgroup hierarchy into the sandbox. However, that design assumption was not part of behavioral qualification.

Changes:
- Extended the actual Bubblewrap qualification payload to inspect its own `/proc/self/mountinfo` and fail if any cgroup v1 or v2 filesystem mount is visible inside the untrusted execution namespace.
- The payload also attempts to open `/sys/fs/cgroup/cgroup.procs` for read/write and records failure if that control surface is accessible.
- Added `cgroup_filesystem_inaccessible` as a required qualification check. The API's existing fail-closed qualification gate therefore refuses strict API execution when untrusted code can see a cgroup filesystem.
- Added a regression test proving that qualification fails when the cgroup filesystem visibility check fails.

Security boundary:
- This gate tests the namespace/mount view of the actual Bubblewrap-launched process; it does not infer safety from host directory mode bits or assume a particular systemd ownership/GID arrangement.
- If the check passes on the deployment host, ordinary filesystem-based access to cgroup v1/v2 control files is absent from the untrusted process's mount namespace, reducing the ability to rename or replace sibling cgroups through that interface.
- It does not prove the delegated parent is safe against the Cortex service identity itself, a privileged host actor, kernel vulnerabilities, cgroup migration through privileged mechanisms, or an unexpected writable cgroup mount introduced by future sandbox changes. The parent must remain administratively controlled; service and workload identity separation is still a stronger defense-in-depth option.
- The probe conservatively rejects any visible cgroup v1/v2 mount, even if it might be read-only. This is intentional for the strict execution profile.

Verification:
- Source, tests, and report are committed as source changes only. **No Python tests were executed** and no live Bubblewrap/cgroup qualification was performed in this increment.
- GitHub Actions has not yet provided a usable runner result for the current branch; no CI pass is claimed.
- Strict sandbox remains **NOT RUNTIME-VERIFIED** until the full suite runs and this new behavioral check passes on a real deployment host with delegated cgroup v2.

No model training, checkpoint or dataset mutation, benchmark promotion, PR merge, or `main` update occurred.


## Stage 16A.29 — CI isolation and API limiter connection lifecycle

Additional pre-CI corrections:

- The first self-hosted qualification workflow is restricted to the owner-controlled `fibonera/cortex-remediation-stage16a` branch. It has no `pull_request` trigger; the job also checks the triggering actor and exact branch.
- Third-party GitHub Actions are pinned to immutable commit SHAs: `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683` (v4.2.2) and `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065` (v5.6.0).
- Every CI job creates a fresh Python virtual environment under `RUNNER_TEMP`; dependency installation no longer modifies the runner account's system Python.
- The CI preflight verifies that the selected interpreter is the virtual environment, refuses a mounted Windows C: drive, and compiles the Cortex and test packages before test collection.
- The workflow runs dependency consistency checks, records the resolved package inventory, collects the test suite before execution, and hashes tracked verification artifacts before and after tests.
- The rate limiter had a connection-lifecycle defect: using a SQLite connection as a context manager commits/rolls back transactions but does not close the connection. `consume()` now uses `contextlib.closing`, and a regression test checks connection closure on both accepted and rate-limited requests.
- `.gitignore` now excludes local environment files, virtual environments, local secrets directories, and runner workspace files to reduce accidental credential/artifact commits.

Verification status:
- These changes are source-committed; the new connection-lifecycle regression and complete test suite have **not yet executed**.
- The latest restricted workflow run is [44872381628](https://github.com/Alex10007/cortex-codegen/actions/runs/44872381628), currently waiting for a matching self-hosted runner.
- Older queued/pending workflow runs were created before the final branch restriction and virtual-environment correction. They must be canceled in GitHub Actions before the runner is registered; do not let the ephemeral laptop runner accept those older jobs.
- The first run remains a CPU-only software test. It does not qualify GPU training, production sandbox security, distributed training, or model capability.

**Stage 16A.29 status: implemented in source; NOT RUNTIME-VERIFIED.** No training, checkpoint mutation, benchmark promotion, PR merge, or main-branch change occurred.


## Stage 16A.30 — CI workspace cleanliness gate

- The post-test integrity step now checks tracked-file diffs, staged changes, and untracked non-ignored files. A test that leaves a new repository file behind cannot silently pass the workspace-integrity gate.
- Ignored runtime outputs remain governed by the repository ignore rules; tracked benchmark and verification evidence remains protected by the before/after SHA-256 check.
- This is a repository cleanliness check, not a complete filesystem snapshot and not a security boundary against privileged host modification.

**Status:** workflow source committed; runtime execution remains pending. No test result is inferred from the workflow definition.
