# Stage 16A.10 — Strict Sandbox Launch and Evidence Correctness

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Branch:** `fibonera/cortex-remediation-stage16a`

## Findings

A source-level review of the Stage 16A.8 implementation found two correctness defects that needed to be fixed before its kernel-evidence qualification could be trusted:

1. **Cgroup launch bootstrap was not valid Python source.** The launcher assembled source lines containing literal backslash-n sequences rather than actual line boundaries. The interpreter therefore could not reliably execute the pipe-gated bootstrap. Since the bootstrap is the process held before cgroup attachment, this could prevent every strict cgroup-backed execution from reaching the requested command. The source is now generated from explicit line elements and has a compile-based regression test.
2. **The process-limit probe could be masked by RLIMIT_NPROC.** Strict mode previously set the per-process RLIMIT_NPROC to the same numeric value as the cgroup pids.max limit. On an unprivileged host, RLIMIT_NPROC could reject process creation before the cgroup controller records a pids.events:max event. Strict mode now raises the inherited soft RLIMIT_NPROC to its existing hard ceiling; cgroup pids.max remains the authoritative aggregate cap. The non-strict fallback retains its prior RLIMIT_NPROC behavior.

## Implemented

- Added `_cgroup_bootstrap_source()` to build valid newline-delimited launcher code.
- Added compile-time regression coverage for the launcher source.
- Added an explicit strict-mode process-limit policy to `_limits()`, keeping cgroup pids accounting independently observable.
- Added regression tests for strict and degraded RLIMIT_NPROC behavior.
- No changes to model architecture, training data, checkpoints, benchmark fingerprints, Stage 12 champion, or virtual parameter store.

## Verification status

**PATCHED; TESTS NOT EXECUTED.** The regression tests are committed as source, but GitHub Actions has continued to fail before job steps are created. No test pass is claimed. These fixes do not constitute live qualification of Bubblewrap, delegated cgroup v2, CPU throttling, memory OOM, or aggregate process enforcement.

The strict sandbox remains unavailable until a Linux host with working Bubblewrap and administrator-delegated cgroup v2 is configured and the behavioral qualification actually passes. Seccomp remains uncertified; disk I/O limits remain unconfigured.

## Scope and safety

No training was started. No benchmark was promoted. No checkpoint or dataset was modified. PR #2 remains open and unmerged; `main` is unchanged.
