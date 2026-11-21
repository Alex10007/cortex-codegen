# Stage 16A.8 — Kernel-Evidence Resource Qualification

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Branch:** `fibonera/cortex-remediation-stage16a`

## Implemented

- Added cgroup v2 counter snapshots captured after process-tree termination and before the per-execution cgroup is removed.
- Captures kernel-maintained `cpu.stat`, `memory.events`, `pids.events`, and `memory.current`. Malformed or unavailable evidence fails strict execution rather than silently reporting a pass.
- Exposes CPU quota parameters to the strict sandbox runner so qualification can apply a deliberately constrained quota.
- Strengthened process-cap qualification: a failed child spawn alone is no longer accepted. The kernel `pids.events:max` counter must increase.
- Added a CPU-bound workload and requires positive `nr_throttled` and `throttled_usec` evidence from `cpu.stat`.
- Added concurrent memory-allocation workers whose individual address-space limits are higher than the aggregate cgroup allowance; qualification requires kernel `memory.events` OOM and kill counters to increase.
- Added unit coverage for parsing and retaining resource counters.
- Preserved fail-closed behavior if the kernel evidence cannot be collected.

## Seccomp decision

Seccomp is **not enabled in this increment**. The repository does not yet establish a tested, maintained filter compiler/runtime and syscall allowlist compatible with the supported Python runtime, Bubblewrap launcher, and project execution workloads. Installing a speculative default-deny filter without an executable compatibility matrix would risk either breaking legitimate execution or creating a false security claim. Strict sandbox execution therefore remains explicitly **without seccomp certification**. A later increment must select a supported mechanism, define the syscall policy, and prove both required-syscall success and denied-syscall behavior on the target Linux kernel before enabling it.

## Verification status

**IMPLEMENTED; NOT RUNTIME-VERIFIED.** These changes add evidence-driven qualification, but live CPU throttling, aggregate memory OOM, and pids-limit checks require a Linux host with a real delegated cgroup v2 subtree and working Bubblewrap. The repository's GitHub Actions runner has previously failed before runner allocation; source edits and unit tests must not be described as passing until a runner executes them.

Disk I/O controls remain `NOT_CONFIGURED`. This increment does not certify seccomp, kernel escape resistance, privileged cgroup migration resistance, or a complete network audit. No model training, benchmark promotion, checkpoint/dataset mutation, merge, or `main` branch change is included.
