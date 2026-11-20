# Stage 16A.7 — Aggregate Resource Enforcement and Process-Tree Containment

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Branch:** `fibonera/cortex-remediation-stage16a`

## Implemented

- Added `cortex/execution/resources.py`, a Linux cgroup v2 controller for an administrator-provisioned delegated subtree selected through `CORTEX_CGROUP_ROOT`.
- Requires `cpu`, `memory`, and `pids` controllers to be available and enabled. Cortex does not alter the host controller hierarchy.
- Each strict sandbox run creates a unique child cgroup and configures:
  - `cpu.max`: one CPU-equivalent quota by default;
  - `memory.max`: the requested aggregate memory ceiling;
  - `memory.swap.max=0`: no cgroup swap allowance;
  - `memory.oom.group=1`: group-wide OOM behavior;
  - `pids.max`: aggregate process/task ceiling.
- Requires `cgroup.kill` and `cgroup.events`. If required controls are unavailable, strict execution refuses to start.
- Sandbox process launch now uses a pipe-gated bootstrap. The launcher blocks before executing Bubblewrap or untrusted code; Cortex moves the launcher PID into its cgroup and only then releases the gate. EOF on the gate exits without executing the command.
- After the direct process exits or times out, Cortex kills the POSIX process group and then issues cgroup-wide `cgroup.kill`, covering descendants that deliberately start a new session and escape ordinary process-group cleanup.
- Cortex waits for `cgroup.events` to report `populated 0` before removing the per-run cgroup. Failure to prove an empty cgroup is a hard execution error.
- Execution metadata reports cgroup enforcement, aggregate CPU/memory/process controls, and successful empty-cgroup cleanup separately. The isolation descriptor includes the cgroup v2 controls.
- API workers cache the full behavioral qualification result once per process. Each actual strict execution still creates and enforces a fresh cgroup.
- The qualification now requires delegated cgroup v2 support, attempts to exceed the aggregate process cap, starts detached new-session descendants, and requires successful cgroup-wide cleanup.
- Added unit tests for controller availability/delegation, applied control values, attach/kill/empty lifecycle, missing required controls, invalid limits, and strict-mode refusal when no delegated cgroup is configured. Updated behavioral qualification and API-gate tests.

## Administrator prerequisites

The service administrator must:
1. Use a unified cgroup v2 host.
2. Provision a delegated subtree for the Cortex service account.
3. Enable `cpu`, `memory`, and `pids` in that subtree's `cgroup.subtree_control`.
4. Ensure `memory.swap.max`, `memory.oom.group`, `cgroup.kill`, and `cgroup.events` are available.
5. Grant the Cortex service account permission to create and configure child cgroups.
6. Set `CORTEX_CGROUP_ROOT` to the delegated subtree.

Cortex intentionally does not elevate privileges or modify root cgroup controller delegation.

## Explicit limitations

- Numeric CPU throttling and memory-pressure/OOM ceilings have not yet been stress-tested against controlled workloads. The controller configures the limits; the behavioral qualification is designed to verify delegated controls, aggregate process-cap behavior and detached-descendant cleanup when run on a qualified host.
- Disk I/O bandwidth/IOPS controls are `NOT_CONFIGURED`; no I/O enforcement is claimed.
- No seccomp syscall policy is installed or certified.
- Cgroup membership prevents ordinary process/session detachment from escaping cleanup, but deliberate privileged cgroup migration, kernel vulnerabilities, and host-kernel escape are not certified.
- The external network check probes one endpoint only and is not a complete network audit.
- A live Linux deployment qualification remains necessary. Unit tests cannot substitute for a real delegated cgroup v2 environment.

## Verification status

**IMPLEMENTED; NOT RUNTIME-VERIFIED.** The code and regression tests are committed, but GitHub Actions has repeatedly failed before runner allocation. The latest observed job has no executed steps or test logs. No passing test-suite or live cgroup qualification is claimed.

No model training, benchmark promotion, checkpoint/dataset change, PR merge, or `main` branch change was performed.

**Next:** Stage 16A.8 should add controlled CPU and memory pressure tests, syscall filtering with an explicit allow/deny policy, and a host-side adversarial escape test plan. I/O limits should be added only with a real device mapping and measured policy rather than guessed defaults.
