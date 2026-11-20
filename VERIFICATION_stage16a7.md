# Stage 16A.7 — Aggregate Resource Enforcement and Process-Tree Containment

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Branch:** `fibonera/cortex-remediation-stage16a`

## Implemented

- Added `cortex/execution/resources.py` with a cgroup v2 controller for an administrator-provisioned delegated subtree selected explicitly through `CORTEX_CGROUP_ROOT`.
- Requires the `cpu`, `memory`, and `pids` controllers to be both available and enabled in the delegated subtree. Cortex does not mutate the host controller hierarchy.
- Configures per-execution `cpu.max`, `memory.max`, `pids.max`, and disables swap through `memory.swap.max` when exposed by the kernel.
- Requires `cgroup.kill` so process-tree cleanup is not inferred from a direct-child timeout.
- The module reports I/O limits as `NOT_CONFIGURED`; no disk-I/O enforcement is claimed.
- Added unit tests for missing delegation, missing/disabled controllers, applied limits, attach/kill lifecycle, missing `cgroup.kill`, and invalid limit values.

## Integration boundary

This increment introduces the controller and its regression tests. It does **not yet claim that sandbox processes are attached to cgroups**. The next integration step must gate process launch until the child is attached, then prove that aggregate memory/process limits and cgroup-wide termination affect descendants. API execution must remain fail-closed until that integration and a live Linux qualification pass.

## Administrator prerequisites

The service administrator must create a delegated cgroup v2 subtree, enable `cpu`, `memory`, and `pids` controllers for it, grant the Cortex service account write access, and set `CORTEX_CGROUP_ROOT` to that subtree. Cortex intentionally does not write to root `cgroup.subtree_control` or attempt privileged controller delegation.

## Verification

**CONTROLLER IMPLEMENTED; SANDBOX INTEGRATION AND LIVE ENFORCEMENT NOT YET VERIFIED.** The existing GitHub Actions runner allocation problem remains unresolved. Unit tests are regression specifications until executed. No cgroup behavior, resource-limit enforcement, or descendant-kill pass is claimed.

I/O bandwidth/IOPS controls, seccomp syscall policy, and deliberate namespace-escape resistance remain outstanding. No training, benchmark promotion, checkpoint or dataset change, PR merge, or `main` branch change was performed.
