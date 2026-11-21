# Stage 16A.6 — Behavioral Sandbox Isolation Qualification

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Branch:** `fibonera/cortex-remediation-stage16a`

## Implemented

- Added `cortex/execution/qualification.py`.
- The qualification invokes the exact `sandbox.run(..., require_full_isolation=True)` path used by the API; it does not use a separate weaker Bubblewrap command.
- Behavioral checks cover disposable workspace write/read, host-path denial, read-only `/usr`, expected `/proc` and `/dev`, and external TCP reachability to `1.1.1.1:53`.
- Each check reports `PASS`, `FAIL`, or `NOT_RUN`; aggregate qualification passes only when every check passes.
- Added CLI: `python -m cortex.execution.qualification` (JSON report; exit 1 on failure).
- API code-execution routes now require a successful behavioral qualification and return HTTP 503 on failed/incomplete qualification.
- Added five regression tests covering positive behavior, host-path exposure, missing Bubblewrap, setup failure and API fail-closed behavior.

## Explicit limitations

This is a focused behavioral qualification, not complete sandbox certification. It does not establish seccomp syscall filtering, cgroup-enforced aggregate CPU/memory/process/I/O limits, reliable `RLIMIT_NPROC` enforcement under privileged/root execution, deliberate namespace/session escape resistance, multi-tenant isolation, or kernel exploit resistance. The network check probes one external endpoint only.

## Verification

**IMPLEMENTED; NOT VERIFIED IN A LIVE LINUX/BUBBLEWRAP ENVIRONMENT.** The latest Actions job for the parent commit failed before runner startup (`runner_id=0`, no steps/logs). New tests are regression specifications until executed on a functioning runner. No runtime qualification pass is claimed.

No model training, benchmark promotion, checkpoint or dataset change, PR merge, or `main` branch change was performed.
