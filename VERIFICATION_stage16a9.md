# Stage 16A.9 — CI Trigger and Queue Hygiene

**Owner:** Fibonera  
**Repository:** Alex10007/cortex-codegen  
**Branch:** `fibonera/cortex-remediation-stage16a`

## Findings

- PR #2 targets `vibe/cortex-core-b71b2c`, while the workflow's pull-request trigger previously listened only to PRs targeting `main`. That configuration excluded PR #2 from pull-request-triggered validation.
- Rapid commits generated a large burst of workflow runs. The workflow had no concurrency group, so superseded commits could each request hosted runner capacity.
- The most recent Stage 16A.8 Actions run failed in seconds. A retry was issued; the retry also failed in seconds. The test job returned no steps and no retrievable logs. Earlier observations included `runner_id=0`. This is consistent with failure before a runner executes the workflow, but does not identify a definitive GitHub-side root cause.

## Implemented

- Added the actual PR base branch `vibe/cortex-core-b71b2c` to `pull_request.branches`.
- Added per-source-branch concurrency with `cancel-in-progress: true` to discard superseded runs.
- Set explicit read-only `contents: read` workflow permissions.
- Added bounded job timeouts (30 minutes for tests, 10 minutes for deployment).

## Verification

**Workflow configuration updated; runtime CI still blocked.** The Actions retry of run [44821801150](https://github.com/Alex10007/cortex-codegen/actions/runs/44821801150) failed again without producing job steps or logs. No tests are claimed as executed or passed. These workflow changes address a confirmed PR-trigger mismatch and unnecessary run queuing; they cannot repair an unavailable GitHub-hosted runner pool or account-level Actions configuration.

Next operational requirement: inspect repository/org Actions settings and runner availability, or attach a supported owner-managed self-hosted Linux runner. Do not change the runner label to `self-hosted` until such a runner is actually registered and online.

No model training, benchmark promotion, checkpoint/data modification, merge, or `main` branch update occurred. Stage 16A remains **NOT VERIFIED**.
