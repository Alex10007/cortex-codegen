"""Behavioral qualification for the Bubblewrap execution path used by Cortex."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from cortex.execution.sandbox import NetworkIsolationUnavailable, _supports_bwrap, run
from cortex.execution.resources import probe_cgroup_v2


def qualify_sandbox() -> dict[str, Any]:
    """Exercise filesystem, network, aggregate process and cgroup cleanup boundaries.

    This focused probe does not certify seccomp, numeric CPU/memory ceilings,
    deliberate cgroup migration, or kernel exploit resistance.
    """
    checks = {
        "bwrap_available": "PASS" if _supports_bwrap() else "FAIL",
        "cgroup_v2_delegation": "PASS" if probe_cgroup_v2().get("available") else "FAIL",
        "isolated_execution": "NOT_RUN",
        "aggregate_limits_enforced": "NOT_RUN",
        "detached_descendant_cleanup": "NOT_RUN",
        "aggregate_process_cap_observed": "NOT_RUN",
        "workspace_writable": "NOT_RUN",
        "host_path_inaccessible": "NOT_RUN",
        "runtime_readonly": "NOT_RUN",
        "proc_dev_available": "NOT_RUN",
        "network_unreachable": "NOT_RUN",
    }
    if checks["bwrap_available"] != "PASS" or checks["cgroup_v2_delegation"] != "PASS":
        return {"passed": False, "checks": checks,
                "limitations": [
                    "Bubblewrap namespace capability and delegated cgroup v2 controllers are both required.",
                    "The cgroup probe verifies delegation; sandbox.run must still attach each launched process before exec.",
                ]}

    with tempfile.TemporaryDirectory(prefix="cortex-qualification-host-") as host_dir:
        host_sentinel = os.path.join(host_dir, "host-sentinel.txt")
        with open(host_sentinel, "w", encoding="utf-8") as handle:
            handle.write("CORTEX_HOST_SENTINEL")

        probe = r"""
import json, os, socket, subprocess, sys
host_path = __HOST_SENTINEL__
result = {}
try:
    with open(".cortex-workspace-probe", "w", encoding="utf-8") as f: f.write("ok")
    with open(".cortex-workspace-probe", encoding="utf-8") as f:
        result["workspace_writable"] = f.read() == "ok"
except OSError:
    result["workspace_writable"] = False
host_visible = False
try:
    with open(host_path, encoding="utf-8") as f:
        host_visible = "CORTEX_HOST_SENTINEL" in f.read()
except OSError:
    pass
try:
    with open(host_path + ".write-probe", "w", encoding="utf-8") as f:
        f.write("must-not-write")
    os.unlink(host_path + ".write-probe")
    host_visible = True
except OSError:
    pass
result["host_path_inaccessible"] = not host_visible
try:
    with open("/usr/.cortex-runtime-write-probe", "w", encoding="utf-8") as f:
        f.write("must-not-write")
    os.unlink("/usr/.cortex-runtime-write-probe")
    result["runtime_readonly"] = False
except OSError:
    result["runtime_readonly"] = True
result["proc_dev_available"] = os.path.exists("/proc/self/status") and os.path.exists("/dev/null")
try:
    sock = socket.create_connection(("1.1.1.1", 53), timeout=1.0)
    sock.close()
    result["network_unreachable"] = False
except OSError:
    result["network_unreachable"] = True
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
result["detached_child_pid"] = child.pid
spawned = [child]
for _ in range(31):
    try:
        spawned.append(subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL))
    except OSError:
        break
result["process_cap_observed"] = len(spawned) < 32
print(json.dumps(result, sort_keys=True))
""".replace("__HOST_SENTINEL__", repr(host_sentinel))

        try:
            execution = run(code=probe, timeout_s=8.0, cpu_s=5, mem_mb=256,
                            nproc=16, require_full_isolation=True)
        except Exception as exc:
            checks["isolated_execution"] = "FAIL"
            return {"passed": False, "checks": checks,
                    "error": f"{type(exc).__name__}: {exc}"}

        checks["isolated_execution"] = "PASS" if execution.ok else "FAIL"
        limits = execution.limits or {}
        checks["aggregate_limits_enforced"] = (
            "PASS"
            if all(limits.get(name) is True for name in (
                "cgroup_v2_enforced", "cgroup_empty_after_cleanup",
                "aggregate_cpu_limit", "aggregate_memory_limit",
                "aggregate_process_limit"))
            else "FAIL"
        )
        try:
            observed = json.loads(execution.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            observed = {}
        for name in ("workspace_writable", "host_path_inaccessible",
                     "runtime_readonly", "proc_dev_available", "network_unreachable"):
            checks[name] = "PASS" if observed.get(name) is True else "FAIL"
        checks["detached_descendant_cleanup"] = (
            "PASS"
            if isinstance(observed.get("detached_child_pid"), int)
            and limits.get("cgroup_empty_after_cleanup") is True
            else "FAIL"
        )
        checks["aggregate_process_cap_observed"] = (
            "PASS" if observed.get("process_cap_observed") is True else "FAIL"
        )

    return {
        "passed": all(status == "PASS" for status in checks.values()),
        "checks": checks,
        "limitations": [
            "No seccomp syscall-filtering policy is certified.",
            "RLIMIT_NPROC is not a reliable aggregate process cap for privileged/root execution.",
            "Cgroup v2 CPU, memory and process-count limits are configured per execution; this qualification does not yet stress the numeric memory or CPU ceilings.",
            "Disk I/O bandwidth/IOPS limits are not configured.",
            "Detached descendants are checked through cgroup-wide cleanup; deliberate cgroup migration or kernel-level escape resistance is not certified.",
            "Network reachability is probed against 1.1.1.1:53 only.",
        ],
    }


def main() -> int:
    report = qualify_sandbox()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
