"""Behavioral qualification for the Bubblewrap execution path used by Cortex."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from cortex.execution.sandbox import NetworkIsolationUnavailable, _supports_bwrap, run


def qualify_sandbox() -> dict[str, Any]:
    """Exercise workspace, host-path, runtime-mount, proc/dev and network boundaries.

    This focused probe does not certify seccomp, cgroups, resource-limit
    enforcement, namespace escape resistance, or kernel exploit resistance.
    """
    checks = {
        "bwrap_available": "PASS" if _supports_bwrap() else "FAIL",
        "isolated_execution": "NOT_RUN",
        "workspace_writable": "NOT_RUN",
        "host_path_inaccessible": "NOT_RUN",
        "runtime_readonly": "NOT_RUN",
        "proc_dev_available": "NOT_RUN",
        "network_unreachable": "NOT_RUN",
    }
    if checks["bwrap_available"] != "PASS":
        return {"passed": False, "checks": checks,
                "limitations": ["Bubblewrap is unavailable or its namespace probe failed."]}

    with tempfile.TemporaryDirectory(prefix="cortex-qualification-host-") as host_dir:
        host_sentinel = os.path.join(host_dir, "host-sentinel.txt")
        with open(host_sentinel, "w", encoding="utf-8") as handle:
            handle.write("CORTEX_HOST_SENTINEL")

        probe = r"""
import json, os, socket
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
        try:
            observed = json.loads(execution.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            observed = {}
        for name in ("workspace_writable", "host_path_inaccessible",
                     "runtime_readonly", "proc_dev_available", "network_unreachable"):
            checks[name] = "PASS" if observed.get(name) is True else "FAIL"

    return {
        "passed": all(status == "PASS" for status in checks.values()),
        "checks": checks,
        "limitations": [
            "No seccomp syscall-filtering policy is certified.",
            "RLIMIT_NPROC is not a reliable aggregate process cap for privileged/root execution.",
            "CPU, memory, process-count and disk/output limits are not independently stress-qualified.",
            "No cgroup enforcement or deliberate namespace/session escape resistance is certified.",
            "Network reachability is probed against 1.1.1.1:53 only.",
        ],
    }


def main() -> int:
    report = qualify_sandbox()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
