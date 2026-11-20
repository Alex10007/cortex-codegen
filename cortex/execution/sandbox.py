"""Sandboxed execution of generated code.

Isolation layers (measured, reported per run):
- temporary working copy of the workspace (host tree is never touched)
- cleared environment, no inherited shell
- resource limits: CPU seconds, address-space bytes, process/file limits
- hard wall-clock timeout
- network namespace isolation via `unshare -rn` when the kernel permits it;
  otherwise runs rlimit-only and the run report says so explicitly.
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from cortex.interfaces import ExecutionResult


class NetworkIsolationUnavailable(RuntimeError):
    pass


_UNSHARE_OK: bool | None = None


def _supports_unshare() -> bool:
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        try:
            r = subprocess.run(
                ["unshare", "-rn", sys.executable, "-c", "pass"],
                capture_output=True, timeout=15,
            )
            _UNSHARE_OK = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _UNSHARE_OK = False
    return _UNSHARE_OK


def _limits(cpu_s: int, mem_mb: int, nproc: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        b = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run(code: str | None = None, argv: Sequence[str] | None = None,
        workspace: str | None = None, timeout_s: float = 20.0,
        cpu_s: int = 15, mem_mb: int = 1024, nproc: int = 64,
        audit: Callable[[str, dict], None] | None = None,
        require_full_isolation: bool = False) -> ExecutionResult:
    """Execute in a disposable workspace.

    When bubblewrap is available, use user/pid/network namespaces, a read-only
    runtime, and a dedicated writable workspace. The legacy rlimit fallback
    is explicitly degraded and must not be used for API-hosted untrusted code.
    """
    cmd = list(argv) if argv else [sys.executable, "-c", code or ""]
    with tempfile.TemporaryDirectory(prefix="cortex-sandbox-") as td:
        if workspace:
            for item in Path(workspace).iterdir():
                dst = Path(td) / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": td, "TMPDIR": td,
               "SANDBOX": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        import site

        usersite = site.getusersitepackages()
        if os.path.isdir(usersite):
            env["PYTHONPATH"] = usersite
        bwrap = shutil.which("bwrap")
        if require_full_isolation and not bwrap:
            raise NetworkIsolationUnavailable(
                "full sandbox isolation requires bubblewrap; refusing "
                "same-user host execution")

        netns = _supports_unshare()
        filesystem_isolated = False
        if bwrap:
            full_cmd = [
                bwrap, "--die-with-parent", "--new-session", "--unshare-all",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--dir", "/workspace", "--bind", td, "/workspace",
                "--chdir", "/workspace",
            ]
            for runtime_path in ("/usr", "/bin", "/lib", "/lib64"):
                if os.path.exists(runtime_path):
                    full_cmd.extend(["--ro-bind-try", runtime_path, runtime_path])
            for runtime_path in (sys.prefix, sys.base_prefix):
                if (runtime_path and os.path.exists(runtime_path)
                        and not any(runtime_path == root
                                    or runtime_path.startswith(root + os.sep)
                                    for root in ("/usr", "/bin", "/lib", "/lib64"))):
                    full_cmd.extend(["--ro-bind-try", runtime_path, runtime_path])
            full_cmd.extend([
                "--clearenv", "--setenv", "PATH",
                "/usr/local/bin:/usr/bin:/bin",
                "--setenv", "HOME", "/workspace",
                "--setenv", "TMPDIR", "/tmp",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "--", *cmd,
            ])
            filesystem_isolated = True
        elif netns:
            full_cmd = ["unshare", "-rn"] + cmd
        else:
            full_cmd = cmd
        start = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                full_cmd, cwd=(td if not bwrap else "/"), env=env,
                capture_output=True, text=True, timeout=timeout_s,
                preexec_fn=_limits(cpu_s, mem_mb, nproc),
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
            if (require_full_isolation and bwrap and rc != 0
                    and ("namespace" in (err or "").lower()
                         or "operation not permitted" in (err or "").lower())):
                raise NetworkIsolationUnavailable(
                    "bubblewrap could not establish required namespaces; "
                    "refusing degraded execution")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc, out, err = -9, exc.stdout or "", exc.stderr or ""
        dur = time.time() - start

    if filesystem_isolated:
        isolation = "bwrap+user-pid-net-ipc-uts+rlimits+readonly-runtime+workspace"
    else:
        isolation = ("netns+rlimits+temporary-workspace" if netns
                     else "rlimits+temporary-workspace")
    result = ExecutionResult(
        ok=(not timed_out and rc == 0),
        exit_code=rc,
        stdout=out or "",
        stderr=err or "",
        duration_s=round(dur, 3),
        timed_out=timed_out,
        isolation=isolation,
        limits={"cpu_seconds": cpu_s, "mem_mb": mem_mb, "nproc": nproc,
                "timeout_s": timeout_s,
                "network_blocked": bool(netns or filesystem_isolated),
                "filesystem_isolated": filesystem_isolated,
                "user_namespace_isolated": filesystem_isolated,
                "full_isolation_required": require_full_isolation},
    )
    if require_full_isolation and not filesystem_isolated:
        raise NetworkIsolationUnavailable(
            "full filesystem/user namespace isolation was not established")
    if audit:
        audit("sandbox.run", {"cmd": full_cmd[:3], "ok": result.ok,
                              "timed_out": timed_out, "isolation": isolation})
    return result
