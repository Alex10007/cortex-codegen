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
import signal
import stat
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from cortex.interfaces import ExecutionResult
from cortex.execution.resources import CgroupUnavailable, CgroupV2, configured_cgroup_root


class NetworkIsolationUnavailable(RuntimeError):
    pass


_UNSHARE_OK: bool | None = None
_BWRAP_OK: bool | None = None


def _supports_bwrap() -> bool:
    """Probe the actual namespace setup, not merely the bwrap executable."""
    global _BWRAP_OK
    if _BWRAP_OK is None:
        binary = shutil.which("bwrap")
        if not binary:
            _BWRAP_OK = False
        else:
            try:
                probe = subprocess.run(
                    [binary, "--die-with-parent", "--new-session",
                     "--unshare-all", "--ro-bind-try", "/usr", "/usr",
                     "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                     "--", "/usr/bin/true"],
                    capture_output=True, timeout=8)
                _BWRAP_OK = probe.returncode == 0
            except (OSError, subprocess.SubprocessError):
                _BWRAP_OK = False
    return _BWRAP_OK


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


def _limits(cpu_s: int, mem_mb: int, nproc: int, *,
           aggregate_process_limit: bool = False):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        b = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
        if aggregate_process_limit:
            # In strict mode the cgroup pids.max is the authoritative aggregate
            # limit. Raise the inherited soft RLIMIT_NPROC to its existing hard
            # ceiling so it cannot mask a cgroup pids.events:max observation.
            _, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
            resource.setrlimit(
                resource.RLIMIT_NPROC, (hard_limit, hard_limit))
        else:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _cgroup_bootstrap_source() -> str:
    """Return a syntactically valid launcher that waits for cgroup attachment."""
    return "\n".join((
        "import os,sys",
        "fd=int(sys.argv[1])",
        "if not os.read(fd,1): os._exit(126)",
        "os.close(fd)",
        "os.execvp(sys.argv[2],sys.argv[2:])",
    ))


def _copy_workspace_no_symlinks(source: str, destination: str) -> None:
    """Copy a workspace without following links or special files.

    Descriptor-relative opens plus O_NOFOLLOW prevent a symlink in the source
    tree from being dereferenced into host files while the copy is made. If the
    platform cannot provide these primitives, fail closed rather than falling
    back to path-based copying.
    """
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if os.name != "posix" or any(not hasattr(os, item) for item in required):
        raise RuntimeError("safe workspace copying requires POSIX O_NOFOLLOW")

    root_flags = os.O_RDONLY | os.O_DIRECTORY
    nofollow = os.O_NOFOLLOW
    source_fd = os.open(os.path.realpath(source), root_flags | nofollow)
    destination_fd = os.open(destination, root_flags | nofollow)
    copied_files = 0
    copied_bytes = 0
    max_files = 100_000
    max_bytes = 2 * 1024 * 1024 * 1024

    def copy_directory(src_fd: int, dst_fd: int) -> None:
        nonlocal copied_files, copied_bytes
        for name in os.listdir(src_fd):
            info = os.stat(name, dir_fd=src_fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"workspace contains a symlink: {name}")
            if stat.S_ISDIR(info.st_mode):
                os.mkdir(name, mode=0o700, dir_fd=dst_fd)
                child_src = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | nofollow,
                    dir_fd=src_fd)
                child_dst = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | nofollow,
                    dir_fd=dst_fd)
                try:
                    copy_directory(child_src, child_dst)
                finally:
                    os.close(child_src)
                    os.close(child_dst)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"workspace contains a non-regular file: {name}")
            copied_files += 1
            if copied_files > max_files:
                raise ValueError("workspace contains too many files")
            file_fd = os.open(name, os.O_RDONLY | nofollow, dir_fd=src_fd)
            try:
                actual = os.fstat(file_fd)
                if not stat.S_ISREG(actual.st_mode):
                    raise ValueError(f"workspace entry changed during copy: {name}")
                target_fd = os.open(
                    name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                    0o600, dir_fd=dst_fd)
                try:
                    with os.fdopen(os.dup(file_fd), "rb") as source_file, \
                            os.fdopen(os.dup(target_fd), "wb") as target_file:
                        while True:
                            chunk = source_file.read(1024 * 1024)
                            if not chunk:
                                break
                            copied_bytes += len(chunk)
                            if copied_bytes > max_bytes:
                                raise ValueError("workspace exceeds the 2 GiB copy limit")
                            target_file.write(chunk)
                    os.fchmod(target_fd, stat.S_IMODE(actual.st_mode) & 0o777)
                finally:
                    os.close(target_fd)
            finally:
                os.close(file_fd)

    try:
        copy_directory(source_fd, destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)



def _cleanup_execution(
    proc: subprocess.Popen,
    cgroup: CgroupV2 | None,
    readers: Sequence[threading.Thread],
    streams: Sequence[Any],
) -> tuple[dict[str, object], bool]:
    """Reap a launched process tree and its cgroup on every exit path."""
    failures: list[str] = []
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            failures.append(f"process-group termination failed: {exc}")
    elif proc.poll() is None:
        try:
            proc.kill()
        except OSError as exc:
            failures.append(f"process termination failed: {exc}")

    if cgroup is not None:
        try:
            cgroup.kill()
        except Exception as exc:
            failures.append(f"cgroup kill failed: {exc}")

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError as exc:
            failures.append(f"direct-child kill failed: {exc}")
        try:
            proc.wait(timeout=5)
        except Exception as exc:
            failures.append(f"direct-child reap failed: {exc}")
    except Exception as exc:
        failures.append(f"direct-child wait failed: {exc}")

    for reader in readers:
        try:
            reader.join(timeout=5)
        except Exception as exc:
            failures.append(f"output-reader join failed: {exc}")
    if any(reader.is_alive() for reader in readers):
        for stream in streams:
            try:
                stream.close()
            except Exception as exc:
                failures.append(f"output-pipe close failed: {exc}")
        for reader in readers:
            reader.join(timeout=1)
        if any(reader.is_alive() for reader in readers):
            failures.append("output readers remained alive after pipe closure")

    metrics: dict[str, object] = {}
    empty = False
    if cgroup is not None:
        try:
            cgroup.wait_empty()
            empty = True
        except Exception as exc:
            failures.append(f"cgroup did not become empty: {exc}")
        if empty:
            try:
                metrics = cgroup.snapshot()
            except Exception as exc:
                failures.append(f"cgroup evidence snapshot failed: {exc}")
        try:
            cgroup.close()
        except Exception as exc:
            failures.append(f"cgroup removal failed: {exc}")
            empty = False

    if failures:
        raise RuntimeError("sandbox cleanup incomplete: " + "; ".join(failures))
    return metrics, (cgroup is not None and empty)


def run(code: str | None = None, argv: Sequence[str] | None = None,
        workspace: str | None = None, timeout_s: float = 20.0,
        cpu_s: int = 15, mem_mb: int = 1024, nproc: int = 64,
        audit: Callable[[str, dict], None] | None = None,
        require_full_isolation: bool = False,
        cpu_quota_us: int = 100_000,
        cpu_period_us: int = 100_000) -> ExecutionResult:
    """Execute in a disposable workspace.

    When bubblewrap is available, use user/pid/network namespaces, a read-only
    runtime, and a dedicated writable workspace. The legacy rlimit fallback
    is explicitly degraded and must not be used for API-hosted untrusted code.
    """
    cmd = list(argv) if argv else [sys.executable, "-c", code or ""]
    with tempfile.TemporaryDirectory(prefix="cortex-sandbox-") as td:
        if workspace:
            _copy_workspace_no_symlinks(workspace, td)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": td, "TMPDIR": td,
               "SANDBOX": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        import site

        usersite = site.getusersitepackages()
        if os.path.isdir(usersite):
            env["PYTHONPATH"] = usersite
        bwrap = shutil.which("bwrap") if _supports_bwrap() else None
        if require_full_isolation and not bwrap:
            raise NetworkIsolationUnavailable(
                "full sandbox isolation requires working bubblewrap "
                "user/pid/network namespaces; refusing same-user host execution")

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
        cgroup = None
        if require_full_isolation:
            cgroup_root = configured_cgroup_root()
            if not cgroup_root:
                raise NetworkIsolationUnavailable(
                    "full sandbox isolation requires an administrator-delegated "
                    "cgroup v2 subtree configured with CORTEX_CGROUP_ROOT")
            try:
                cgroup = CgroupV2.create(
                    cgroup_root, mem_mb=mem_mb, nproc=nproc,
                    cpu_quota_us=cpu_quota_us, cpu_period_us=cpu_period_us)
            except (CgroupUnavailable, OSError, ValueError) as exc:
                raise NetworkIsolationUnavailable(
                    f"aggregate cgroup resource enforcement unavailable: {exc}") from exc

        start = time.time()
        timed_out = False
        max_output_bytes = 4 * 1024 * 1024  # per stream; memory use is bounded
        captured = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}

        def drain(stream, key: str) -> None:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                remaining = max_output_bytes - len(captured[key])
                if remaining > 0:
                    captured[key].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[key] = True

        if cgroup is not None:
            # Hold the interpreter before exec so no untrusted instruction runs
            # until the launcher PID has entered the delegated cgroup.
            try:
                gate_read, gate_write = os.pipe()
                bootstrap = _cgroup_bootstrap_source()
                launch_cmd = [sys.executable, "-c", bootstrap, str(gate_read), *full_cmd]
            except BaseException as setup_error:
                try:
                    cgroup.kill()
                    cgroup.wait_empty()
                    cgroup.close()
                except Exception as cleanup_error:
                    setup_error.add_note(
                        f"cgroup cleanup after launch preparation failure also failed: {cleanup_error}")
                raise
            try:
                try:
                    proc = subprocess.Popen(
                        launch_cmd, cwd=(td if not bwrap else "/"), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        preexec_fn=_limits(
                            cpu_s, mem_mb, nproc,
                            aggregate_process_limit=True),
                        start_new_session=(os.name == "posix"),
                        pass_fds=(gate_read,),
                    )
                except Exception as launch_error:
                    os.close(gate_write)
                    try:
                        cgroup.kill()
                        cgroup.wait_empty()
                        cgroup.close()
                    except Exception as cleanup_error:
                        launch_error.add_note(
                            f"cgroup cleanup after launch failure also failed: {cleanup_error}")
                    raise
            finally:
                os.close(gate_read)
            try:
                cgroup.attach(proc.pid)
                os.write(gate_write, b"1")
            except Exception as exc:
                os.close(gate_write)
                try:
                    _cleanup_execution(proc, cgroup, [], ())
                except Exception as cleanup_error:
                    exc.add_note(
                        f"sandbox cleanup after cgroup-attach failure also failed: {cleanup_error}")
                raise NetworkIsolationUnavailable(
                    f"could not attach sandbox launcher to cgroup before execution: {exc}") from exc
            os.close(gate_write)
        else:
            proc = subprocess.Popen(
                full_cmd, cwd=(td if not bwrap else "/"), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=_limits(cpu_s, mem_mb, nproc),
                start_new_session=(os.name == "posix"),
            )
        stdout_reader = threading.Thread(
            target=drain, args=(proc.stdout, "stdout"), daemon=True)
        stderr_reader = threading.Thread(
            target=drain, args=(proc.stderr, "stderr"), daemon=True)
        readers: list[threading.Thread] = []
        try:
            stdout_reader.start()
            readers.append(stdout_reader)
            stderr_reader.start()
            readers.append(stderr_reader)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
        except BaseException as primary_error:
            try:
                cgroup_metrics, cgroup_empty_after_cleanup = _cleanup_execution(
                    proc, cgroup, readers, (proc.stdout, proc.stderr))
            except Exception as cleanup_error:
                primary_error.add_note(
                    f"sandbox cleanup also failed: {cleanup_error}")
            raise
        else:
            cgroup_metrics, cgroup_empty_after_cleanup = _cleanup_execution(
                proc, cgroup, readers, (proc.stdout, proc.stderr))

        rc = proc.returncode
        out = bytes(captured["stdout"]).decode("utf-8", errors="replace")
        err = bytes(captured["stderr"]).decode("utf-8", errors="replace")
        if truncated["stdout"]:
            out += "\n[stdout truncated at 4 MiB]"
        if truncated["stderr"]:
            err += "\n[stderr truncated at 4 MiB]"
        if (require_full_isolation and bwrap and rc != 0
                and ((err or "").lower().startswith("bwrap:")
                     or "namespace" in (err or "").lower()
                     or "operation not permitted" in (err or "").lower())):
            raise NetworkIsolationUnavailable(
                "bubblewrap could not establish required namespaces; "
                "refusing degraded execution")
        dur = time.time() - start

    if filesystem_isolated:
        isolation = "bwrap+user-pid-net-ipc-uts+rlimits+cgroup-v2-cpu-memory-pids+readonly-runtime+workspace"
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
                "full_isolation_required": require_full_isolation,
                "max_output_bytes_per_stream": max_output_bytes,
                "stdout_truncated": truncated["stdout"],
                "stderr_truncated": truncated["stderr"],
                "cgroup_v2_enforced": cgroup is not None,
                "cgroup_empty_after_cleanup": cgroup_empty_after_cleanup,
                "aggregate_cpu_limit": cgroup is not None,
                "aggregate_memory_limit": cgroup is not None,
                "aggregate_process_limit": cgroup is not None,
                "cgroup_metrics": cgroup_metrics,
                "cpu_quota_us": cpu_quota_us if cgroup is not None else None,
                "cpu_period_us": cpu_period_us if cgroup is not None else None,
                "io_limit": "NOT_CONFIGURED"},
    )
    if require_full_isolation and not filesystem_isolated:
        raise NetworkIsolationUnavailable(
            "full filesystem/user namespace isolation was not established")
    if audit:
        audit("sandbox.run", {"cmd": full_cmd[:3], "ok": result.ok,
                              "timed_out": timed_out, "isolation": isolation})
    return result
