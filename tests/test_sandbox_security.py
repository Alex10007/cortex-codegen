"""Adversarial regression tests for workspace and timeout isolation."""
import os
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path

import pytest

from cortex.execution.sandbox import run


@pytest.mark.parametrize("link_kind", ["file", "directory"])
def test_workspace_copy_rejects_symlinks_without_reading_targets(link_kind):
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
        source = Path(root) / "workspace"
        source.mkdir()
        secret = Path(outside) / "secret.txt"
        secret.write_text("HOST-SECRET-MUST-NOT-ENTER-SANDBOX")
        link = source / "external"
        if link_kind == "file":
            link.symlink_to(secret)
        else:
            external_dir = Path(outside) / "external-dir"
            external_dir.mkdir()
            (external_dir / "secret.txt").write_text("HOST-SECRET-MUST-NOT-ENTER-SANDBOX")
            link.symlink_to(external_dir, target_is_directory=True)

        with pytest.raises(ValueError, match="symlink"):
            run(code="print('should not execute')", workspace=str(source))


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").exists(), reason="Linux /proc is required")
def test_timeout_kills_descendant_processes():
    with tempfile.TemporaryDirectory() as td:
        pid_file = Path(td) / "child.pid"
        child_code = "import time; time.sleep(60)"
        parent_code = (
            "import subprocess,sys,time,pathlib\n"
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))\n"
            "time.sleep(60)\n"
        )
        result = run(code=parent_code, timeout_s=1.0)
        assert result.timed_out
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":  # terminated; awaiting init's reap
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"descendant process {child_pid} survived timeout")


def test_workspace_copy_rejects_fifo_and_other_special_files():
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is unavailable on this platform")
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td) / "workspace"
        workspace.mkdir()
        os.mkfifo(workspace / "blocked.fifo")
        with pytest.raises(ValueError, match="non-regular"):
            run(code="pass", workspace=str(workspace))


def test_stdout_and_stderr_capture_are_bounded():
    size = 6 * 1024 * 1024
    result = run(
        code=(
            "import sys\n"
            f"sys.stdout.write('O' * {size})\n"
            f"sys.stderr.write('E' * {size})\n"
        ),
        timeout_s=10.0,
    )
    assert result.exit_code == 0
    assert len(result.stdout.encode("utf-8")) < 4 * 1024 * 1024 + 100
    assert len(result.stderr.encode("utf-8")) < 4 * 1024 * 1024 + 100
    assert result.limits["stdout_truncated"] is True
    assert result.limits["stderr_truncated"] is True
    assert "stdout truncated" in result.stdout
    assert "stderr truncated" in result.stderr


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").exists(), reason="Linux /proc is required")
def test_background_descendant_is_killed_when_parent_exits():
    with tempfile.TemporaryDirectory() as td:
        pid_file = Path(td) / "child.pid"
        child_code = "import time; time.sleep(60)"
        parent_code = (
            "import subprocess,sys,pathlib\n"
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))\n"
        )
        result = run(code=parent_code, timeout_s=5.0)
        assert result.ok
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"background descendant {child_pid} survived parent exit")


def test_cleanup_execution_reaps_process_and_closes_cgroup():
    from cortex.execution.sandbox import _cleanup_execution

    class FakeCgroup:
        def __init__(self):
            self.events = []
        def kill(self):
            self.events.append("kill")
        def wait_empty(self):
            self.events.append("wait_empty")
        def snapshot(self):
            self.events.append("snapshot")
            return {"verified": True}
        def close(self):
            self.events.append("close")

    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    group = FakeCgroup()
    metrics, empty = _cleanup_execution(proc, group, [], (proc.stdout, proc.stderr))
    assert proc.returncode is not None
    assert metrics == {"verified": True}
    assert empty is True
    assert group.events == ["kill", "wait_empty", "snapshot", "close"]


def test_cleanup_execution_surfaces_cgroup_kill_failure_after_reaping():
    from cortex.execution.sandbox import _cleanup_execution

    class FailingCgroup:
        def __init__(self):
            self.closed = False
        def kill(self):
            raise OSError("injected cgroup.kill failure")
        def wait_empty(self):
            pass
        def snapshot(self):
            return {}
        def close(self):
            self.closed = True

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    group = FailingCgroup()
    with pytest.raises(RuntimeError, match="cgroup kill failed"):
        _cleanup_execution(proc, group, [], (proc.stdout, proc.stderr))
    assert proc.returncode is not None
    assert group.closed is True


def test_reader_start_failure_still_reaps_launched_process(monkeypatch):
    from cortex.execution import sandbox

    monkeypatch.setattr(sandbox, "_supports_bwrap", lambda: False)
    monkeypatch.setattr(sandbox, "_supports_unshare", lambda: False)
    real_popen = subprocess.Popen
    launched = []

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(sandbox.subprocess, "Popen", capture_popen)
    real_start = threading.Thread.start
    starts = 0

    def fail_second_reader(thread):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("injected second-reader startup failure")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_second_reader)
    with pytest.raises(RuntimeError, match="injected second-reader startup failure"):
        sandbox.run(code="import time; time.sleep(30)", timeout_s=10.0)

    assert len(launched) == 1
    assert launched[0].returncode is not None
