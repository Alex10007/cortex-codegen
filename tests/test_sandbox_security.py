"""Adversarial regression tests for workspace and timeout isolation."""
import os
import subprocess
import sys
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
