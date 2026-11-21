import tempfile
from pathlib import Path

from cortex.execution.sandbox import run


def test_sandbox_runs_simple_code():
    r = run(code="print('hello cortex')")
    assert r.ok and r.exit_code == 0
    assert "hello cortex" in r.stdout
    assert r.isolation


def test_sandbox_timeout_enforced():
    r = run(code="import time; time.sleep(30)", timeout_s=3.0)
    assert not r.ok
    assert r.timed_out


def test_sandbox_network_is_isolated_when_kernel_allows():
    r = run(code=(
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('93.184.216.34', 80))\n"
        "    print('NETWORK-REACHABLE')\n"
        "except OSError as e:\n"
        "    print('NETWORK-BLOCKED:', type(e).__name__)\n"
    ), timeout_s=15.0)
    assert r.ok, r.stderr
    if r.limits["network_blocked"]:
        assert "NETWORK-BLOCKED" in r.stdout


def test_sandbox_memory_limit_enforced():
    r = run(code="x = 'A' * (400 * 1024 * 1024)", mem_mb=128, timeout_s=30)
    assert not r.ok


def test_sandbox_workspace_copy_isolated():
    with tempfile.TemporaryDirectory() as src:
        (Path(src) / "marker.py").write_text("print('from-marker')\n")
        r = run(argv=["python3", "-c", "import marker"], workspace=src)
        assert r.ok and "from-marker" in r.stdout
        assert Path(src, "marker.py").exists()
