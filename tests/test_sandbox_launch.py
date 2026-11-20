"""Regression tests for the strict sandbox's cgroup-gated process launch."""
import resource

from cortex.execution import sandbox


def test_cgroup_bootstrap_is_valid_python_with_real_line_boundaries():
    source = sandbox._cgroup_bootstrap_source()
    compile(source, "<cortex-cgroup-bootstrap>", "exec")
    assert len(source.splitlines()) == 5
    assert "os.execvp(sys.argv[2],sys.argv[2:])" in source


def test_strict_cgroup_uses_kernel_pids_limit_not_rlimit_nproc(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "getrlimit",
                        lambda which: (16, 4096) if which == resource.RLIMIT_NPROC
                        else (0, 0))
    monkeypatch.setattr(resource, "setrlimit",
                        lambda which, limits: calls.append((which, limits)))

    sandbox._limits(5, 256, 16, aggregate_process_limit=True)()

    assert (resource.RLIMIT_NPROC, (4096, 4096)) in calls


def test_degraded_sandbox_retains_rlimit_nproc(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "setrlimit",
                        lambda which, limits: calls.append((which, limits)))

    sandbox._limits(5, 256, 16)()

    assert (resource.RLIMIT_NPROC, (16, 16)) in calls
