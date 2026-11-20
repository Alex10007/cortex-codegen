"""Unit tests for fail-closed cgroup v2 resource enforcement."""
from pathlib import Path

import pytest

from cortex.execution.resources import CgroupUnavailable, CgroupV2, probe_cgroup_v2


def make_fake_cgroup_root(root: Path) -> Path:
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory pids io")
    (root / "cgroup.subtree_control").write_text("cpu memory pids")
    return root


def make_fake_child(path: Path) -> None:
    # The caller's monkeypatched mkdir has already created this directory.
    for name in ("cpu.max", "memory.max", "pids.max", "memory.swap.max",
                 "cgroup.procs", "cgroup.kill", "cgroup.events", "memory.oom.group"):
        (path / name).write_text("")


def test_probe_requires_explicit_delegated_root(monkeypatch):
    monkeypatch.delenv("CORTEX_CGROUP_ROOT", raising=False)
    result = probe_cgroup_v2()
    assert result["available"] is False
    assert result["status"] == "UNSUPPORTED"
    assert result["io_limit"] == "NOT_CONFIGURED"


def test_probe_reports_missing_or_disabled_controllers(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    (root / "cgroup.controllers").write_text("cpu memory")
    (root / "cgroup.subtree_control").write_text("cpu")
    monkeypatch.setenv("CORTEX_CGROUP_ROOT", str(root))
    result = probe_cgroup_v2()
    assert result["available"] is False
    assert result["missing_controllers"] == ["pids"]
    assert result["disabled_controllers"] == ["memory", "pids"]


def test_create_applies_cpu_memory_and_process_limits(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_mkdir = Path.mkdir
    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path.parent == root:
            make_fake_child(path)
        return result
    monkeypatch.setattr(Path, "mkdir", mkdir)
    group = CgroupV2.create(root, mem_mb=256, nproc=12, cpu_quota_us=50_000)
    assert (group.path / "cpu.max").read_text() == "50000 100000"
    assert (group.path / "memory.max").read_text() == str(256 * 1024 * 1024)
    assert (group.path / "pids.max").read_text() == "12"
    (group.path / "cpu.stat").write_text("usage_usec 1234\nnr_periods 10\nnr_throttled 3\nthrottled_usec 2000\n")
    (group.path / "memory.events").write_text("low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\noom_group_kill 0\n")
    (group.path / "pids.events").write_text("max 4\n")
    (group.path / "memory.current").write_text("0")
    snapshot = group.snapshot()
    assert snapshot["cpu_stat"]["nr_throttled"] == 3
    assert snapshot["memory_events"]["oom_kill"] == 1
    assert snapshot["pids_events"]["max"] == 4
    group.attach(1234)
    assert (group.path / "cgroup.procs").read_text() == "1234"
    group.kill()
    assert (group.path / "cgroup.kill").read_text() == "1"
    (group.path / "cgroup.events").write_text("populated 0\nfrozen 0\n")
    group.wait_empty()
    # Regular files emulate cgroup virtual files in this unit test.
    for item in group.path.iterdir():
        item.unlink()
    group.close()
    assert not group.path.exists()


def test_create_fails_closed_if_cgroup_kill_missing(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_mkdir = Path.mkdir
    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path.parent == root:
            for name in ("cpu.max", "memory.max", "pids.max", "cgroup.procs", "cgroup.events", "memory.swap.max", "memory.oom.group"):
                (path / name).write_text("")
        return result
    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(CgroupUnavailable, match="cgroup.kill"):
        CgroupV2.create(root, mem_mb=128, nproc=4)


def test_create_rejects_nonpositive_limits(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    with pytest.raises(ValueError):
        CgroupV2.create(root, mem_mb=0, nproc=4)



def test_full_isolation_refuses_to_launch_without_delegated_cgroup(monkeypatch):
    from cortex.execution import sandbox

    monkeypatch.setattr(sandbox, "_supports_bwrap", lambda: True)
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "configured_cgroup_root", lambda: None)
    with pytest.raises(sandbox.NetworkIsolationUnavailable, match="CORTEX_CGROUP_ROOT"):
        sandbox.run(code="raise SystemExit('must not execute')",
                    require_full_isolation=True)


def test_create_rejects_invalid_cpu_quota_types_and_values(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    for kwargs in (
        {"cpu_quota_us": 0},
        {"cpu_period_us": -1},
        {"cpu_quota_us": True},
        {"cpu_period_us": 100_000.0},
    ):
        with pytest.raises(ValueError, match="positive integers"):
            CgroupV2.create(root, mem_mb=128, nproc=4, **kwargs)


def test_snapshot_rejects_missing_required_kernel_counters(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_mkdir = Path.mkdir
    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path.parent == root:
            make_fake_child(path)
        return result
    # The test creates a cgroup using regular files as a minimal kernel fixture.
    from cortex.execution import resources
    original_create = CgroupV2.create
    group_path = root / "cortex-test"
    group_path.mkdir()
    for name in ("cpu.stat", "memory.events", "pids.events", "memory.current"):
        (group_path / name).write_text("")
    (group_path / "cpu.stat").write_text("usage_usec 10\\n")
    (group_path / "memory.events").write_text("oom 0\\noom_kill 0\\noom_group_kill 0\\n")
    (group_path / "pids.events").write_text("max 0\\n")
    (group_path / "memory.current").write_text("0")
    group = CgroupV2(group_path)
    with pytest.raises(CgroupUnavailable, match="nr_throttled"):
        group.snapshot()
    for item in group_path.iterdir():
        item.unlink()
    group_path.rmdir()
