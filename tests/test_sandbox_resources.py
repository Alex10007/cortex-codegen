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
    # Regular files provide a minimal fixture for the kernel counter interface.
    group_path = root / "cortex-test"
    group_path.mkdir()
    for name in ("cpu.stat", "memory.events", "pids.events", "memory.current"):
        (group_path / name).write_text("")
    (group_path / "cpu.stat").write_text("usage_usec 10\n")
    (group_path / "memory.events").write_text("oom 0\noom_kill 0\noom_group_kill 0\n")
    (group_path / "pids.events").write_text("max 0\n")
    (group_path / "memory.current").write_text("0")
    group = CgroupV2(group_path)
    with pytest.raises(CgroupUnavailable, match="nr_throttled"):
        group.snapshot()
    for item in group_path.iterdir():
        item.unlink()
    group_path.rmdir()


def test_create_rejects_cpu_period_outside_kernel_range(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    for period in (999, 1_000_001):
        with pytest.raises(ValueError, match="cpu_period_us"):
            CgroupV2.create(
                root, mem_mb=128, nproc=4, cpu_period_us=period)


def test_counter_parser_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "cpu.stat"
    path.write_text("nr_throttled 1\nthrottled_usec 2\nnr_throttled 3\n")
    with pytest.raises(CgroupUnavailable, match="duplicate counter"):
        CgroupV2._read_counters(path)


def test_snapshot_rejects_non_decimal_memory_current(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-test"
    group_path.mkdir()
    (group_path / "cpu.stat").write_text(
        "nr_throttled 0\nthrottled_usec 0\n")
    (group_path / "memory.events").write_text(
        "oom 0\noom_kill 0\noom_group_kill 0\n")
    (group_path / "pids.events").write_text("max 0\n")
    (group_path / "memory.current").write_text("+1")
    with pytest.raises(CgroupUnavailable, match="non-negative decimal integer"):
        CgroupV2(group_path).snapshot()
    for item in group_path.iterdir():
        item.unlink()
    group_path.rmdir()


def test_create_rejects_cpu_quota_below_kernel_minimum(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    with pytest.raises(ValueError, match="cpu_quota_us"):
        CgroupV2.create(
            root, mem_mb=128, nproc=4, cpu_quota_us=999)



@pytest.mark.parametrize("contents, message", [
    ("frozen 0\n", "missing required populated"),
    ("populated 0\npopulated 1\n", "duplicate event"),
    ("populated\n", "malformed event"),
    ("populated 2\n", "invalid boolean event"),
    ("populated -1\n", "malformed event"),
    ("populated 0 extra\n", "malformed event"),
])
def test_cgroup_events_reject_ambiguous_or_invalid_records(tmp_path, contents, message):
    path = tmp_path / "cgroup.events"
    path.write_text(contents)
    with pytest.raises(CgroupUnavailable, match=message):
        CgroupV2._read_events(path)


def test_wait_empty_requires_explicit_valid_populated_zero(tmp_path):
    group_path = tmp_path / "cortex-test"
    group_path.mkdir()
    events = group_path / "cgroup.events"
    events.write_text("frozen 0\n")
    group = CgroupV2(group_path)
    with pytest.raises(CgroupUnavailable, match="missing required populated"):
        group.wait_empty(timeout_s=0.01)
    events.write_text("populated 0\nfrozen 0\n")
    group.wait_empty(timeout_s=0.01)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_wait_empty_rejects_invalid_timeout_values(tmp_path, timeout):
    group_path = tmp_path / "cortex-test"
    group_path.mkdir()
    (group_path / "cgroup.events").write_text("populated 0\n")
    with pytest.raises(ValueError, match="finite positive"):
        CgroupV2(group_path).wait_empty(timeout_s=timeout)



def test_create_rejects_delegated_root_reached_through_symlink(tmp_path):
    real_root = make_fake_cgroup_root(tmp_path / "delegated-real")
    alias = tmp_path / "delegated-alias"
    alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(CgroupUnavailable, match="symlink aliases"):
        CgroupV2.create(alias, mem_mb=128, nproc=4)


def test_uuid_collision_does_not_reuse_or_remove_existing_cgroup(tmp_path, monkeypatch):
    from cortex.execution import resources

    root = make_fake_cgroup_root(tmp_path / "delegated")
    existing = root / "cortex-fixed"
    existing.mkdir()
    sentinel = existing / "owner-sentinel"
    sentinel.write_text("owned by another execution")
    monkeypatch.setattr(
        resources.uuid, "uuid4",
        lambda: type("FixedUUID", (), {"hex": "fixed"})(),
    )

    with pytest.raises(CgroupUnavailable, match="cannot create delegated execution cgroup"):
        CgroupV2.create(root, mem_mb=128, nproc=4)

    assert sentinel.read_text() == "owned by another execution"
    assert existing.is_dir()
