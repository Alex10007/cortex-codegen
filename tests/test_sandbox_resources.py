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
    path.mkdir()
    for name in ("cpu.max", "memory.max", "pids.max", "memory.swap.max",
                 "cgroup.procs", "cgroup.kill"):
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
    group.attach(1234)
    assert (group.path / "cgroup.procs").read_text() == "1234"
    group.kill()
    assert (group.path / "cgroup.kill").read_text() == "1"
    group.close()
    assert not group.path.exists()


def test_create_fails_closed_if_cgroup_kill_missing(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_mkdir = Path.mkdir
    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        if path.parent == root:
            for name in ("cpu.max", "memory.max", "pids.max", "cgroup.procs"):
                (path / name).write_text("")
        return result
    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(CgroupUnavailable, match="cgroup.kill"):
        CgroupV2.create(root, mem_mb=128, nproc=4)


def test_create_rejects_nonpositive_limits(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    with pytest.raises(ValueError):
        CgroupV2.create(root, mem_mb=0, nproc=4)
