"""Unit tests for fail-closed cgroup v2 resource enforcement."""
import os
import threading
import time
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
    real_mkdir = os.mkdir
    def mkdir(name, mode=0o777, *, dir_fd=None):
        result = real_mkdir(name, mode, dir_fd=dir_fd)
        if isinstance(name, str) and name.startswith("cortex-") and dir_fd is not None:
            parent = Path(os.readlink(f"/proc/self/fd/{dir_fd}"))
            make_fake_child(parent / name)
        return result
    monkeypatch.setattr(os, "mkdir", mkdir)
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
    # Emulate cgroupfs pseudo-files, which disappear with the cgroup directory.
    real_rmdir = os.rmdir
    def rmdir(name, *, dir_fd=None):
        if name == group.path.name:
            for item in group.path.iterdir():
                item.unlink()
        return real_rmdir(name, dir_fd=dir_fd)
    monkeypatch.setattr(os, "rmdir", rmdir)
    group.close()
    assert not group.path.exists()


def test_create_fails_closed_if_cgroup_kill_missing(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_mkdir = os.mkdir
    def mkdir(name, mode=0o777, *, dir_fd=None):
        result = real_mkdir(name, mode, dir_fd=dir_fd)
        if isinstance(name, str) and name.startswith("cortex-") and dir_fd is not None:
            path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / name
            for control in ("cpu.max", "memory.max", "pids.max", "cgroup.procs",
                            "cgroup.events", "memory.swap.max", "memory.oom.group"):
                (path / control).write_text("")
        return result
    monkeypatch.setattr(os, "mkdir", mkdir)
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



def test_probe_rejects_delegated_root_reached_through_intermediate_symlink(tmp_path, monkeypatch):
    make_fake_cgroup_root(tmp_path / "delegated-real")
    alias_parent = tmp_path / "parent-alias"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    aliased_root = alias_parent / "delegated-real"
    monkeypatch.setenv("CORTEX_CGROUP_ROOT", str(aliased_root))

    result = probe_cgroup_v2()

    assert result["available"] is False
    assert result["status"] == "FAIL"
    assert "symlink aliases" in result["reason"]


def test_probe_and_create_reject_missing_delegated_root_consistently(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("CORTEX_CGROUP_ROOT", str(missing))

    result = probe_cgroup_v2()

    assert result["available"] is False
    assert result["status"] == "FAIL"
    with pytest.raises(CgroupUnavailable, match="cannot be resolved safely"):
        CgroupV2.create(missing, mem_mb=128, nproc=4)


def test_create_rejects_delegated_root_reached_through_symlink(tmp_path):
    real_root = make_fake_cgroup_root(tmp_path / "delegated-real")
    alias = tmp_path / "delegated-alias"
    alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(CgroupUnavailable, match="symlink aliases"):
        CgroupV2.create(alias, mem_mb=128, nproc=4)


def test_cgroup_refuses_operations_after_path_replacement(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    group = CgroupV2(group_path)

    moved = root / "cortex-original"
    group_path.rename(moved)
    group_path.mkdir()
    (group_path / "cgroup.events").write_text("populated 0\n")

    with pytest.raises(CgroupUnavailable, match="identity changed"):
        group.kill()
    with pytest.raises(CgroupUnavailable, match="identity changed"):
        group.wait_empty()
    with pytest.raises(CgroupUnavailable, match="identity changed"):
        group.close()

    assert group_path.is_dir()
    assert moved.is_dir()


def test_cgroup_rejects_symlink_replacement_without_touching_target(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    original = root / "cortex-owned"
    original.mkdir()
    group = CgroupV2(original)
    replacement = root / "replacement"
    replacement.mkdir()
    sentinel = replacement / "owner-sentinel"
    sentinel.write_text("untouched")
    original.rmdir()
    original.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(CgroupUnavailable, match="identity changed|not the original directory"):
        group.close()

    assert sentinel.read_text() == "untouched"
    assert replacement.is_dir()


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


def test_cgroup_resource_operations_reject_symlinked_control_files(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    external = tmp_path / "external-control"
    external.write_text("untouched")
    (group_path / "cgroup.kill").symlink_to(external)

    group = CgroupV2(group_path)

    with pytest.raises(CgroupUnavailable, match="cannot configure cgroup.kill"):
        group.kill()

    assert external.read_text() == "untouched"
    group._close_directory_fds()


def test_cgroup_manager_retains_directory_descriptors_for_owned_inode(tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    (group_path / "cgroup.events").write_text("populated 0\n")

    group = CgroupV2(group_path)
    directory_identity = (group._identity[0], group._identity[1])
    descriptor_identity = os.fstat(group._dir_fd)
    assert (descriptor_identity.st_dev, descriptor_identity.st_ino) == directory_identity
    assert os.fstat(group._parent_fd).st_ino == os.stat(root).st_ino

    group._close_directory_fds()
    assert group._dir_fd == -1
    assert group._parent_fd == -1
    assert group_path.exists()


def test_create_uses_descriptor_relative_setup_and_rolls_back_failed_configuration(
        tmp_path, monkeypatch):
    from cortex.execution import resources

    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_mkdir = os.mkdir
    created = []

    def mkdir(name, mode=0o777, *, dir_fd=None):
        result = real_mkdir(name, mode, dir_fd=dir_fd)
        if isinstance(name, str) and name.startswith("cortex-") and dir_fd is not None:
            created.append(name)
        return result

    def fail_configuration(directory_fd, name, value):
        raise CgroupUnavailable("injected descriptor-relative setup failure")

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(resources.CgroupV2, "_write_to_fd", staticmethod(fail_configuration))

    with pytest.raises(CgroupUnavailable, match="injected descriptor-relative"):
        CgroupV2.create(root, mem_mb=128, nproc=4)

    assert len(created) == 1
    assert not (root / created[0]).exists()


def test_create_refuses_symlinked_initial_control_file_and_rolls_back(
        tmp_path, monkeypatch):
    from cortex.execution import resources

    root = make_fake_cgroup_root(tmp_path / "delegated")
    external = tmp_path / "external-memory-limit"
    external.write_text("untouched")
    real_mkdir = os.mkdir
    created = []

    def mkdir(name, mode=0o777, *, dir_fd=None):
        result = real_mkdir(name, mode, dir_fd=dir_fd)
        if isinstance(name, str) and name.startswith("cortex-") and dir_fd is not None:
            child = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / name
            created.append(child)
            for control in ("cpu.max", "memory.max", "pids.max", "memory.swap.max",
                            "memory.oom.group", "cgroup.procs", "cgroup.kill",
                            "cgroup.events"):
                (child / control).write_text("")
            (child / "memory.max").unlink()
            (child / "memory.max").symlink_to(external)
        return result

    real_rmdir = os.rmdir
    def rmdir(name, *, dir_fd=None):
        child = root / name
        if child in created and child.exists():
            # Temporary regular files emulate cgroupfs pseudo-files in this unit fixture.
            for item in child.iterdir():
                item.unlink()
        return real_rmdir(name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(os, "rmdir", rmdir)
    with pytest.raises(CgroupUnavailable, match="cannot configure memory.max"):
        CgroupV2.create(root, mem_mb=128, nproc=4)

    assert external.read_text() == "untouched"
    assert len(created) == 1
    assert not created[0].exists()


def test_delegated_root_component_walk_rejects_intermediate_symlink(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    delegated = make_fake_cgroup_root(real_root / "delegated")
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(CgroupUnavailable, match="symlink aliases"):
        CgroupV2._open_delegated_root(alias / "delegated")


def test_delegated_root_component_open_rejects_swap_after_lstat(tmp_path, monkeypatch):
    delegated = make_fake_cgroup_root(tmp_path / "delegated")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    backup = tmp_path / "delegated-original"
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (not swapped and path == delegated.name and dir_fd is not None):
            delegated.rename(backup)
            delegated.symlink_to(replacement, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    try:
        with pytest.raises(CgroupUnavailable, match="without following links"):
            CgroupV2._open_delegated_root(delegated)
        assert swapped
        assert delegated.is_symlink()
        assert backup.is_dir()
    finally:
        if delegated.is_symlink():
            delegated.unlink()
        if backup.exists():
            backup.rename(delegated)



def test_directory_descriptor_cleanup_attempts_both_fds_after_first_close_error(
        tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    group = CgroupV2(group_path)
    child_fd, parent_fd = group._dir_fd, group._parent_fd
    real_close = os.close
    attempted = []

    def flaky_close(descriptor):
        attempted.append(descriptor)
        if descriptor == child_fd:
            real_close(descriptor)
            raise OSError("injected close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", flaky_close)
    failures = group._close_directory_fds()

    assert attempted == [child_fd, parent_fd]
    assert len(failures) == 1
    assert "injected close failure" in failures[0]
    assert group._dir_fd == -1
    assert group._parent_fd == -1
    with pytest.raises(OSError):
        os.fstat(child_fd)
    with pytest.raises(OSError):
        os.fstat(parent_fd)


def test_constructor_identity_failure_closes_both_transferred_descriptors(
        tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    parent_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    child_fd = os.open(group_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    real_close = os.close
    closed = []

    def recording_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "close", recording_close)
    with pytest.raises(CgroupUnavailable, match="changed while opening"):
        CgroupV2(
            group_path, expected_identity=(0, 0),
            parent_fd=parent_fd, directory_fd=child_fd)

    assert set(closed) == {parent_fd, child_fd}
    with pytest.raises(OSError):
        os.fstat(parent_fd)
    with pytest.raises(OSError):
        os.fstat(child_fd)


def test_delegated_root_walk_closes_new_descriptor_if_previous_close_fails(
        tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    real_close_fd = CgroupV2._close_fd
    closed_traversal = []
    injected = False

    def flaky_close_fd(descriptor, label):
        nonlocal injected
        error = real_close_fd(descriptor, label)
        if label.startswith("previous delegated-root") and not injected:
            injected = True
            return "injected previous-descriptor close failure"
        if label == "delegated-root traversal":
            closed_traversal.append(descriptor)
        return error

    monkeypatch.setattr(
        CgroupV2, "_close_fd", staticmethod(flaky_close_fd))
    with pytest.raises(CgroupUnavailable, match="cannot release previous delegated-root"):
        CgroupV2._open_delegated_root(root)

    assert injected
    assert len(closed_traversal) == 1
    with pytest.raises(OSError):
        os.fstat(closed_traversal[0])


def test_close_is_idempotent_after_successful_removal(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    (group_path / "cgroup.events").write_text("populated 0\n")
    group = CgroupV2(group_path)
    real_rmdir = os.rmdir
    removals = []

    def rmdir(name, *, dir_fd=None):
        if name == group_path.name:
            removals.append(name)
            for item in group_path.iterdir():
                item.unlink()
        return real_rmdir(name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "rmdir", rmdir)
    group.close()
    group.close()

    assert removals == [group_path.name]
    assert group._closed is True
    assert group._dir_fd == -1
    assert group._parent_fd == -1
    assert not group_path.exists()



def test_cgroup_operations_remain_bound_to_pinned_parent_after_path_replacement(
        tmp_path):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    (group_path / "cgroup.events").write_text("populated 0\\n")
    group = CgroupV2(group_path)

    moved_root = tmp_path / "delegated-original"
    root.rename(moved_root)
    replacement_root = make_fake_cgroup_root(root)
    replacement_group = replacement_root / group_path.name
    replacement_group.mkdir()
    sentinel = replacement_group / "owner-sentinel"
    sentinel.write_text("replacement entry must survive")

    # The manager must operate only through its pinned original parent fd.
    group.close()

    assert not (moved_root / group_path.name).exists()
    assert sentinel.read_text() == "replacement entry must survive"
    assert replacement_group.is_dir()


def test_concurrent_close_calls_are_serialized(tmp_path, monkeypatch):
    root = make_fake_cgroup_root(tmp_path / "delegated")
    group_path = root / "cortex-owned"
    group_path.mkdir()
    (group_path / "cgroup.events").write_text("populated 0\\n")
    group = CgroupV2(group_path)

    first_waiting = threading.Event()
    release_first = threading.Event()
    wait_calls = []
    removal_calls = []
    errors = []
    real_rmdir = os.rmdir

    def controlled_wait_empty(timeout_s=5.0):
        wait_calls.append(threading.get_ident())
        if len(wait_calls) == 1:
            first_waiting.set()
            if not release_first.wait(timeout=3):
                raise AssertionError("test did not release first close")
        # Re-check through the manager's real descriptor-bound event reader.
        fields = CgroupV2._read_events_text(group._read_at("cgroup.events"))
        assert fields["populated"] == 0

    def recording_rmdir(name, *, dir_fd=None):
        if name == group_path.name:
            removal_calls.append(name)
            for item in group_path.iterdir():
                item.unlink()
        return real_rmdir(name, dir_fd=dir_fd)

    monkeypatch.setattr(group, "wait_empty", controlled_wait_empty)
    monkeypatch.setattr(os, "rmdir", recording_rmdir)

    def close_worker():
        try:
            group.close()
        except Exception as exc:  # captured for assertion in the parent thread
            errors.append(exc)

    first = threading.Thread(target=close_worker)
    second = threading.Thread(target=close_worker)
    first.start()
    assert first_waiting.wait(timeout=3)
    second.start()
    time.sleep(0.05)
    release_first.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(wait_calls) == 1
    assert removal_calls == [group_path.name]
    assert group._closed is True
    assert not group_path.exists()
