"""Linux cgroup v2 resource enforcement for untrusted execution.

A delegated cgroup subtree must be provisioned by the service administrator and
selected with CORTEX_CGROUP_ROOT. The module never modifies the host cgroup
controller hierarchy itself.
"""
from __future__ import annotations

import os
import uuid
import time
import math
import stat
from pathlib import Path


class CgroupUnavailable(RuntimeError):
    """Required cgroup v2 delegation or controller is unavailable."""


class CgroupV2:
    REQUIRED_CONTROLLERS = frozenset({"cpu", "memory", "pids"})

    def __init__(self, path: Path, *, expected_identity: tuple[int, int] | None = None,
                 parent_fd: int | None = None, directory_fd: int | None = None):
        self.path = path
        self._closed = False
        self._identity = expected_identity or self._path_identity(path)
        self._parent_fd = -1
        self._dir_fd = -1
        directory_flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                           | getattr(os, "O_CLOEXEC", 0))
        try:
            self._parent_fd = (parent_fd if parent_fd is not None
                               else os.open(path.parent, directory_flags))
            self._dir_fd = (directory_fd if directory_fd is not None else os.open(
                path.name, directory_flags | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._parent_fd))
            info = os.fstat(self._dir_fd)
            if (info.st_dev, info.st_ino) != self._identity:
                raise CgroupUnavailable(
                    "execution cgroup changed while opening its directory")
        except Exception as setup_error:
            failures = self._close_directory_fds()
            for failure in failures:
                setup_error.add_note(f"descriptor cleanup failed: {failure}")
            raise

    def _close_directory_fds(self) -> list[str]:
        """Attempt every owned descriptor close; safe to call repeatedly."""
        failures: list[str] = []
        for attribute in ("_dir_fd", "_parent_fd"):
            descriptor = getattr(self, attribute, -1)
            if descriptor < 0:
                continue
            # Invalidate ownership before close: retrying close after EINTR can
            # close an unrelated descriptor if the kernel already released it.
            setattr(self, attribute, -1)
            try:
                os.close(descriptor)
            except OSError as exc:
                failures.append(f"{attribute} fd {descriptor}: {exc}")
        return failures

    @staticmethod
    def _close_fd(descriptor: int, label: str) -> str | None:
        if descriptor < 0:
            return None
        try:
            os.close(descriptor)
        except OSError as exc:
            return f"{label} fd {descriptor}: {exc}"
        return None

    @staticmethod
    def _raise_cleanup_failures(message: str, failures: list[str]) -> None:
        if failures:
            raise CgroupUnavailable(message + "; ".join(failures))

    def _read_at(self, name: str) -> str:
        self._assert_owned_path()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=self._dir_fd)
            with os.fdopen(descriptor, "r", encoding="ascii") as stream:
                return stream.read()
        except OSError as exc:
            raise CgroupUnavailable(
                f"cannot read {name} from owned cgroup: {exc}") from exc

    def _write_at(self, name: str, value: str) -> None:
        self._assert_owned_path()
        flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=self._dir_fd)
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(value)
        except OSError as exc:
            raise CgroupUnavailable(
                f"cannot configure {name} in owned cgroup: {exc}") from exc

    @staticmethod
    def _path_identity(path: Path) -> tuple[int, int]:
        try:
            info = path.lstat()
        except OSError as exc:
            raise CgroupUnavailable(f"cannot inspect execution cgroup identity: {exc}") from exc
        if not path.is_dir() or path.is_symlink():
            raise CgroupUnavailable("execution cgroup path is not the original directory")
        return info.st_dev, info.st_ino

    def _assert_owned_path(self) -> None:
        """Fail closed if the pathname no longer resolves to this execution's directory."""
        if self._path_identity(self.path) != self._identity:
            raise CgroupUnavailable(
                "execution cgroup path identity changed; refusing operation on an unowned cgroup")

    @staticmethod
    def _resolve_delegated_root(root: str | os.PathLike[str]) -> Path:
        """Apply identical canonical-path and symlink policy to probe and create."""
        base = Path(root)
        if not base.is_absolute():
            raise CgroupUnavailable(
                "CORTEX_CGROUP_ROOT must be an existing absolute delegated cgroup v2 directory")
        try:
            resolved_base = base.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CgroupUnavailable(
                f"CORTEX_CGROUP_ROOT cannot be resolved safely: {exc}") from exc
        # Reject aliases in every component, not only a symlink at the leaf.
        if base != resolved_base or not resolved_base.is_dir():
            raise CgroupUnavailable(
                "CORTEX_CGROUP_ROOT must not traverse symlink aliases and must be a directory")
        return resolved_base

    @staticmethod
    def _open_delegated_root(root: str | os.PathLike[str]) -> tuple[Path, int]:
        """Open every delegated-root path component without following symlinks."""
        base = CgroupV2._resolve_delegated_root(root)
        flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                 | getattr(os, "O_CLOEXEC", 0))
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        current_fd = -1
        try:
            current_fd = os.open("/", flags)
            for component in base.parts[1:]:
                if component in ("", ".", ".."):
                    raise CgroupUnavailable(
                        "delegated cgroup root contains an unsafe path component")
                try:
                    entry = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                except OSError as exc:
                    raise CgroupUnavailable(
                        f"cannot inspect delegated-root component {component}: {exc}") from exc
                if not stat.S_ISDIR(entry.st_mode):
                    raise CgroupUnavailable(
                        f"delegated-root component {component} is not a real directory")
                try:
                    next_fd = os.open(
                        component, flags | nofollow, dir_fd=current_fd)
                except OSError as exc:
                    raise CgroupUnavailable(
                        f"cannot open delegated-root component {component} without following links: {exc}") from exc
                try:
                    opened = os.fstat(next_fd)
                    if (not stat.S_ISDIR(opened.st_mode)
                            or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)):
                        raise CgroupUnavailable(
                            f"delegated-root component {component} changed identity while opening")
                except Exception as acquisition_error:
                    close_error = CgroupV2._close_fd(
                        next_fd, f"new delegated-root component {component}")
                    if close_error:
                        acquisition_error.add_note(
                            f"descriptor cleanup failed: {close_error}")
                    raise
                previous_fd = current_fd
                current_fd = next_fd
                close_error = CgroupV2._close_fd(
                    previous_fd, f"previous delegated-root component before {component}")
                if close_error:
                    raise CgroupUnavailable(
                        f"cannot release previous delegated-root descriptor: {close_error}")
            final_info = os.fstat(current_fd)
            if not stat.S_ISDIR(final_info.st_mode):
                raise CgroupUnavailable("delegated cgroup root is not a directory")
            result_fd = current_fd
            current_fd = -1
            return base, result_fd
        finally:
            close_error = CgroupV2._close_fd(
                current_fd, "delegated-root traversal")
            if close_error:
                # A cleanup failure must not replace an exception already in flight.
                # When returning successfully, however, the descriptor is transferred.
                import sys
                active_error = sys.exception()
                if active_error is not None:
                    active_error.add_note(f"descriptor cleanup failed: {close_error}")
                else:
                    raise CgroupUnavailable(
                        f"cannot close delegated-root traversal descriptor: {close_error}")

    @staticmethod
    def _read_from_fd(directory_fd: int, name: str) -> str:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            with os.fdopen(descriptor, "r", encoding="ascii") as stream:
                return stream.read()
        except OSError as exc:
            raise CgroupUnavailable(f"cannot read delegated cgroup file {name}: {exc}") from exc

    @staticmethod
    def _write_to_fd(directory_fd: int, name: str, value: str) -> None:
        flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(value)
        except OSError as exc:
            raise CgroupUnavailable(f"cannot configure {name}: {exc}") from exc

    @staticmethod
    def _require_control_file(directory_fd: int, name: str) -> None:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise CgroupUnavailable(f"required cgroup control {name} is unavailable: {exc}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise CgroupUnavailable(f"required cgroup control {name} is not a regular control file")

    @classmethod
    def create(cls, root: str | os.PathLike[str], *, mem_mb: int,
               nproc: int, cpu_quota_us: int = 100_000,
               cpu_period_us: int = 100_000) -> "CgroupV2":
        base = cls._resolve_delegated_root(root)
        limits = {
            "mem_mb": mem_mb,
            "nproc": nproc,
            "cpu_quota_us": cpu_quota_us,
            "cpu_period_us": cpu_period_us,
        }
        if any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("cgroup resource limits must be positive integers")
        if cpu_quota_us < 1_000:
            raise ValueError("cpu_quota_us must be at least 1000 microseconds")
        if not 1_000 <= cpu_period_us <= 1_000_000:
            raise ValueError("cpu_period_us must be between 1000 and 1000000 microseconds")

        directory_flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                           | getattr(os, "O_CLOEXEC", 0))
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        parent_fd = -1
        child_fd = -1
        child_name = "cortex-" + uuid.uuid4().hex
        path = base / child_name
        child_identity: tuple[int, int] | None = None
        transferred = False
        try:
            # Walk and pin every delegated-root component without following links.
            opened_base, parent_fd = cls._open_delegated_root(root)
            if opened_base != base:
                raise CgroupUnavailable("delegated cgroup root changed during acquisition")

            available = set(cls._read_from_fd(parent_fd, "cgroup.controllers").split())
            enabled = set(cls._read_from_fd(parent_fd, "cgroup.subtree_control").split())
            missing = cls.REQUIRED_CONTROLLERS - available
            disabled = cls.REQUIRED_CONTROLLERS - enabled
            if missing or disabled:
                raise CgroupUnavailable(
                    f"cgroup controllers unavailable={sorted(missing)} disabled={sorted(disabled)}; "
                    "administrator must delegate and enable cpu,memory,pids")

            # mkdirat semantics: collision fails without following or reusing an existing entry.
            try:
                os.mkdir(child_name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise CgroupUnavailable(
                    "cannot create delegated execution cgroup without reusing an existing entry"
                ) from exc
            entry_info = os.stat(child_name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(entry_info.st_mode):
                raise CgroupUnavailable("new execution cgroup entry is not a directory")
            child_identity = (entry_info.st_dev, entry_info.st_ino)
            child_fd = os.open(child_name, directory_flags | nofollow, dir_fd=parent_fd)
            child_info = os.fstat(child_fd)
            if (not stat.S_ISDIR(child_info.st_mode)
                    or (child_info.st_dev, child_info.st_ino) != child_identity):
                raise CgroupUnavailable(
                    "new execution cgroup changed identity while opening")

            cls._write_to_fd(child_fd, "cpu.max", f"{cpu_quota_us} {cpu_period_us}")
            cls._write_to_fd(child_fd, "memory.max", str(mem_mb * 1024 * 1024))
            cls._write_to_fd(child_fd, "pids.max", str(nproc))
            cls._write_to_fd(child_fd, "memory.swap.max", "0")
            cls._write_to_fd(child_fd, "memory.oom.group", "1")
            for required in ("cgroup.procs", "cgroup.kill", "cgroup.events"):
                cls._require_control_file(child_fd, required)

            manager_parent_fd, manager_child_fd = parent_fd, child_fd
            parent_fd = child_fd = -1
            manager = cls(path, expected_identity=child_identity,
                          parent_fd=manager_parent_fd, directory_fd=manager_child_fd)
            transferred = True
            return manager
        except Exception as setup_error:
            close_error = cls._close_fd(child_fd, "partially created cgroup")
            child_fd = -1
            if close_error:
                setup_error.add_note(f"descriptor cleanup failed: {close_error}")
            if parent_fd >= 0 and child_identity is not None:
                try:
                    current = os.stat(child_name, dir_fd=parent_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) != child_identity:
                        raise CgroupUnavailable(
                            "partially configured cgroup identity changed; refusing rollback removal")
                    os.rmdir(child_name, dir_fd=parent_fd)
                except (OSError, CgroupUnavailable) as cleanup_error:
                    setup_error.add_note(
                        f"failed to safely remove partially configured cgroup {path}: {cleanup_error}")
            raise
        finally:
            if not transferred:
                failures = []
                close_error = cls._close_fd(child_fd, "untransferred child cgroup")
                if close_error:
                    failures.append(close_error)
                child_fd = -1
                close_error = cls._close_fd(parent_fd, "untransferred delegated-root")
                if close_error:
                    failures.append(close_error)
                parent_fd = -1
                if failures:
                    import sys
                    active_error = sys.exception()
                    if active_error is not None:
                        for failure in failures:
                            active_error.add_note(f"descriptor cleanup failed: {failure}")
                    else:
                        cls._raise_cleanup_failures(
                            "failed to release cgroup setup descriptors: ", failures)

    @staticmethod
    def _write(path: Path, value: str) -> None:
        try:
            path.write_text(value, encoding="ascii")
        except OSError as exc:
            raise CgroupUnavailable(f"cannot configure {path.name}: {exc}") from exc

    def attach(self, pid: int) -> None:
        self._assert_owned_path()
        if self._closed:
            raise CgroupUnavailable("execution cgroup is already closed")
        self._write_at("cgroup.procs", str(pid))

    def kill(self) -> None:
        self._assert_owned_path()
        if self._closed:
            return
        self._write_at("cgroup.kill", "1")

    @staticmethod
    def _read_events(path: Path) -> dict[str, int]:
        """Parse cgroup.events without accepting ambiguous or partial evidence."""
        try:
            contents = path.read_text(encoding="ascii")
        except OSError as exc:
            raise CgroupUnavailable(f"cannot read cgroup.events: {exc}") from exc
        return CgroupV2._read_events_text(contents)

    @staticmethod
    def _read_events_text(contents: str) -> dict[str, int]:
        lines = contents.splitlines()
        fields: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) != 2 or not parts[1].isascii() or not parts[1].isdecimal():
                raise CgroupUnavailable(
                    f"malformed event in cgroup.events: {line!r}")
            key, raw_value = parts
            if key in fields:
                raise CgroupUnavailable(
                    f"duplicate event in cgroup.events: {key!r}")
            value = int(raw_value)
            if key in {"populated", "frozen"} and value not in {0, 1}:
                raise CgroupUnavailable(
                    f"invalid boolean event in cgroup.events: {key}={value}")
            fields[key] = value
        if "populated" not in fields:
            raise CgroupUnavailable(
                "cgroup.events is missing required populated field")
        return fields

    def wait_empty(self, timeout_s: float = 5.0) -> None:
        self._assert_owned_path()
        events = "cgroup.events"
        if type(timeout_s) not in (int, float):
            raise ValueError("timeout_s must be a finite positive number")
        try:
            timeout_value = float(timeout_s)
        except OverflowError as exc:
            raise ValueError("timeout_s must be a finite positive number") from exc
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        deadline = time.monotonic() + timeout_value
        while True:
            fields = self._read_events_text(self._read_at(events))
            if fields["populated"] == 0:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CgroupUnavailable(
                    "cgroup still contains processes after cgroup.kill")
            time.sleep(min(0.02, remaining))

    @staticmethod
    def _read_counters(path: Path) -> dict[str, int]:
        try:
            contents = path.read_text(encoding="ascii")
        except OSError as exc:
            raise CgroupUnavailable(f"cannot read {path.name}: {exc}") from exc
        return CgroupV2._read_counters_text(contents, path.name)

    @staticmethod
    def _read_counters_text(contents: str, name: str) -> dict[str, int]:
        lines = contents.splitlines()
        counters: dict[str, int] = {}
        for line in lines:
            fields = line.split()
            if len(fields) != 2 or not fields[1].isdigit():
                raise CgroupUnavailable(f"malformed counter in {name}: {line!r}")
            if fields[0] in counters:
                raise CgroupUnavailable(
                    f"duplicate counter in {name}: {fields[0]!r}")
            counters[fields[0]] = int(fields[1])
        if not counters:
            raise CgroupUnavailable(f"no counters available in {name}")
        required = {
            "cpu.stat": {"nr_throttled", "throttled_usec"},
            "memory.events": {"oom", "oom_kill", "oom_group_kill"},
            "pids.events": {"max"},
        }.get(name, set())
        missing = required - counters.keys()
        if missing:
            raise CgroupUnavailable(
                f"missing required counters in {name}: {sorted(missing)}")
        return counters

    def snapshot(self) -> dict[str, object]:
        """Capture kernel-maintained resource events before the cgroup is removed."""
        self._assert_owned_path()
        cpu = self._read_counters_text(self._read_at("cpu.stat"), "cpu.stat")
        memory_events = self._read_counters_text(
            self._read_at("memory.events"), "memory.events")
        pids_events = self._read_counters_text(self._read_at("pids.events"), "pids.events")
        try:
            memory_text = self._read_at("memory.current").strip()
            if not memory_text.isascii() or not memory_text.isdecimal():
                raise ValueError("memory.current must be a non-negative decimal integer")
            memory_current = int(memory_text)
        except (OSError, ValueError) as exc:
            raise CgroupUnavailable(f"cannot read memory.current: {exc}") from exc
        return {
            "cpu_stat": cpu,
            "memory_events": memory_events,
            "pids_events": pids_events,
            "memory_current_bytes": memory_current,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._assert_owned_path()
        self.wait_empty()
        self._assert_owned_path()
        try:
            current = os.stat(
                self.path.name, dir_fd=self._parent_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != self._identity:
                raise CgroupUnavailable(
                    "execution cgroup path identity changed; refusing descriptor-relative removal")
            os.rmdir(self.path.name, dir_fd=self._parent_fd)
        except OSError as exc:
            raise CgroupUnavailable(
                f"execution cgroup is not empty or could not be removed: {exc}") from exc
        self._closed = True
        failures = self._close_directory_fds()
        self._raise_cleanup_failures(
            "cgroup was removed but descriptor closure reported errors: ", failures)


def configured_cgroup_root() -> str | None:
    """Return an explicitly delegated subtree; never guess a host cgroup path."""
    return os.environ.get("CORTEX_CGROUP_ROOT") or None


def probe_cgroup_v2() -> dict[str, object]:
    root = configured_cgroup_root()
    if root is None:
        return {
            "available": False,
            "status": "UNSUPPORTED",
            "reason": "CORTEX_CGROUP_ROOT is not configured",
            "controllers_required": sorted(CgroupV2.REQUIRED_CONTROLLERS),
            "io_limit": "NOT_CONFIGURED",
        }
    try:
        base, root_fd = CgroupV2._open_delegated_root(root)
        try:
            available = set(CgroupV2._read_from_fd(
                root_fd, "cgroup.controllers").split())
            enabled = set(CgroupV2._read_from_fd(
                root_fd, "cgroup.subtree_control").split())
        except Exception as probe_error:
            close_error = CgroupV2._close_fd(root_fd, "delegated-root probe")
            if close_error:
                probe_error.add_note(f"descriptor cleanup failed: {close_error}")
            raise
        else:
            close_error = CgroupV2._close_fd(root_fd, "delegated-root probe")
            if close_error:
                raise CgroupUnavailable(
                    f"cannot release delegated-root probe descriptor: {close_error}")
    except (OSError, RuntimeError, CgroupUnavailable) as exc:
        return {
            "available": False,
            "status": "FAIL",
            "reason": f"cannot inspect delegated cgroup subtree: {exc}",
            "controllers_required": sorted(CgroupV2.REQUIRED_CONTROLLERS),
            "io_limit": "NOT_CONFIGURED",
        }
    missing = CgroupV2.REQUIRED_CONTROLLERS - available
    disabled = CgroupV2.REQUIRED_CONTROLLERS - enabled
    ok = not missing and not disabled
    return {
        "available": ok,
        "status": "PASS" if ok else "FAIL",
        "controllers_available": sorted(available),
        "controllers_enabled": sorted(enabled),
        "controllers_required": sorted(CgroupV2.REQUIRED_CONTROLLERS),
        "missing_controllers": sorted(missing),
        "disabled_controllers": sorted(disabled),
        "io_limit": "NOT_CONFIGURED",
    }
