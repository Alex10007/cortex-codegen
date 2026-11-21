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
from pathlib import Path


class CgroupUnavailable(RuntimeError):
    """Required cgroup v2 delegation or controller is unavailable."""


class CgroupV2:
    REQUIRED_CONTROLLERS = frozenset({"cpu", "memory", "pids"})

    def __init__(self, path: Path):
        self.path = path
        self._closed = False

    @classmethod
    def create(cls, root: str | os.PathLike[str], *, mem_mb: int,
               nproc: int, cpu_quota_us: int = 100_000,
               cpu_period_us: int = 100_000) -> "CgroupV2":
        base = Path(root)
        if not base.is_absolute():
            raise CgroupUnavailable(
                "CORTEX_CGROUP_ROOT must be an existing absolute delegated cgroup v2 directory")
        try:
            resolved_base = base.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CgroupUnavailable(
                f"CORTEX_CGROUP_ROOT cannot be resolved safely: {exc}") from exc
        # Reject symlinks in every path component, not only at the final root
        # component. All subsequent child paths use this canonical directory.
        if base != resolved_base or not resolved_base.is_dir():
            raise CgroupUnavailable(
                "CORTEX_CGROUP_ROOT must not traverse symlink aliases and must be a directory")
        base = resolved_base
        controllers_file = base / "cgroup.controllers"
        subtree_file = base / "cgroup.subtree_control"
        if not controllers_file.is_file() or not subtree_file.is_file():
            raise CgroupUnavailable("configured cgroup root is not a cgroup v2 directory")
        available = set(controllers_file.read_text(encoding="ascii").split())
        enabled = set(subtree_file.read_text(encoding="ascii").split())
        missing = cls.REQUIRED_CONTROLLERS - available
        disabled = cls.REQUIRED_CONTROLLERS - enabled
        if missing or disabled:
            raise CgroupUnavailable(
                f"cgroup controllers unavailable={sorted(missing)} disabled={sorted(disabled)}; "
                "administrator must delegate and enable cpu,memory,pids")
        limits = {
            "mem_mb": mem_mb,
            "nproc": nproc,
            "cpu_quota_us": cpu_quota_us,
            "cpu_period_us": cpu_period_us,
        }
        if any(type(value) is not int or value < 1
               for value in limits.values()):
            raise ValueError("cgroup resource limits must be positive integers")
        # The CFS bandwidth controller requires a quota and period of at
        # least 1ms; cgroup v2 documents a 1ms–1s period range.
        if cpu_quota_us < 1_000:
            raise ValueError("cpu_quota_us must be at least 1000 microseconds")
        if not 1_000 <= cpu_period_us <= 1_000_000:
            raise ValueError("cpu_period_us must be between 1000 and 1000000 microseconds")

        path = base / ("cortex-" + uuid.uuid4().hex)
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            raise CgroupUnavailable(f"cannot create delegated execution cgroup: {exc}") from exc

        try:
            cls._write(path / "cpu.max", f"{cpu_quota_us} {cpu_period_us}")
            cls._write(path / "memory.max", str(mem_mb * 1024 * 1024))
            cls._write(path / "pids.max", str(nproc))
            swap_file = path / "memory.swap.max"
            if not swap_file.exists():
                raise CgroupUnavailable("memory.swap.max is unavailable; swap cannot be bounded")
            cls._write(swap_file, "0")
            oom_group = path / "memory.oom.group"
            if not oom_group.exists():
                raise CgroupUnavailable("memory.oom.group is unavailable")
            cls._write(oom_group, "1")
            if not (path / "cgroup.procs").exists():
                raise CgroupUnavailable("cgroup.procs is unavailable")
            if not (path / "cgroup.kill").exists():
                raise CgroupUnavailable("cgroup.kill is unavailable; process-tree cleanup cannot be guaranteed")
            if not (path / "cgroup.events").exists():
                raise CgroupUnavailable("cgroup.events is unavailable; process-tree exit cannot be verified")
            return cls(path)
        except Exception as setup_error:
            try:
                path.rmdir()
            except OSError as cleanup_error:
                setup_error.add_note(
                    f"failed to remove partially configured cgroup {path}: {cleanup_error}")
            raise

    @staticmethod
    def _write(path: Path, value: str) -> None:
        try:
            path.write_text(value, encoding="ascii")
        except OSError as exc:
            raise CgroupUnavailable(f"cannot configure {path.name}: {exc}") from exc

    def attach(self, pid: int) -> None:
        if self._closed:
            raise CgroupUnavailable("execution cgroup is already closed")
        self._write(self.path / "cgroup.procs", str(pid))

    def kill(self) -> None:
        if self._closed:
            return
        self._write(self.path / "cgroup.kill", "1")

    @staticmethod
    def _read_events(path: Path) -> dict[str, int]:
        """Parse cgroup.events without accepting ambiguous or partial evidence."""
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except OSError as exc:
            raise CgroupUnavailable(f"cannot read cgroup.events: {exc}") from exc
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
        events = self.path / "cgroup.events"
        if not events.exists():
            raise CgroupUnavailable("cgroup.events is unavailable; cannot verify process-tree exit")
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
            fields = self._read_events(events)
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
            lines = path.read_text(encoding="ascii").splitlines()
        except OSError as exc:
            raise CgroupUnavailable(f"cannot read {path.name}: {exc}") from exc
        counters: dict[str, int] = {}
        for line in lines:
            fields = line.split()
            if len(fields) != 2 or not fields[1].isdigit():
                raise CgroupUnavailable(f"malformed counter in {path.name}: {line!r}")
            if fields[0] in counters:
                raise CgroupUnavailable(
                    f"duplicate counter in {path.name}: {fields[0]!r}")
            counters[fields[0]] = int(fields[1])
        if not counters:
            raise CgroupUnavailable(f"no counters available in {path.name}")
        required = {
            "cpu.stat": {"nr_throttled", "throttled_usec"},
            "memory.events": {"oom", "oom_kill", "oom_group_kill"},
            "pids.events": {"max"},
        }.get(path.name, set())
        missing = required - counters.keys()
        if missing:
            raise CgroupUnavailable(
                f"missing required counters in {path.name}: {sorted(missing)}")
        return counters

    def snapshot(self) -> dict[str, object]:
        """Capture kernel-maintained resource events before the cgroup is removed."""
        cpu = self._read_counters(self.path / "cpu.stat")
        memory_events = self._read_counters(self.path / "memory.events")
        pids_events = self._read_counters(self.path / "pids.events")
        try:
            memory_text = (self.path / "memory.current").read_text(
                encoding="ascii").strip()
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
        self.wait_empty()
        try:
            self.path.rmdir()
        except OSError as exc:
            raise CgroupUnavailable(
                f"execution cgroup is not empty or could not be removed: {exc}") from exc
        self._closed = True


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
    base = Path(root)
    try:
        available = set((base / "cgroup.controllers").read_text(encoding="ascii").split())
        enabled = set((base / "cgroup.subtree_control").read_text(encoding="ascii").split())
    except OSError as exc:
        return {
            "available": False,
            "status": "FAIL",
            "reason": f"cannot inspect delegated cgroup subtree: {exc}",
            "controllers_required": sorted(CgroupV2.REQUIRED_CONTROLLERS),
            "io_limit": "NOT_CONFIGURED",
        }
    missing = CgroupV2.REQUIRED_CONTROLLERS - available
    disabled = CgroupV2.REQUIRED_CONTROLLERS - enabled
    ok = (base.is_absolute() and base.is_dir() and not base.is_symlink()
          and not missing and not disabled)
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
