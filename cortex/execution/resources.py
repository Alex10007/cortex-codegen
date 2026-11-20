"""Linux cgroup v2 resource enforcement for untrusted execution.

A delegated cgroup subtree must be provisioned by the service administrator and
selected with CORTEX_CGROUP_ROOT. The module never modifies the host cgroup
controller hierarchy itself.
"""
from __future__ import annotations

import os
import shutil
import uuid
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
        if not base.is_absolute() or not base.is_dir():
            raise CgroupUnavailable("CORTEX_CGROUP_ROOT must be an existing absolute delegated cgroup v2 directory")
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
        if mem_mb < 1 or nproc < 1 or cpu_quota_us < 1 or cpu_period_us < 1:
            raise ValueError("cgroup resource limits must be positive")

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
            if swap_file.exists():
                cls._write(swap_file, "0")
            if not (path / "cgroup.procs").exists():
                raise CgroupUnavailable("cgroup.procs is unavailable")
            if not (path / "cgroup.kill").exists():
                raise CgroupUnavailable("cgroup.kill is unavailable; process-tree cleanup cannot be guaranteed")
            return cls(path)
        except Exception:
            try:
                path.rmdir()
            except OSError:
                pass
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.path.rmdir()
        except OSError as exc:
            raise CgroupUnavailable(
                f"execution cgroup is not empty or could not be removed: {exc}") from exc


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
    ok = base.is_absolute() and base.is_dir() and not missing and not disabled
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
