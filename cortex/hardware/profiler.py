"""Real hardware profiling. Every value is read from the host, never assumed."""
from __future__ import annotations

import os
import platform
import shutil
import sys

from cortex.interfaces import HardwareProfile


def _read_meminfo() -> dict[str, float]:
    values: dict[str, float] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            tokens = rest.split()
            if tokens:
                values[key.strip()] = float(tokens[0]) / (1024.0 ** 2)  # KiB -> GiB
    return values


def profile_hardware() -> HardwareProfile:
    mem = _read_meminfo()
    total = mem.get("MemTotal", 0.0)
    available = mem.get("MemAvailable", total)
    disk_free = shutil.disk_usage(os.getcwd()).free / (1024.0 ** 3)

    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "none"
        torch_version = torch.__version__
    except Exception:
        has_cuda, gpu_name, torch_version = False, "none", "not-installed"

    from os import cpu_count as _cpu_count_impl

    return HardwareProfile(
        cpu_cores=_cpu_count_impl() or 1,
        ram_total_gb=round(total, 2),
        ram_available_gb=round(available, 2),
        disk_free_gb=round(disk_free, 2),
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        platform=f"{platform.system()} {platform.machine()}",
        python_version=sys.version.split()[0],
        torch_version=torch_version,
    )


def can_train_locally(profile: HardwareProfile, param_count: int) -> tuple[bool, str]:
    """Decide whether a parameter count is trainable on this host, with reasons."""
    # AdamW full-precision needs roughly 16 bytes/param (param + grad + 2 moments)
    # plus activation headroom.
    ram_needed_gb = (param_count * 16 / (1024.0 ** 3)) * 1.5
    if profile.ram_available_gb < ram_needed_gb:
        reason = (
            f"needs ~{ram_needed_gb:.1f} GB RAM (16 B/param x1.5 activations headroom), "
            f"available {profile.ram_available_gb:.1f} GB"
        )
        return False, reason
    if not profile.has_cuda and param_count > 200_000_000:
        return False, (
            f"CPU-only host; {param_count:,} params would take impractically long "
            "without accelerators"
        )
    return True, "fits host memory budget"
