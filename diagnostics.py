"""Diagnostic helpers for TUI status panels."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class DiagnosticSnapshot:
    captured_at: str
    python_version: str
    platform: str
    cpu_count: int
    memory_total_mb: int | None
    memory_available_mb: int | None
    gpu_present: bool
    gpu_label: str | None

    def as_payload(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "memory_total_mb": self.memory_total_mb,
            "memory_available_mb": self.memory_available_mb,
            "gpu_present": self.gpu_present,
            "gpu_label": self.gpu_label,
        }


def capture_diagnostics() -> DiagnosticSnapshot:
    memory_total: int | None = None
    memory_available: int | None = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        memory_total = int(vm.total / (1024 * 1024))
        memory_available = int(vm.available / (1024 * 1024))
    except Exception:
        memory_total = None
        memory_available = None

    gpu_present = False
    gpu_label: str | None = None
    try:
        import torch  # type: ignore

        gpu_present = bool(torch.cuda.is_available())
        if gpu_present:
            gpu_label = str(torch.cuda.get_device_name(0))
    except Exception:
        gpu_present = False
        gpu_label = None

    return DiagnosticSnapshot(
        captured_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        cpu_count=int(psutil_cpu_count()),
        memory_total_mb=memory_total,
        memory_available_mb=memory_available,
        gpu_present=gpu_present,
        gpu_label=gpu_label,
    )


def psutil_cpu_count() -> int:
    try:
        import psutil  # type: ignore

        count = psutil.cpu_count(logical=True)
        if isinstance(count, int) and count > 0:
            return count
    except Exception:
        pass
    return int(os_cpu_count_fallback())


def os_cpu_count_fallback() -> int:
    import os

    return os.cpu_count() or 1
