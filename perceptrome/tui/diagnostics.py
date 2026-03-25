"""Diagnostic helpers for TUI status panels."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class DiagnosticSnapshot:
    python_version: str
    platform: str


def capture_diagnostics() -> DiagnosticSnapshot:
    return DiagnosticSnapshot(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
    )
