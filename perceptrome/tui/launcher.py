"""Global launcher helpers for command-palette style navigation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LauncherCommand:
    label: str
    view_id: str


DEFAULT_COMMANDS: tuple[LauncherCommand, ...] = (
    LauncherCommand("Go to Overview", "overview"),
    LauncherCommand("Go to Jobs", "jobs"),
    LauncherCommand("Go to Metrics", "metrics"),
    LauncherCommand("Go to Pipeline", "pipeline"),
    LauncherCommand("Go to History", "history"),
    LauncherCommand("Go to Diagnostics", "diagnostics"),
    LauncherCommand("Go to Settings", "settings"),
)
