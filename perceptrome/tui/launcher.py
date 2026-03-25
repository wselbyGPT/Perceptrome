"""Global launcher registry with state-aware command ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .job_manager import JobStatus


@dataclass(frozen=True, slots=True)
class LauncherContext:
    has_running_job: bool = False
    has_failed_job: bool = False
    has_recent_job: bool = False
    active_panel: str = "overview"


@dataclass(frozen=True, slots=True)
class LauncherCommand:
    command_id: str
    label: str
    category: str
    panel_id: str | None = None
    action: str | None = None
    keywords: tuple[str, ...] = field(default_factory=tuple)
    rank_hint: Callable[[LauncherContext], int] | None = None

    def rank(self, context: LauncherContext) -> int:
        score = 10
        if self.panel_id == context.active_panel:
            score -= 3
        if self.rank_hint is not None:
            score += int(self.rank_hint(context))
        return score


def _rank_for_running(context: LauncherContext) -> int:
    return 40 if context.has_running_job else -5


def _rank_for_idle(context: LauncherContext) -> int:
    return 24 if (not context.has_running_job and not context.has_failed_job) else -8


def _rank_for_failed(context: LauncherContext) -> int:
    return 36 if context.has_failed_job else -6


DEFAULT_COMMANDS: tuple[LauncherCommand, ...] = (
    LauncherCommand("panel.overview", "Jump: Overview", "panel", panel_id="overview"),
    LauncherCommand("panel.config", "Jump: Config", "panel", panel_id="config"),
    LauncherCommand("panel.data", "Jump: Data", "panel", panel_id="data"),
    LauncherCommand("panel.train", "Jump: Train", "panel", panel_id="train"),
    LauncherCommand("panel.generate", "Jump: Generate", "panel", panel_id="generate"),
    LauncherCommand("panel.history", "Jump: History", "panel", panel_id="history"),
    LauncherCommand("panel.troubleshoot", "Jump: Troubleshoot", "panel", panel_id="troubleshoot"),
    LauncherCommand("job.start", "Job: Start", "job", action="start_job", rank_hint=_rank_for_idle),
    LauncherCommand("job.stop", "Job: Stop Active", "job", action="stop_job", rank_hint=_rank_for_running),
    LauncherCommand("job.rerun", "Job: Rerun Last", "job", action="rerun_job", rank_hint=lambda ctx: 22 if ctx.has_recent_job else -8),
    LauncherCommand("inspect.active", "Inspect: Active Job", "inspect", action="inspect_active", rank_hint=_rank_for_running),
    LauncherCommand("inspect.open", "Inspect: Open Artifact", "inspect", action="open_artifact", rank_hint=lambda ctx: 20 if ctx.has_recent_job else -8),
    LauncherCommand("inspect.logs", "Inspect: Show Logs", "inspect", action="show_logs", rank_hint=_rank_for_running),
    LauncherCommand("view.logs", "View: Logs Drawer", "view", action="toggle_logs"),
    LauncherCommand("view.diagnostics", "View: Diagnostics Drawer", "view", action="toggle_diagnostics", rank_hint=_rank_for_failed),
    LauncherCommand("view.resources", "View: Resources Drawer", "view", action="toggle_resources"),
    LauncherCommand("view.traceback", "View: Traceback Drawer", "view", action="toggle_traceback", rank_hint=_rank_for_failed),
    LauncherCommand("view.artifacts", "View: Artifact Details", "view", action="toggle_artifact_details"),
    LauncherCommand("view.reset", "View: Reset Layout", "view", action="reset_layout"),
    LauncherCommand("failure.troubleshoot", "Troubleshoot: Open Failed Job", "failure", action="open_failed", rank_hint=_rank_for_failed),
    LauncherCommand("failure.traceback", "Troubleshoot: Open Traceback", "failure", action="toggle_traceback", rank_hint=_rank_for_failed),
    LauncherCommand("failure.reopen", "Troubleshoot: Reopen Last Failed", "failure", action="reopen_failed", rank_hint=_rank_for_failed),
)


def derive_context(*, active_panel: str, jobs: Iterable[object]) -> LauncherContext:
    rows = list(jobs)
    has_running = any(getattr(row, "status", None) == JobStatus.BUSY for row in rows)
    has_failed = any(getattr(row, "status", None) == JobStatus.FAILED for row in rows)
    return LauncherContext(
        has_running_job=has_running,
        has_failed_job=has_failed,
        has_recent_job=bool(rows),
        active_panel=active_panel,
    )


def rank_commands(context: LauncherContext, commands: Iterable[LauncherCommand] = DEFAULT_COMMANDS) -> list[LauncherCommand]:
    return sorted(commands, key=lambda command: (-command.rank(context), command.label.lower()))
