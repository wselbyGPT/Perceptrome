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
    enabled_when: Callable[[LauncherContext], tuple[bool, str | None] | bool] | None = None
    requires_confirm: bool = False
    parameters: tuple["LauncherParameter", ...] = field(default_factory=tuple)
    rank_hint: Callable[[LauncherContext], int] | None = None

    def rank(self, context: LauncherContext) -> int:
        score = 10
        if self.panel_id == context.active_panel:
            score -= 3
        if self.rank_hint is not None:
            score += int(self.rank_hint(context))
        return score


@dataclass(frozen=True, slots=True)
class LauncherParameter:
    name: str
    label: str
    help_text: str = ""
    required: bool = False


@dataclass(frozen=True, slots=True)
class RankedCommand:
    command: LauncherCommand
    score: int
    disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None


def _enabled_running(context: LauncherContext) -> tuple[bool, str | None]:
    return (True, None) if context.has_running_job else (False, "No active job is running")


def _enabled_idle(context: LauncherContext) -> tuple[bool, str | None]:
    return (True, None) if not context.has_running_job else (False, "Stop the active job before starting another")


def _enabled_recent(context: LauncherContext) -> tuple[bool, str | None]:
    return (True, None) if context.has_recent_job else (False, "No recent jobs available")


def _enabled_failed(context: LauncherContext) -> tuple[bool, str | None]:
    return (True, None) if context.has_failed_job else (False, "No failed jobs available")


def _rank_for_running(context: LauncherContext) -> int:
    return 40 if context.has_running_job else -5


def _rank_for_idle(context: LauncherContext) -> int:
    return 24 if (not context.has_running_job and not context.has_failed_job) else -8


def _rank_for_failed(context: LauncherContext) -> int:
    return 36 if context.has_failed_job else -6


DEFAULT_COMMANDS: tuple[LauncherCommand, ...] = (
    LauncherCommand("panel.overview", "Jump: Overview", "panel", panel_id="overview", keywords=("home", "main")),
    LauncherCommand("panel.config", "Jump: Config", "panel", panel_id="config", keywords=("settings", "configure")),
    LauncherCommand("panel.data", "Jump: Data", "panel", panel_id="data", keywords=("dataset", "inputs")),
    LauncherCommand("panel.train", "Jump: Train", "panel", panel_id="train", keywords=("fit", "training")),
    LauncherCommand("panel.generate", "Jump: Generate", "panel", panel_id="generate", keywords=("inference", "predict")),
    LauncherCommand("panel.history", "Jump: History", "panel", panel_id="history", keywords=("runs", "timeline")),
    LauncherCommand("panel.jobs", "Jump: Jobs", "panel", panel_id="jobs", keywords=("queue", "active")),
    LauncherCommand("panel.metrics", "Jump: Metrics", "panel", panel_id="metrics", keywords=("loss", "signals")),
    LauncherCommand("panel.pipeline", "Jump: Pipeline", "panel", panel_id="pipeline", keywords=("stages", "flow")),
    LauncherCommand("panel.troubleshoot", "Jump: Troubleshoot", "panel", panel_id="troubleshoot", keywords=("debug", "failure")),
    LauncherCommand("job.start", "Job: Start", "job", action="start_job", rank_hint=_rank_for_idle, enabled_when=_enabled_idle, requires_confirm=True),
    LauncherCommand("job.stop", "Job: Stop Active", "job", action="stop_job", rank_hint=_rank_for_running, enabled_when=_enabled_running, requires_confirm=True),
    LauncherCommand("job.rerun", "Job: Rerun Last", "job", action="rerun_job", rank_hint=lambda ctx: 22 if ctx.has_recent_job else -8, enabled_when=_enabled_recent),
    LauncherCommand("inspect.active", "Inspect: Active Job", "inspect", action="inspect_active", rank_hint=_rank_for_running, enabled_when=_enabled_running),
    LauncherCommand("inspect.open", "Inspect: Open Artifact", "inspect", action="open_artifact", rank_hint=lambda ctx: 20 if ctx.has_recent_job else -8, enabled_when=_enabled_recent),
    LauncherCommand("inspect.logs", "Inspect: Show Logs", "inspect", action="show_logs", rank_hint=_rank_for_running, enabled_when=_enabled_running),
    LauncherCommand("view.logs", "View: Logs Drawer", "view", action="toggle_logs"),
    LauncherCommand("view.diagnostics", "View: Diagnostics Drawer", "view", action="toggle_diagnostics", rank_hint=_rank_for_failed, parameters=(LauncherParameter(name="scope", label="Diagnostic Scope", help_text="Current panel or full environment"),)),
    LauncherCommand("view.resources", "View: Resources Drawer", "view", action="toggle_resources"),
    LauncherCommand("view.traceback", "View: Traceback Drawer", "view", action="toggle_traceback", rank_hint=_rank_for_failed, enabled_when=_enabled_failed),
    LauncherCommand("view.artifacts", "View: Artifact Details", "view", action="toggle_artifact_details"),
    LauncherCommand("view.reset", "View: Reset Layout", "view", action="reset_layout"),
    LauncherCommand("failure.troubleshoot", "Troubleshoot: Open Failed Job", "failure", action="open_failed", rank_hint=_rank_for_failed, enabled_when=_enabled_failed),
    LauncherCommand("failure.traceback", "Troubleshoot: Open Traceback", "failure", action="toggle_traceback", rank_hint=_rank_for_failed, enabled_when=_enabled_failed),
    LauncherCommand("failure.reopen", "Troubleshoot: Reopen Last Failed", "failure", action="reopen_failed", rank_hint=_rank_for_failed, enabled_when=_enabled_failed),
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


def rank_and_filter_commands(
    context: LauncherContext,
    *,
    query: str = "",
    commands: Iterable[LauncherCommand] = DEFAULT_COMMANDS,
    launcher_history: Iterable[dict[str, object]] = (),
) -> list[RankedCommand]:
    query_tokens = tuple(token for token in query.strip().lower().split() if token)
    history_boost = _history_scores(launcher_history)
    ranked: list[RankedCommand] = []
    for command in commands:
        disabled_reason = _disabled_reason(command, context)
        score = command.rank(context)
        score += history_boost.get(command.command_id, 0)
        score += _query_score(command, query_tokens)
        if query_tokens and score <= 0:
            continue
        ranked.append(RankedCommand(command=command, score=score, disabled_reason=disabled_reason))
    return sorted(ranked, key=lambda row: (-row.score, row.command.label.lower()))


def _disabled_reason(command: LauncherCommand, context: LauncherContext) -> str | None:
    if command.enabled_when is None:
        return None
    result = command.enabled_when(context)
    if isinstance(result, tuple):
        enabled, reason = result
        return None if enabled else reason or "Unavailable"
    return None if result else "Unavailable"


def _history_scores(history: Iterable[dict[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, row in enumerate(reversed(list(history))):
        command_id = str(row.get("command") or "")
        if not command_id:
            continue
        out[command_id] = out.get(command_id, 0) + max(1, 8 - index)
    return out


def _query_score(command: LauncherCommand, query_tokens: tuple[str, ...]) -> int:
    if not query_tokens:
        return 0
    haystack_label = command.label.lower()
    haystack_category = command.category.lower()
    haystack_keywords = tuple(token.lower() for token in command.keywords)
    haystack_id = command.command_id.lower()
    score = 0
    for token in query_tokens:
        if token in haystack_label:
            score += 20
        if token in haystack_id:
            score += 14
        if token in haystack_category:
            score += 10
        if any(token in keyword for keyword in haystack_keywords):
            score += 12
    return score
