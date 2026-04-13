from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import DataTable, Rule, Static

from perceptrome.tui.events import JobStageUpdatedEvent, JobStartedEvent

from .base import BasePanel

_STAGE_ICONS = {
    "start": "[cyan]>[/]",
    "encode": "[blue]E[/]",
    "train": "[green]T[/]",
    "stream_step": "[green]S[/]",
    "pretrain": "[green]P[/]",
    "generate": "[magenta]G[/]",
    "validate": "[yellow]V[/]",
    "design_loop": "[magenta]D[/]",
    "manifest": "[dim]M[/]",
    "error": "[red]![/]",
    "warn": "[yellow]W[/]",
    "warning": "[yellow]W[/]",
    "log": "[dim]L[/]",
    "progress": "[cyan].[/]",
}


class PipelinePanel(BasePanel):
    PANEL_ID = "pipeline"
    TITLE = "Pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timelines: dict[str, list[dict[str, str]]] = {}

    def compose(self) -> ComposeResult:
        yield Static("[b]Pipeline Stage Timeline[/b]", classes="panel-title")
        yield Static("", id="pipeline-active")
        yield Rule()
        yield DataTable(id="pipeline-table")
        yield Rule()
        yield Static("[b]Stage Sequence[/b]")
        yield Static("", id="pipeline-sequence")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#pipeline-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Time", "Stage", "Message")
        self._render()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, (JobStageUpdatedEvent, JobStartedEvent)):
            if isinstance(event, JobStageUpdatedEvent):
                self._timelines.setdefault(event.job_id, []).append({
                    "time": event.created_at.strftime("%H:%M:%S"),
                    "stage": event.stage,
                    "message": event.message,
                })
                self._timelines[event.job_id] = self._timelines[event.job_id][-50:]
            self.schedule_throttled_render("pipeline", self._render)

    def _render(self) -> None:
        ctx = self.selected_job_context()
        selected = ctx.get("selected_job_id")
        active_job = ctx.get("active_job")

        # Active job header
        header = self.query_one("#pipeline-active", Static)
        if active_job:
            stage = getattr(active_job, "current_stage", "-")
            header.update(f"Active: {active_job.id}  |  Stage: [cyan]{stage}[/]  |  {active_job.title}")
        else:
            header.update("[dim]No active job[/]")

        # Timeline table
        events = list(self._timelines.get(selected, [])) if selected else []
        table = self.query_one("#pipeline-table", DataTable)
        table.clear()
        for entry in events[-30:]:
            icon = _STAGE_ICONS.get(entry["stage"], " ")
            table.add_row(entry["time"], f"{icon} {entry['stage']}", (entry["message"])[:60])

        # Stage sequence visualization
        seq = self.query_one("#pipeline-sequence", Static)
        if events:
            stages_seen: list[str] = []
            for entry in events:
                s = entry["stage"]
                if not stages_seen or stages_seen[-1] != s:
                    stages_seen.append(s)
            icons = [_STAGE_ICONS.get(s, f"[dim]{s}[/]") for s in stages_seen[-20:]]
            seq.update("  " + " -> ".join(icons) + f"  ({len(events)} events)")
        else:
            seq.update("[dim]No stage events for selected job[/]")
