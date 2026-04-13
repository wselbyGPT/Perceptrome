from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label, ProgressBar, Rule, Sparkline, Static

from perceptrome.tui.events import (
    JobCompletedEvent,
    JobFailedEvent,
    JobMetricUpdatedEvent,
    JobStageUpdatedEvent,
    JobStartedEvent,
)
from perceptrome.tui.job_manager import JobStatus

from .base import BasePanel


def _badge(status: object) -> str:
    if status == JobStatus.HEALTHY:
        return "[green]DONE[/]"
    if status == JobStatus.BUSY:
        return "[cyan]RUN [/]"
    if status == JobStatus.WARNING:
        return "[yellow]WARN[/]"
    if status == JobStatus.STALLED:
        return "[dark_orange]STAL[/]"
    if status == JobStatus.FAILED:
        return "[red]FAIL[/]"
    return f"[dim]{status}[/]"


class OverviewPanel(BasePanel):
    PANEL_ID = "overview"
    TITLE = "Overview"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_data: dict[str, list[float]] = {}

    def compose(self) -> ComposeResult:
        yield Static("[b]Dashboard[/b]", classes="panel-title")
        with Horizontal(id="overview-top-row"):
            with Vertical(id="overview-stat-active", classes="overview-stat"):
                yield Label("Active", classes="stat-label")
                yield Static("0", id="stat-active-val", classes="stat-value")
            with Vertical(id="overview-stat-done", classes="overview-stat"):
                yield Label("Completed", classes="stat-label")
                yield Static("0", id="stat-done-val", classes="stat-value")
            with Vertical(id="overview-stat-failed", classes="overview-stat"):
                yield Label("Failed", classes="stat-label")
                yield Static("0", id="stat-failed-val", classes="stat-value")
            with Vertical(id="overview-stat-total", classes="overview-stat"):
                yield Label("Total", classes="stat-label")
                yield Static("0", id="stat-total-val", classes="stat-value")
        yield Rule()
        yield Static("[b]Active Job[/b]", id="active-job-label")
        yield ProgressBar(total=100, show_eta=False, id="overview-progress")
        yield Static("", id="overview-active-info")
        yield Rule()
        yield Static("[b]Loss Trend[/b]")
        yield Sparkline([], id="overview-sparkline")
        yield Rule()
        yield Static("[b]Recent Jobs[/b]")
        yield DataTable(id="overview-jobs-table")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#overview-jobs-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Status", "Job ID", "Kind", "Stage", "Message")
        self._refresh_all()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobMetricUpdatedEvent):
            self._loss_data.setdefault(event.job_id, []).append(float(event.latest_value))
            self._loss_data[event.job_id] = self._loss_data[event.job_id][-60:]
        if isinstance(event, (JobStartedEvent, JobStageUpdatedEvent, JobMetricUpdatedEvent,
                              JobCompletedEvent, JobFailedEvent)):
            self.schedule_throttled_render("overview", self._refresh_all)

    def _refresh_all(self) -> None:
        jobs = self.current_jobs()
        active = [j for j in jobs if getattr(j, "status", None) == JobStatus.BUSY]
        done = [j for j in jobs if getattr(j, "status", None) == JobStatus.HEALTHY]
        failed = [j for j in jobs if getattr(j, "status", None) == JobStatus.FAILED]

        self.query_one("#stat-active-val", Static).update(str(len(active)))
        self.query_one("#stat-done-val", Static).update(str(len(done)))
        self.query_one("#stat-failed-val", Static).update(str(len(failed)))
        self.query_one("#stat-total-val", Static).update(str(len(jobs)))

        # Active job progress
        progress_bar = self.query_one("#overview-progress", ProgressBar)
        info = self.query_one("#overview-active-info", Static)
        if active:
            job = active[0]
            pct = getattr(getattr(job, "progress", None), "percent", None)
            if isinstance(pct, (int, float)):
                progress_bar.update(progress=min(100, int(pct * 100)))
            else:
                progress_bar.update(progress=0)
            step = getattr(getattr(job, "progress", None), "step", None) or "-"
            total = getattr(getattr(job, "progress", None), "total_steps", None) or "-"
            stage = getattr(job, "current_stage", "") or "-"
            info.update(f"  {job.title}  |  stage: {stage}  |  step {step}/{total}")
        else:
            progress_bar.update(progress=0)
            info.update("  No active jobs")

        # Sparkline - show the most recent active job's loss, or most recent job with data
        spark = self.query_one("#overview-sparkline", Sparkline)
        loss_data: list[float] = []
        for job in active:
            if job.id in self._loss_data:
                loss_data = self._loss_data[job.id]
                break
        if not loss_data:
            for jid, vals in self._loss_data.items():
                if vals:
                    loss_data = vals
                    break
        spark.data = loss_data if loss_data else [0.0]

        # Job table
        table = self.query_one("#overview-jobs-table", DataTable)
        table.clear()
        for job in jobs[:12]:
            table.add_row(
                _badge(job.status),
                job.id[:24],
                job.kind,
                getattr(job, "current_stage", "") or "-",
                (job.message or "-")[:50],
            )
