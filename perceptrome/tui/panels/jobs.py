from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Rule, Static

from perceptrome.tui.events import (
    JobCompletedEvent,
    JobFailedEvent,
    JobStartedEvent,
    JobStageUpdatedEvent,
)
from perceptrome.tui.job_manager import JobStatus

from .base import BasePanel


class JobsPanel(BasePanel):
    PANEL_ID = "jobs"
    TITLE = "Jobs"

    def compose(self) -> ComposeResult:
        yield Static("[b]Job Queue[/b]", classes="panel-title")
        with Horizontal(id="jobs-action-row"):
            yield Button("Cancel selected", id="jobs-cancel", variant="error")
            yield Button("Rerun selected", id="jobs-rerun", variant="warning")
            yield Button("Clone to draft", id="jobs-clone", variant="primary")
            yield Button("View logs", id="jobs-logs")
            yield Button("View artifacts", id="jobs-artifacts")
        yield DataTable(id="jobs-table")
        yield Rule()
        yield Static("[b]Selected Job Detail[/b]")
        yield Static("Select a job above.", id="jobs-detail")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#jobs-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Status", "Job ID", "Kind", "Title", "Stage", "Progress", "Message")
        self._render_jobs()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, (JobStartedEvent, JobStageUpdatedEvent,
                              JobCompletedEvent, JobFailedEvent)):
            self.schedule_throttled_render("jobs", self._render_jobs)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "jobs-cancel":
            ok = self.app.cancel_selected_job()
            self._set_detail("Cancel requested" if ok else "No job to cancel")
        elif event.button.id == "jobs-rerun":
            ok = self.app.rerun_selected_job()
            self._set_detail("Rerun submitted" if ok else "No job to rerun")
        elif event.button.id == "jobs-clone":
            ok = self.app.clone_selected_job_to_draft()
            self._set_detail("Cloned to draft" if ok else "No job to clone")
        elif event.button.id == "jobs-logs":
            self.app.action_show_logs()
        elif event.button.id == "jobs-artifacts":
            self.app._set_panel("artifacts")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        jobs = self.current_jobs()
        if 0 <= event.cursor_row < len(jobs):
            job = jobs[event.cursor_row]
            self.app.state.set_selected_job(job.id)
            self._show_job_detail(job)

    def on_key(self, event) -> None:
        if event.key in {"up", "down", "j", "k"}:
            jobs = self.current_jobs()
            if not jobs:
                return
            selected = self.selected_job_context().get("selected_job_id")
            ids = [j.id for j in jobs]
            idx = ids.index(selected) if selected in ids else 0
            idx = min(len(ids) - 1, idx + 1) if event.key in {"down", "j"} else max(0, idx - 1)
            self.app.state.set_selected_job(ids[idx])
            self._render_jobs()
            event.stop()
        elif event.key == "enter":
            self.app._set_panel("train")
            event.stop()
        elif event.key == "c":
            self.app.cancel_selected_job()
        elif event.key == "r":
            self.app.rerun_selected_job()

    def _render_jobs(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        jobs = self.current_jobs()
        ctx = self.selected_job_context()
        sel = ctx.get("selected_job_id")

        for job in jobs[:30]:
            sts_map = {
                JobStatus.BUSY: "[cyan]RUN [/]",
                JobStatus.HEALTHY: "[green]DONE[/]",
                JobStatus.FAILED: "[red]FAIL[/]",
                JobStatus.STALLED: "[yellow]STAL[/]",
                JobStatus.WARNING: "[yellow]WARN[/]",
            }
            sts = sts_map.get(job.status, str(job.status.value))
            pct = getattr(getattr(job, "progress", None), "percent", None)
            pct_str = f"{pct * 100:.0f}%" if isinstance(pct, (int, float)) else "-"
            marker = ">" if job.id == sel else " "
            table.add_row(
                sts,
                f"{marker}{job.id[:22]}",
                job.kind,
                (job.title or "-")[:20],
                getattr(job, "current_stage", "") or "-",
                pct_str,
                (job.message or "-")[:40],
            )

        if not jobs:
            self._set_detail("No jobs yet. Submit one from the Train or Data panel.")

    def _show_job_detail(self, job) -> None:
        prog = getattr(job, "progress", None)
        met = getattr(job, "metrics", None)
        lines = [
            f"Job ID:    {job.id}",
            f"Kind:      {job.kind}",
            f"Title:     {job.title}",
            f"Status:    {job.status.value}",
            f"Stage:     {getattr(job, 'current_stage', '-')}",
            f"Reason:    {getattr(job, 'status_reason', '-')}",
            f"Step:      {getattr(prog, 'step', '-')}/{getattr(prog, 'total_steps', '-')}",
            f"Epoch:     {getattr(prog, 'epoch', '-')}",
            f"Progress:  {getattr(prog, 'percent', '-')}",
            f"Loss:      {getattr(met, 'latest_loss', '-')}",
            f"Rolling:   {getattr(met, 'rolling_loss', '-')}",
            f"Created:   {job.created_at.isoformat() if hasattr(job.created_at, 'isoformat') else job.created_at}",
            f"Updated:   {job.updated_at.isoformat() if hasattr(job.updated_at, 'isoformat') else job.updated_at}",
            f"Artifacts: {len(getattr(job, 'artifacts', []))}",
        ]
        if getattr(job, "last_error", ""):
            lines.append(f"Error:     {job.last_error}")
        if getattr(job, "last_warning", ""):
            lines.append(f"Warning:   {job.last_warning}")
        self._set_detail("\n".join(lines))

    def _set_detail(self, text: str) -> None:
        self.query_one("#jobs-detail", Static).update(text)
