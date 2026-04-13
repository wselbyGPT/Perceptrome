from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Rule, Static

from .base import BasePanel


class TroubleshootPanel(BasePanel):
    PANEL_ID = "troubleshoot"
    TITLE = "Troubleshoot"

    def compose(self) -> ComposeResult:
        yield Static("[b]Troubleshoot Failed Jobs[/b]", classes="panel-title")
        with Horizontal():
            yield Button("Open traceback", id="tb-open", variant="warning")
            yield Button("Open logs", id="tb-logs")
            yield Button("Rerun", id="tb-rerun", variant="success")
            yield Button("Clone to draft", id="tb-clone")
            yield Button("View artifacts", id="tb-artifacts")
        yield Rule()

        yield Static("[b]Failed Jobs[/b]")
        yield DataTable(id="tb-failed-table")
        yield Rule()

        yield Static("[b]Failure Detail[/b]")
        yield Static("No failed jobs detected.", id="tb-detail")
        yield Rule()

        yield Static("[b]Traceback Preview[/b]")
        yield Static("", id="tb-traceback")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#tb-failed-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Job ID", "Kind", "Stage", "Error")
        self._render_troubleshoot()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("troubleshoot", self._render_troubleshoot)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tb-open":
            self.app.action_show_traceback()
        elif event.button.id == "tb-logs":
            self.app.action_show_logs()
        elif event.button.id == "tb-rerun":
            self.app.rerun_selected_job()
        elif event.button.id == "tb-clone":
            self.app.clone_selected_job_to_draft()
        elif event.button.id == "tb-artifacts":
            self.app._set_panel("artifacts")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from perceptrome.tui.job_manager import JobStatus
        failed = [j for j in self.current_jobs() if j.status == JobStatus.FAILED]
        if 0 <= event.cursor_row < len(failed):
            job = failed[event.cursor_row]
            self.app.state.set_selected_job(job.id)
            self._show_failure_detail(job)

    def _render_troubleshoot(self) -> None:
        from perceptrome.tui.job_manager import JobStatus
        jobs = self.current_jobs()
        failed = [j for j in jobs if j.status == JobStatus.FAILED]

        table = self.query_one("#tb-failed-table", DataTable)
        table.clear()
        for job in failed[:15]:
            error = getattr(job, "last_error", "") or getattr(job, "message", "") or "-"
            table.add_row(
                job.id[:20],
                job.kind,
                getattr(job, "current_stage", "") or "-",
                error[:50],
            )

        if not failed:
            self.query_one("#tb-detail", Static).update(
                "No failed jobs detected. Diagnostics will appear here when failures are recorded."
            )
            self.query_one("#tb-traceback", Static).update("")
            return

        # Show first failed job by default
        failed_record = self.app.state.open_last_failed_job()
        if failed_record:
            self._show_failure_detail_from_record(failed_record)

    def _show_failure_detail(self, job) -> None:
        lines = [
            f"Job ID:  {job.id}",
            f"Kind:    {job.kind}",
            f"Stage:   {getattr(job, 'current_stage', '-')}",
            f"Reason:  {getattr(job, 'status_reason', '-')}",
            f"Error:   {getattr(job, 'last_error', '-')}",
            f"Warning: {getattr(job, 'last_warning', '-')}",
        ]
        meta = getattr(job, "status_metadata", {})
        if meta.get("traceback_path"):
            lines.append(f"Traceback: {meta['traceback_path']}")
            preview = self._traceback_preview(meta["traceback_path"])
            self.query_one("#tb-traceback", Static).update(preview)
        else:
            self.query_one("#tb-traceback", Static).update("[dim]No traceback path available[/]")
        self.query_one("#tb-detail", Static).update("\n".join(lines))

    def _show_failure_detail_from_record(self, record) -> None:
        summary = record.failure_summary
        traceback_path = summary.traceback_path if summary else None
        lines = [
            f"Job ID:  {record.id}",
            f"Kind:    {record.kind}",
            f"Status:  {record.status}",
            f"Stage:   {(summary.stage if summary else 'unknown')}",
            f"Error:   {(summary.latest_warning_or_error if summary else record.config.get('last_error', '-'))}",
            f"TB path: {traceback_path or '-'}",
            f"Log:     {record.config.get('log_path', '-')}",
            f"Manifest:{record.config.get('manifest_path', '-')}",
        ]
        self.query_one("#tb-detail", Static).update("\n".join(lines))
        self.query_one("#tb-traceback", Static).update(self._traceback_preview(traceback_path))

    def _traceback_preview(self, path: str | None) -> str:
        if not path:
            return "[dim]No traceback file path.[/]"
        candidate = Path(path)
        if not candidate.exists() or not candidate.is_file():
            return f"[dim]{path} (missing on disk)[/]"
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return f"[red]Unable to read: {exc}[/]"
        return "\n".join(lines[:20]) if lines else "[dim]Traceback file is empty.[/]"
