from __future__ import annotations

from pathlib import Path

from textual.containers import Horizontal
from textual.widgets import Button, Static

from .base import BasePanel


class TroubleshootPanel(BasePanel):
    PANEL_ID = "troubleshoot"
    TITLE = "Troubleshoot"

    def compose(self):
        with Horizontal():
            yield Button("Open traceback", id="tb-open", variant="warning")
            yield Button("Open logs", id="tb-logs")
            yield Button("Rerun", id="tb-rerun")
            yield Button("Clone to draft", id="tb-clone")
            yield Button("Jump artifacts", id="tb-artifacts")
        yield Static(id="troubleshoot-body")

    def on_mount(self) -> None:
        super().on_mount()
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

    def _render_troubleshoot(self) -> None:
        body = self.query_one("#troubleshoot-body", Static)
        failed = self.app.state.open_last_failed_job()
        if failed is None:
            body.update("No failed jobs detected. Diagnostics will appear here when failures are recorded.")
            return

        summary = failed.failure_summary
        traceback_path = summary.traceback_path if summary else None
        preview = self._traceback_preview(traceback_path)
        lines = [
            f"Job: {failed.id}",
            f"Status: {failed.status}",
            f"Failure stage: {(summary.stage if summary else 'unknown')}",
            f"Error: {(summary.latest_warning_or_error if summary else failed.config.get('last_error', 'n/a'))}",
            f"Traceback path: {traceback_path or 'n/a'}",
            f"Log path: {failed.config.get('log_path', 'n/a')}",
            f"Manifest: {failed.config.get('manifest_path', 'n/a')}",
            "",
            "Traceback preview:",
            preview,
        ]
        body.update("\n".join(lines))

    def _traceback_preview(self, path: str | None) -> str:
        if not path:
            return "No traceback file path persisted."
        candidate = Path(path)
        if not candidate.exists() or not candidate.is_file():
            return "Persisted traceback path is missing on disk."
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return f"Unable to read traceback: {exc}"
        return "\n".join(lines[:14]) if lines else "Traceback file is empty."
