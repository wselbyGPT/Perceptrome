from __future__ import annotations

from pathlib import Path

from textual.widgets import Static

from .base import BasePanel


class TroubleshootPanel(BasePanel):
    PANEL_ID = "troubleshoot"
    TITLE = "Troubleshoot"

    def compose(self):
        yield Static(id="troubleshoot-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render_troubleshoot()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("troubleshoot", self._render_troubleshoot)

    def _render_troubleshoot(self) -> None:
        body = self.query_one("#troubleshoot-body", Static)
        failed = self.app.state.open_last_failed_job()
        if failed is None:
            body.update("No failed jobs detected. Diagnostics will appear here when failures are recorded.")
            return

        summary = failed.failure_summary
        if summary is None:
            body.update(f"Failed job {failed.id} has no persisted failure summary yet.")
            return

        traceback_path = summary.traceback_path or self._guess_traceback_path(failed)
        preview = self._traceback_preview(traceback_path)
        body.update(
            "\n".join(
                [
                    f"Job: {failed.id}",
                    f"Status: {failed.status}",
                    f"Failure stage: {summary.stage or 'unknown'}",
                    f"Error: {summary.latest_warning_or_error or 'n/a'}",
                    f"Suggested action: {summary.suggested_next_action}",
                    f"Traceback path: {traceback_path or 'not available'}",
                    "",
                    "Traceback preview:",
                    preview,
                ]
            )
        )

    def _guess_traceback_path(self, failed_job: object) -> str | None:
        for artifact in getattr(failed_job, "artifacts", []) or []:
            if not isinstance(artifact, dict):
                continue
            path = str(artifact.get("path") or "")
            if path.endswith(".traceback.txt") or "traceback" in path.lower():
                return path
        return None

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
        if not lines:
            return "Traceback file is empty."
        return "\n".join(lines[:12])
