from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class JobsPanel(BasePanel):
    PANEL_ID = "jobs"
    TITLE = "Jobs"

    def compose(self):
        yield Static("Queued and running jobs appear in this panel.", id="jobs-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render_jobs()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.call_after_refresh(self._render_jobs)

    def _render_jobs(self) -> None:
        body = self.query_one("#jobs-body", Static)
        jobs = getattr(self.app, "jobs", None)
        if jobs is None:
            body.update("No job manager available.")
            return
        cards = jobs.list_jobs()
        if not cards:
            body.update("No jobs yet.")
            return
        lines = []
        for card in cards[:12]:
            artifacts = f" artifacts={len(card.artifacts)}" if card.artifacts else ""
            lines.append(f"{card.id} [{card.status.value}] {card.message}{artifacts}")
        body.update("\n".join(lines))
