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
            self.schedule_throttled_render("jobs", self._render_jobs)

    def _render_jobs(self) -> None:
        body = self.query_one("#jobs-body", Static)
        cards = self.current_jobs()
        if not cards:
            body.update("No jobs yet.")
            return

        context = self.selected_job_context()
        selected_job_id = context["selected_job_id"]
        active_job_id = context["active_job_id"]
        if selected_job_id is None and active_job_id is None:
            selected_job_id = cards[0].id

        lines = []
        for card in cards[:12]:
            artifacts = f" artifacts={len(card.artifacts)}" if card.artifacts else ""
            marker = "*" if card.id in {selected_job_id, active_job_id} else " "
            lines.append(f"{marker} {card.id} [{card.status.value}] {card.message}{artifacts}")
        body.update("\n".join(lines))
