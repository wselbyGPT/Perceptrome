from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class TrainPanel(BasePanel):
    PANEL_ID = "train"
    TITLE = "Train"

    def compose(self):
        yield Static(id="train-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.call_after_refresh(self._render)

    def _render(self) -> None:
        body = self.query_one("#train-body", Static)
        jobs = getattr(self.app, "jobs", None)
        if jobs is None:
            body.update("No job manager available.")
            return
        rows = jobs.list_jobs()
        if not rows:
            body.update("No jobs yet. Use launcher: Job: Start.")
            return
        lines = [f"{row.id} [{row.status.value}] {row.message}" for row in rows[:8]]
        body.update("\n".join(lines))
