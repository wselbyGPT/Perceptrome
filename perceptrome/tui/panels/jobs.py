from __future__ import annotations

from textual.events import Key
from textual.widgets import Static

from .base import BasePanel


class JobsPanel(BasePanel):
    PANEL_ID = "jobs"
    TITLE = "Jobs"

    def compose(self):
        yield Static(id="jobs-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render_jobs()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("jobs", self._render_jobs)

    def on_key(self, event: Key) -> None:
        jobs = self.current_jobs()
        if not jobs:
            return
        if event.key in {"up", "down", "j", "k"}:
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

    def _render_jobs(self) -> None:
        body = self.query_one("#jobs-body", Static)
        cards = self.current_jobs()
        if not cards:
            body.update("No jobs yet.")
            return
        context = self.selected_job_context()
        selected_job_id = context["selected_job_id"]
        lines = ["Jobs (↑/↓ select, Enter open)"]
        for card in cards[:20]:
            marker = ">" if card.id == selected_job_id else " "
            lines.append(f"{marker} {card.id} [{card.status.value}] stage={card.current_stage or '-'} reason={card.status_reason or '-'}")
            lines.append(f"  {card.message}")
        body.update("\n".join(lines))
