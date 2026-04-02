from __future__ import annotations

from textual.events import Key
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
            self.schedule_throttled_render("train", self._render)

    def on_key(self, event: Key) -> None:
        if event.key == "r":
            self.app.rerun_selected_job()
        elif event.key == "c":
            self.app.cancel_selected_job()
        elif event.key == "l":
            self.app.action_show_logs()
        elif event.key == "a":
            self.app._set_panel("artifacts")
        elif event.key == "p":
            self.app._set_panel("pipeline")

    def _render(self) -> None:
        body = self.query_one("#train-body", Static)
        rows = self.current_jobs()
        if not rows:
            body.update("No jobs yet. Use launcher: Job: Start.")
            return

        context = self.selected_job_context()
        selected_job_id = context["selected_job_id"] or rows[0].id
        self.app.state.set_selected_job(selected_job_id)

        cards: list[str] = ["Training Jobs (r rerun, c cancel, l logs, a artifacts, p pipeline)"]
        for row in rows[:8]:
            marker = "▶" if row.id == selected_job_id else " "
            progress = getattr(row, "progress", None)
            percent = getattr(progress, "percent", None)
            pct = f"{percent * 100:0.1f}%" if isinstance(percent, (int, float)) and percent <= 1 else (f"{percent:0.1f}%" if isinstance(percent, (int, float)) else "-")
            cards.append(f"{marker} {row.id} [{row.status.value}] stage={row.current_stage or '-'}")
            cards.append(f"  Progress: {pct}; step {getattr(progress, 'step', '-')}/{getattr(progress, 'total_steps', '-')}")
            cards.append(f"  Reason: {row.status_reason or '-'}")
            cards.append(f"  {row.message or '-'}")
        body.update("\n".join(cards))
