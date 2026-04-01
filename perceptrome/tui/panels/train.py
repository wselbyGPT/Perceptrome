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
        if event.key not in {"up", "down", "j", "k"}:
            return
        jobs = self.current_jobs()
        if not jobs:
            return
        session = getattr(self.app.state, "get_session", lambda: None)()
        selected_id = getattr(session, "selected_job_id", None)
        ids = [job.id for job in jobs]
        idx = ids.index(selected_id) if selected_id in ids else 0
        if event.key in {"down", "j"}:
            idx = min(len(ids) - 1, idx + 1)
        else:
            idx = max(0, idx - 1)
        self.app.state.set_selected_job(ids[idx])
        self.schedule_throttled_render("train", self._render)
        event.stop()

    def _render(self) -> None:
        body = self.query_one("#train-body", Static)
        rows = self.current_jobs()
        if not rows:
            body.update("No jobs yet. Use launcher: Job: Start.")
            return

        context = self.selected_job_context()
        selected_job_id = context["selected_job_id"]
        active_job_id = context["active_job_id"]
        if selected_job_id is None:
            selected_job_id = rows[0].id
            self.app.state.set_selected_job(selected_job_id)

        cards: list[str] = ["Training Jobs (↑/↓ or j/k to select)"]
        for row in rows[:8]:
            marker = "▶" if row.id == selected_job_id else " "
            active = " • ACTIVE" if row.id == active_job_id else ""
            progress = getattr(row, "progress", None)
            percent = getattr(progress, "percent", None)
            pct = f"{percent:0.1f}%" if isinstance(percent, (int, float)) else "-"
            step = getattr(progress, "step", None)
            total_steps = getattr(progress, "total_steps", None)
            stage = getattr(row, "current_stage", "") or "-"
            metrics = getattr(row, "metrics", None)
            latest_loss = getattr(metrics, "latest_loss", None)
            rolling_loss = getattr(metrics, "rolling_loss", None)
            loss_text = f"latest={latest_loss:.5f} rolling={rolling_loss:.5f}" if latest_loss is not None and rolling_loss is not None else "latest=- rolling=-"
            cards.extend(
                [
                    f"{marker} {row.id} [{row.status.value}]{active}",
                    f"  Stage: {stage}",
                    f"  Progress: {pct} (step {step or '-'} / {total_steps or '-'})",
                    f"  Metrics: {loss_text}",
                    f"  Message: {row.message or '-'}",
                    "",
                ]
            )
        body.update("\n".join(cards))
