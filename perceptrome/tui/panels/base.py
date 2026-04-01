"""Shared panel base class."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Static


class BasePanel(Vertical):
    PANEL_ID = "base"
    TITLE = "Base"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._render_timers: dict[str, Timer] = {}

    def compose(self):
        yield Static(f"{self.TITLE} panel", classes="panel-title")

    def on_mount(self) -> None:
        """Panels receive job events from app-level dispatcher."""

    def on_unmount(self) -> None:
        """No panel-local subscriptions to clean up."""

    def handle_tui_event(self, event: object) -> None:
        _ = event

    def current_jobs(self) -> list[object]:
        jobs = getattr(self.app, "jobs", None)
        if jobs is None:
            return []
        return list(jobs.list_jobs())

    def selected_job_context(self) -> dict[str, object | None]:
        session = getattr(self.app.state, "get_session", lambda: None)()
        selected_job_id = getattr(session, "selected_job_id", None)
        active_job_id = getattr(session, "active_job_id", None)
        jobs = self.current_jobs()
        selected = next((row for row in jobs if getattr(row, "id", None) == selected_job_id), None)
        active = next((row for row in jobs if getattr(row, "id", None) == active_job_id), None)
        return {
            "selected_job_id": selected_job_id,
            "active_job_id": active_job_id,
            "selected_job": selected,
            "active_job": active,
            "latest_job": jobs[0] if jobs else None,
        }

    def schedule_throttled_render(
        self,
        key: str,
        render_fn: Callable[[], None],
        *,
        delay: float = 0.05,
    ) -> None:
        prior = self._render_timers.pop(key, None)
        if prior is not None:
            prior.stop()

        def _run_and_clear() -> None:
            self._render_timers.pop(key, None)
            render_fn()

        self._render_timers[key] = self.set_timer(delay, _run_and_clear)
