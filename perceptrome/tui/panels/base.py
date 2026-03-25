"""Shared panel base class."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Static


class BasePanel(Vertical):
    PANEL_ID = "base"
    TITLE = "Base"

    def compose(self):
        yield Static(f"{self.TITLE} panel", classes="panel-title")

    def on_mount(self) -> None:
        jobs = getattr(self.app, "jobs", None)
        self._job_subscription_token = None
        if jobs is not None:
            self._job_subscription_token = jobs.subscribe(self.handle_tui_event)

    def on_unmount(self) -> None:
        jobs = getattr(self.app, "jobs", None)
        token = getattr(self, "_job_subscription_token", None)
        if jobs is not None and token is not None:
            jobs.unsubscribe(token)

    def handle_tui_event(self, event: object) -> None:
        _ = event
