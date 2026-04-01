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
        """Panels receive job events from app-level dispatcher."""

    def on_unmount(self) -> None:
        """No panel-local subscriptions to clean up."""

    def handle_tui_event(self, event: object) -> None:
        _ = event
