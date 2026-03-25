"""Shared panel base class."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Static


class BasePanel(Vertical):
    PANEL_ID = "base"
    TITLE = "Base"

    def compose(self):
        yield Static(f"{self.TITLE} panel", classes="panel-title")
