from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class HistoryPanel(BasePanel):
    PANEL_ID = "history"
    TITLE = "History"

    def compose(self):
        yield Static("Run history, artifacts, and lineage snapshots.")
