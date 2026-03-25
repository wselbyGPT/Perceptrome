from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class JobsPanel(BasePanel):
    PANEL_ID = "jobs"
    TITLE = "Jobs"

    def compose(self):
        yield Static("Queued and running jobs appear in this panel.")
