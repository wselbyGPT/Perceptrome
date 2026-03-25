from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class OverviewPanel(BasePanel):
    PANEL_ID = "overview"
    TITLE = "Overview"

    def compose(self):
        yield Static("Overview of Perceptrome jobs and system status.")
