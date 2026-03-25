from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class OverviewPanel(BasePanel):
    PANEL_ID = "overview"
    TITLE = "Overview"

    def compose(self):
        yield Static("Perceptrome launch summary, active run health, and quick links.")
