from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class MetricsPanel(BasePanel):
    PANEL_ID = "metrics"
    TITLE = "Metrics"

    def compose(self):
        yield Static("Training and scoring metrics are rendered here.")
