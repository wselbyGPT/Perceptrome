from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class DiagnosticsPanel(BasePanel):
    PANEL_ID = "diagnostics"
    TITLE = "Diagnostics"

    def compose(self):
        yield Static("Environment diagnostics and health checks.")
