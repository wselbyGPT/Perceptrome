from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class TroubleshootPanel(BasePanel):
    PANEL_ID = "troubleshoot"
    TITLE = "Troubleshoot"

    def compose(self):
        yield Static("Failure triage, tracebacks, diagnostics, and restart guidance.")
