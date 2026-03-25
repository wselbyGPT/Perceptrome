from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class GeneratePanel(BasePanel):
    PANEL_ID = "generate"
    TITLE = "Generate"

    def compose(self):
        yield Static("Sequence generation controls, constraints, and export actions.")
