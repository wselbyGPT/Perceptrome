from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class DataPanel(BasePanel):
    PANEL_ID = "data"
    TITLE = "Data"

    def compose(self):
        yield Static("Dataset cache readiness, fetch plans, and accession coverage.")
