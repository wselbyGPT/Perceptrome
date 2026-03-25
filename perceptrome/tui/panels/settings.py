from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class SettingsPanel(BasePanel):
    PANEL_ID = "settings"
    TITLE = "Settings"

    def compose(self):
        yield Static("Application settings and toggles.")
