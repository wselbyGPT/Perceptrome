from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class PipelinePanel(BasePanel):
    PANEL_ID = "pipeline"
    TITLE = "Pipeline"

    def compose(self):
        yield Static("Pipeline stages and progress details.")
