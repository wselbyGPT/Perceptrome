from __future__ import annotations

from textual.widgets import Static

from perceptrome.tui.events import JobStageUpdatedEvent

from .base import BasePanel


class PipelinePanel(BasePanel):
    PANEL_ID = "pipeline"
    TITLE = "Pipeline"

    def compose(self):
        yield Static("Pipeline stages and progress details.", id="pipeline-body")

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobStageUpdatedEvent):
            body = self.query_one("#pipeline-body", Static)
            body.update(f"{event.job_id}: [{event.stage}] {event.message}")
