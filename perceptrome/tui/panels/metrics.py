from __future__ import annotations

from textual.widgets import Static

from perceptrome.tui.events import JobMetricUpdatedEvent

from .base import BasePanel


class MetricsPanel(BasePanel):
    PANEL_ID = "metrics"
    TITLE = "Metrics"

    def compose(self):
        yield Static("Training and scoring metrics are rendered here.", id="metrics-body")

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobMetricUpdatedEvent):
            body = self.query_one("#metrics-body", Static)
            step = "-" if event.step is None else str(event.step)
            body.update(
                f"Job: {event.job_id}\n"
                f"Metric: {event.metric_name}\n"
                f"Step: {step}\n"
                f"Latest loss: {event.latest_value:.6f}\n"
                f"Rolling loss: {event.rolling_value:.6f}"
            )
