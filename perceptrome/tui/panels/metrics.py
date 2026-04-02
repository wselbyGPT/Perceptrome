from __future__ import annotations

from textual.widgets import Static

from perceptrome.tui.events import JobMetricUpdatedEvent

from .base import BasePanel


def _spark(values: list[float]) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(bars[min(len(bars)-1, int((v - lo) / span * (len(bars)-1)))] for v in values)


class MetricsPanel(BasePanel):
    PANEL_ID = "metrics"
    TITLE = "Metrics"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._losses: dict[str, list[float]] = {}

    def compose(self):
        yield Static(id="metrics-body")

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobMetricUpdatedEvent):
            self._losses.setdefault(event.job_id, []).append(float(event.latest_value))
            self._losses[event.job_id] = self._losses[event.job_id][-40:]
            self._render()

    def on_mount(self) -> None:
        self._render()

    def _render(self) -> None:
        body = self.query_one("#metrics-body", Static)
        selected = self.selected_job_context().get("selected_job_id")
        vals = list(self._losses.get(selected, [])) if selected else []
        if not vals:
            body.update("No metrics yet.")
            return
        rolling = sum(vals[-5:]) / min(5, len(vals))
        body.update(
            "\n".join(
                [
                    f"Job: {selected}",
                    f"Current: {vals[-1]:.6f}",
                    f"Min/Max: {min(vals):.6f} / {max(vals):.6f}",
                    f"Rolling(5): {rolling:.6f}",
                    f"Trend: {_spark(vals)}",
                ]
            )
        )
