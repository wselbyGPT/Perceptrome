from __future__ import annotations

from textual.widgets import Static

from perceptrome.tui.events import JobStageUpdatedEvent

from .base import BasePanel


class PipelinePanel(BasePanel):
    PANEL_ID = "pipeline"
    TITLE = "Pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeline: dict[str, list[str]] = {}

    def compose(self):
        yield Static(id="pipeline-body")

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobStageUpdatedEvent):
            self._timeline.setdefault(event.job_id, []).append(
                f"{event.created_at.isoformat()} [{event.stage}] {event.message}"
            )
            self._timeline[event.job_id] = self._timeline[event.job_id][-30:]
            self._render()

    def on_mount(self) -> None:
        self._render()

    def _render(self) -> None:
        body = self.query_one("#pipeline-body", Static)
        selected = self.selected_job_context().get("selected_job_id")
        lines = [f"Pipeline timeline for: {selected or '(none)'}"]
        if selected and self._timeline.get(selected):
            lines.extend(self._timeline[selected][-20:])
        else:
            lines.append("No stage events yet.")
        body.update("\n".join(lines))
