from __future__ import annotations

from textual.widgets import Static

from .base import BasePanel


class EventsPanel(BasePanel):
    PANEL_ID = "events"
    TITLE = "Events"

    def compose(self):
        yield Static(id="events-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "kind"):
            self.schedule_throttled_render("events", self._render)

    def _render(self) -> None:
        body = self.query_one("#events-body", Static)
        rows = self.app.state.read_events_tail(limit=20)
        if not rows:
            body.update("No persisted TUI events yet.")
            return
        lines = ["Recent TUI Events"]
        for row in rows:
            lines.append(f"- {row.get('created_at', '?')} [{row.get('kind', '?')}] {row.get('message', '')}")
        body.update("\n".join(lines))
