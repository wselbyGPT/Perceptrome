from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Rule, Static

from .base import BasePanel


class EventsPanel(BasePanel):
    PANEL_ID = "events"
    TITLE = "Events"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit = 40

    def compose(self) -> ComposeResult:
        yield Static("[b]Event Log[/b]", classes="panel-title")
        with Horizontal():
            yield Button("Refresh", id="events-refresh", variant="primary")
            yield Button("Show more", id="events-more")
        yield DataTable(id="events-table")
        yield Rule()
        yield Static("", id="events-count")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#events-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Time", "Kind", "Message")
        self._limit = 40
        self._render()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "kind"):
            self.schedule_throttled_render("events", self._render)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "events-refresh":
            self._render()
        elif event.button.id == "events-more":
            self._limit = min(200, self._limit + 40)
            self._render()

    def _render(self) -> None:
        rows = self.app.state.read_events_tail(limit=self._limit)
        table = self.query_one("#events-table", DataTable)
        table.clear()
        for row in reversed(rows):
            time = str(row.get("created_at", ""))
            if "T" in time:
                time = time.split("T")[1][:8]
            table.add_row(
                time,
                str(row.get("kind", "?")),
                str(row.get("message", ""))[:60],
            )
        self.query_one("#events-count", Static).update(f"Showing {len(rows)} events (limit {self._limit})")
