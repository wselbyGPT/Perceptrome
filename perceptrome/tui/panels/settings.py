from __future__ import annotations

import shutil
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Rule, Select, Static, Switch

from .base import BasePanel

_THEMES = [("Dark (default)", "dark"), ("Light", "light")]


class SettingsPanel(BasePanel):
    PANEL_ID = "settings"
    TITLE = "Settings"

    def compose(self) -> ComposeResult:
        yield Static("[b]Application Settings[/b]", classes="panel-title")

        # Paths
        yield Static("[b]Paths[/b]")
        with Horizontal(id="settings-paths-row"):
            with Vertical():
                yield Label("Default config path")
                yield Input(value="config/stream_config.yaml", id="settings-config-path")
            with Vertical():
                yield Label("State directory")
                yield Static("", id="settings-state-dir")

        yield Rule()

        # Display
        yield Static("[b]Display[/b]")
        with Horizontal(id="settings-display-row"):
            with Vertical():
                yield Label("Events panel limit")
                yield Input(value="100", id="settings-events-limit")
            with Vertical():
                yield Label("History panel limit")
                yield Input(value="25", id="settings-history-limit")

        yield Rule()

        # Toggles
        yield Static("[b]Toggles[/b]")
        with Horizontal(id="settings-toggles-row"):
            with Vertical():
                yield Label("Auto-refresh overview")
                yield Switch(value=True, id="settings-auto-refresh")
            with Vertical():
                yield Label("Persist session on exit")
                yield Switch(value=True, id="settings-persist-session")

        yield Rule()

        # Cache management
        yield Static("[b]Cache & State[/b]")
        yield Static("", id="settings-cache-info")
        with Horizontal(id="settings-cache-row"):
            yield Button("Clear event log", id="settings-clear-events")
            yield Button("Clear launcher history", id="settings-clear-history")
            yield Button("Reset session", id="settings-reset-session", variant="error")

        yield Rule()
        yield Static("", id="settings-status")

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_info()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-clear-events":
            self._clear_events()
        elif event.button.id == "settings-clear-history":
            self._clear_launcher_history()
        elif event.button.id == "settings-reset-session":
            self._reset_session()

    def _refresh_info(self) -> None:
        state = getattr(self.app, "state", None)
        if state is None:
            return
        root = state.root
        self.query_one("#settings-state-dir", Static).update(str(root))

        # Cache sizes
        events_path = root / "events.jsonl"
        events_size = events_path.stat().st_size if events_path.exists() else 0
        history_path = root / "launcher_history.json"
        history_size = history_path.stat().st_size if history_path.exists() else 0
        session_path = root / "session.json"
        session_size = session_path.stat().st_size if session_path.exists() else 0

        def _fmt(n: int) -> str:
            if n < 1024:
                return f"{n} B"
            if n < 1024 * 1024:
                return f"{n / 1024:.1f} KB"
            return f"{n / (1024 * 1024):.1f} MB"

        self.query_one("#settings-cache-info", Static).update(
            f"  Event log: {_fmt(events_size)}  |  "
            f"Launcher history: {_fmt(history_size)}  |  "
            f"Session: {_fmt(session_size)}"
        )

    def _clear_events(self) -> None:
        state = getattr(self.app, "state", None)
        if state is None:
            return
        events_path = state.root / "events.jsonl"
        if events_path.exists():
            events_path.write_text("", encoding="utf-8")
        self.query_one("#settings-status", Static).update("Event log cleared.")
        self._refresh_info()

    def _clear_launcher_history(self) -> None:
        state = getattr(self.app, "state", None)
        if state is None:
            return
        history_path = state.root / "launcher_history.json"
        if history_path.exists():
            history_path.write_text('{"schema_version": 4, "history": []}\n', encoding="utf-8")
        self.query_one("#settings-status", Static).update("Launcher history cleared.")
        self._refresh_info()

    def _reset_session(self) -> None:
        state = getattr(self.app, "state", None)
        if state is None:
            return
        session_path = state.root / "session.json"
        if session_path.exists():
            session_path.unlink()
        self.query_one("#settings-status", Static).update("Session reset. Restart TUI to apply.")
        self._refresh_info()
