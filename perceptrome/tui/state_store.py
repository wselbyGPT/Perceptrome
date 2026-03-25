"""Centralized state object for the Perceptrome TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import TUIEvent
from .history import HistoryLog


@dataclass(slots=True)
class StateStore:
    """Holds active screen metadata and shared key/value state."""

    active_view: str = "overview"
    values: dict[str, Any] = field(default_factory=dict)
    history: HistoryLog = field(default_factory=HistoryLog)

    def set_active_view(self, view: str) -> None:
        self.active_view = view
        self.history.add(TUIEvent(kind="view", message=f"Switched to {view}"))

    def set_value(self, key: str, value: Any) -> None:
        self.values[key] = value
        self.history.add(TUIEvent(kind="state", message=f"Updated {key}", payload={"value": value}))
