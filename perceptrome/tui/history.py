"""In-memory history store for TUI events and commands."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .events import TUIEvent


class HistoryLog:
    """Tracks a bounded rolling history of events for display in the UI."""

    def __init__(self, max_entries: int = 200) -> None:
        self._events: deque[TUIEvent] = deque(maxlen=max_entries)

    def add(self, event: TUIEvent) -> None:
        self._events.append(event)

    def latest(self, limit: int = 25) -> list[TUIEvent]:
        return list(self._events)[-limit:]

    def all(self) -> Iterable[TUIEvent]:
        return tuple(self._events)
