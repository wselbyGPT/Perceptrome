"""Typed events emitted inside the Perceptrome TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class TUIEvent:
    """Simple structured event payload used by the state and history subsystems."""

    kind: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
