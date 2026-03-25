"""Perceptrome Textual TUI package."""

from __future__ import annotations

try:
    from .app import PerceptromeTUIApp
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    PerceptromeTUIApp = None  # type: ignore

__all__ = ["PerceptromeTUIApp"]
