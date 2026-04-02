from __future__ import annotations

from perceptrome.tui.panels import ALL_PANELS


def test_placeholder_panels_replaced() -> None:
    ids = {panel.PANEL_ID for panel in ALL_PANELS}
    assert "generate" in ids
    assert "events" in ids
