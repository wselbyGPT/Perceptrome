from __future__ import annotations

from perceptrome.tui.diagnostics import capture_diagnostics


def test_capture_diagnostics_contains_structured_resource_payload() -> None:
    snapshot = capture_diagnostics()
    payload = snapshot.as_payload()

    assert payload["python_version"]
    assert payload["platform"]
    assert isinstance(payload["cpu_count"], int)
    assert payload["cpu_count"] >= 1
    assert "captured_at" in payload
    assert "gpu_present" in payload
