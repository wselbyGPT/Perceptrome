from __future__ import annotations

from pathlib import Path

from perceptrome.tui.job_manager import JobManager
from perceptrome.tui.state_store import StateStore


def test_job_manager_uses_same_state_root(tmp_path: Path) -> None:
    store = StateStore(state_root=str(tmp_path / "state" / "tui"))
    manager = JobManager(persist_path=str(store.root / "tui_jobs.json"))
    assert str(store.root) in str(manager._persist_path)
