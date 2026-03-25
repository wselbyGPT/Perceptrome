from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

textual = pytest.importorskip("textual")

from perceptrome.tui.app import PerceptromeTUIApp
from perceptrome.tui.job_manager import JobStatus


@pytest.mark.skipif(not hasattr(textual.app.App, "run_test"), reason="textual test harness unavailable")
def test_tui_app_boots(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    async def _run() -> None:
        app = PerceptromeTUIApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.state.active_view == "overview"
            assert app.query_one("#panel-switcher").current == "panel-overview"

    asyncio.run(_run())


@pytest.mark.skipif(not hasattr(textual.app.App, "run_test"), reason="textual test harness unavailable")
def test_launcher_open_dispatches_action(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    async def _run() -> None:
        app = PerceptromeTUIApp()
        app.jobs._jobs["active"] = SimpleNamespace(id="active", status=JobStatus.BUSY)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_show_launcher()
            await pilot.pause()
            history = app.state.launcher_history(limit=5)
            assert history
            assert any(row.get("action") == "command" for row in history)

    asyncio.run(_run())


@pytest.mark.skipif(not hasattr(textual.app.App, "run_test"), reason="textual test harness unavailable")
def test_panel_switch_action_changes_active_panel(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    async def _run() -> None:
        app = PerceptromeTUIApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._execute_launcher_command("panel.train")
            await pilot.pause()
            assert app.state.active_view == "train"
            assert app.query_one("#panel-switcher").current == "panel-train"

    asyncio.run(_run())
