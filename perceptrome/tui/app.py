"""Textual application entrypoint for Perceptrome."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import ContentSwitcher, Static

from .diagnostics import capture_diagnostics
from .job_manager import JobStatus, JobManager
from .launcher import DEFAULT_COMMANDS, derive_context, rank_commands
from .state_store import StateStore
from .panels import ALL_PANELS


class PerceptromeTUIApp(App[None]):
    """Main Perceptrome text UI shell with thin status strips and active center panel."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #top-status, #bottom-status {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    #workspace {
        height: 1fr;
    }
    #detail-host {
        dock: right;
        width: 46;
        min-width: 36;
        max-width: 60;
        border-left: solid $panel;
        background: $surface;
        display: none;
    }
    #detail-host.-active {
        display: block;
    }
    .detail-surface {
        display: none;
        padding: 1;
    }
    .detail-surface.-active {
        display: block;
    }
    """

    BINDINGS = [
        ("ctrl+p", "show_launcher", "Launcher"),
        ("ctrl+l", "show_logs", "Logs"),
        ("ctrl+d", "show_diagnostics", "Diagnostics"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = StateStore()
        self.jobs = JobManager()
        self.config_overrides: list[str] = []
        self._active_surface = ""

    def compose(self) -> ComposeResult:
        yield Static("Perceptrome • ready", id="top-status")
        with Container(id="workspace"):
            with ContentSwitcher(initial="panel-overview", id="panel-switcher"):
                for panel_cls in ALL_PANELS:
                    with Container(id=f"panel-{panel_cls.PANEL_ID}"):
                        yield panel_cls()
            with Container(id="detail-host"):
                yield Static("Logs", id="surface-title-logs", classes="detail-surface")
                yield Static("Diagnostics", id="surface-title-diagnostics", classes="detail-surface")
                yield Static("Resources", id="surface-title-resources", classes="detail-surface")
                yield Static("Traceback", id="surface-title-traceback", classes="detail-surface")
                yield Static("Artifact Details", id="surface-title-artifact", classes="detail-surface")
        yield Static("No events yet", id="bottom-status")

    def on_mount(self) -> None:
        diagnostics = capture_diagnostics()
        self.state.set_value("python", diagnostics.python_version)
        self.state.set_value("platform", diagnostics.platform)
        self.jobs.reconnect_on_startup()
        self._set_panel(self.state.active_view if self.state.active_view else "overview")

    def _set_panel(self, panel_id: str) -> None:
        self.query_one("#panel-switcher", ContentSwitcher).current = f"panel-{panel_id}"
        self.state.set_active_view(panel_id)
        self.query_one("#top-status", Static).update(f"Perceptrome • panel={panel_id}")

    def _set_event_strip(self, message: str) -> None:
        self.query_one("#bottom-status", Static).update(message)

    def _show_detail_surface(self, surface: str, body: str) -> None:
        host = self.query_one("#detail-host", Container)
        host.add_class("-active")
        self._active_surface = surface
        for row in self.query(".detail-surface"):
            row.remove_class("-active")
        target_id = {
            "logs": "#surface-title-logs",
            "diagnostics": "#surface-title-diagnostics",
            "resources": "#surface-title-resources",
            "traceback": "#surface-title-traceback",
            "artifact": "#surface-title-artifact",
        }.get(surface)
        if target_id is None:
            return
        card = self.query_one(target_id, Static)
        card.update(body)
        card.add_class("-active")
        self._set_event_strip(f"Opened {surface} details")

    def _close_detail_surface(self) -> None:
        self._active_surface = ""
        host = self.query_one("#detail-host", Container)
        host.remove_class("-active")
        for row in self.query(".detail-surface"):
            row.remove_class("-active")

    def _execute_launcher_command(self, command_id: str) -> None:
        by_id = {entry.command_id: entry for entry in DEFAULT_COMMANDS}
        command = by_id.get(command_id)
        if command is None:
            return
        if command.panel_id:
            self._set_panel(command.panel_id)
            self.state.add_launcher_history("open_panel", command=command_id, panel=command.panel_id)
            return

        action = command.action
        active = self.jobs.list_jobs()[0] if self.jobs.list_jobs() else None
        if action == "stop_job" and active and active.status == JobStatus.BUSY:
            self.jobs.cancel(active.id)
            self._set_event_strip(f"Stop requested for {active.id}")
        elif action == "start_job":
            self._set_event_strip("Start requested (use CLI job spec entrypoint).")
        elif action == "rerun_job":
            rerun = self.state.rerun_last_job()
            self._set_event_strip(f"Rerun prepared for {rerun['job_id']}" if rerun else "No recent job to rerun")
        elif action in {"inspect_active", "show_logs", "toggle_logs"}:
            details = f"Active: {active.id} [{active.status.value}]" if active else "No active job"
            self._show_detail_surface("logs", details)
        elif action in {"toggle_diagnostics"}:
            self._show_detail_surface("diagnostics", "Captured diagnostics and environment checks.")
        elif action in {"toggle_resources"}:
            self._show_detail_surface("resources", "CPU/GPU/memory resource snapshot placeholder.")
        elif action in {"toggle_traceback"}:
            failed = self.state.open_last_failed_job()
            trace = failed.failure_summary.traceback_path if failed and failed.failure_summary else "No traceback path"
            self._show_detail_surface("traceback", f"Traceback: {trace}")
        elif action in {"toggle_artifact_details", "open_artifact"}:
            path = self.state.open_latest_checkpoint_output() or "No recent output artifact"
            self._show_detail_surface("artifact", f"Latest artifact: {path}")
        elif action in {"open_failed", "reopen_failed"}:
            self._set_panel("troubleshoot")
            self._show_detail_surface("traceback", "Failed run focused for troubleshooting.")
        elif action == "reset_layout":
            self._close_detail_surface()
            self._set_panel("overview")
            self._set_event_strip("Layout reset")

        self.state.add_launcher_history("command", command=command_id, panel=self.state.active_view)

    def action_show_launcher(self) -> None:
        context = derive_context(active_panel=self.state.active_view, jobs=self.jobs.list_jobs())
        ranked = rank_commands(context)[:12]
        lines = [f"{idx+1:02d}. {entry.label}" for idx, entry in enumerate(ranked)]
        self.notify("Launcher\n" + "\n".join(lines), title="Command Palette")
        if ranked:
            self._execute_launcher_command(ranked[0].command_id)

    def action_show_logs(self) -> None:
        self._execute_launcher_command("view.logs")

    def action_show_diagnostics(self) -> None:
        self._execute_launcher_command("view.diagnostics")


def main() -> None:
    PerceptromeTUIApp().run()


if __name__ == "__main__":
    main()
