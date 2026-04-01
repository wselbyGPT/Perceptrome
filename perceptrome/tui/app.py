"""Textual application entrypoint for Perceptrome."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import ContentSwitcher, Input, ListItem, ListView, Static

from .diagnostics import capture_diagnostics
from .events import JobEventBase
from .job_manager import JobStatus, JobManager
from .launcher import DEFAULT_COMMANDS, RankedCommand, derive_context, rank_and_filter_commands
from .panels import ALL_PANELS, BasePanel
from .state_store import StateStore


class DetailSurface(Vertical):
    """Typed detail surface base widget."""

    SURFACE = "detail"
    TITLE = "Details"

    def compose(self) -> ComposeResult:
        yield Static(self.TITLE, classes="detail-title")
        yield Static("", classes="detail-body")

    def set_body(self, body: str) -> None:
        self.query_one(".detail-body", Static).update(body)


class LogsDetailSurface(DetailSurface):
    SURFACE = "logs"
    TITLE = "Logs"


class DiagnosticsDetailSurface(DetailSurface):
    SURFACE = "diagnostics"
    TITLE = "Diagnostics"


class ResourcesDetailSurface(DetailSurface):
    SURFACE = "resources"
    TITLE = "Resources"


class TracebackDetailSurface(DetailSurface):
    SURFACE = "traceback"
    TITLE = "Traceback"


class ArtifactDetailSurface(DetailSurface):
    SURFACE = "artifact"
    TITLE = "Artifact Details"


DETAIL_WIDGETS: dict[str, type[DetailSurface]] = {
    "logs": LogsDetailSurface,
    "diagnostics": DiagnosticsDetailSurface,
    "resources": ResourcesDetailSurface,
    "traceback": TracebackDetailSurface,
    "artifact": ArtifactDetailSurface,
}


class LauncherModal(ModalScreen[str | None]):
    """Interactive launcher with filter + selection support."""

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, app: "PerceptromeTUIApp") -> None:
        super().__init__()
        self._app_ref = app
        self._filtered: list[RankedCommand] = list(app._ranked_commands())

    def compose(self) -> ComposeResult:
        with Container(id="launcher-modal"):
            yield Static("Command Palette", classes="launcher-title")
            yield Input(placeholder="Type to filter commands…", id="launcher-input")
            yield ListView(id="launcher-list")

    def on_mount(self) -> None:
        self._refresh_list()
        self.query_one("#launcher-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filtered = self._app_ref._ranked_commands(query=event.value)
        self._refresh_list()

    def on_input_submitted(self, _: Input.Submitted) -> None:
        self._select_current_or_first()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        command_id = getattr(event.item, "command_id", None)
        self.dismiss(command_id)

    def key_enter(self) -> None:
        self._select_current_or_first()

    def action_dismiss(self) -> None:
        self.dismiss(None)

    def _refresh_list(self) -> None:
        lst = self.query_one("#launcher-list", ListView)
        lst.clear()
        for ranked in self._filtered[:20]:
            command = ranked.command
            status = "available" if ranked.enabled else f"disabled: {ranked.disabled_reason}"
            item = ListItem(Static(f"{command.label}  [{command.category}] • {status}"))
            item.command_id = command.command_id
            lst.append(item)
        if len(lst) > 0:
            lst.index = 0

    def _select_current_or_first(self) -> None:
        lst = self.query_one("#launcher-list", ListView)
        item = lst.highlighted_child
        if item is None and self._filtered:
            self.dismiss(self._filtered[0].command_id)
            return
        self.dismiss(getattr(item, "command_id", None))


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
        padding: 1;
    }
    #launcher-modal {
        width: 80%;
        max-width: 90;
        height: 70%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    .launcher-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #launcher-list {
        height: 1fr;
        margin-top: 1;
    }
    .detail-title {
        text-style: bold;
        margin-bottom: 1;
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
        self._panel_registry = {panel_cls.PANEL_ID: panel_cls for panel_cls in ALL_PANELS}
        self.config_overrides: list[str] = []
        self._active_surface = ""
        self._job_subscription_token: int | None = None

    def compose(self) -> ComposeResult:
        yield Static("Perceptrome • ready", id="top-status")
        with Container(id="workspace"):
            with ContentSwitcher(initial="panel-overview", id="panel-switcher"):
                for panel_id, panel_cls in self._panel_registry.items():
                    with Container(id=f"panel-{panel_id}"):
                        yield panel_cls()
            yield Container(id="detail-host")
        yield Static("No events yet", id="bottom-status")

    def on_mount(self) -> None:
        diagnostics = capture_diagnostics()
        self.state.set_value("python", diagnostics.python_version)
        self.state.set_value("platform", diagnostics.platform)
        self.jobs.reconnect_on_startup()
        self._job_subscription_token = self.jobs.subscribe(self._on_job_event)
        self._set_panel(self.state.active_view if self.state.active_view else "overview")

    def on_unmount(self) -> None:
        if self._job_subscription_token is not None:
            self.jobs.unsubscribe(self._job_subscription_token)

    def _set_panel(self, panel_id: str) -> None:
        if panel_id not in self._panel_registry:
            panel_id = next(iter(self._panel_registry), "overview")
        self.query_one("#panel-switcher", ContentSwitcher).current = f"panel-{panel_id}"
        self.state.set_active_view(panel_id)
        self.query_one("#top-status", Static).update(f"Perceptrome • panel={panel_id}")

    def _set_event_strip(self, message: str) -> None:
        self.query_one("#bottom-status", Static).update(message)

    def _show_detail_surface(self, surface: str, body: str) -> None:
        host = self.query_one("#detail-host", Container)
        host.remove_children()
        host.add_class("-active")
        self._active_surface = surface
        self.state.set_active_detail_surface(surface)
        widget_cls = DETAIL_WIDGETS.get(surface)
        if widget_cls is None:
            return
        card = widget_cls(classes="detail-surface")
        host.mount(card)
        card.set_body(body)
        self._set_event_strip(f"Opened {surface} details")

    def _close_detail_surface(self) -> None:
        self._active_surface = ""
        self.state.set_active_detail_surface(None)
        host = self.query_one("#detail-host", Container)
        host.remove_class("-active")
        host.remove_children()

    def _execute_launcher_command(self, command_id: str) -> None:
        by_id = {entry.command_id: entry for entry in DEFAULT_COMMANDS}
        command = by_id.get(command_id)
        if command is None:
            return
        availability = next((row for row in self._ranked_commands() if row.command.command_id == command_id), None)
        if availability is not None and not availability.enabled:
            self._set_event_strip(availability.disabled_reason or "Command is currently unavailable")
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
            self._set_panel("artifacts")
            path = self.state.open_latest_checkpoint_output()
            if path:
                self.state.set_selected_artifact_path(path)
            selected = self.state.get_session().selected_artifact_path or "No recent output artifact"
            self._show_detail_surface("artifact", f"Selected artifact: {selected}")
        elif action in {"open_failed", "reopen_failed"}:
            self._set_panel("troubleshoot")
            self._show_detail_surface("traceback", "Failed run focused for troubleshooting.")
        elif action == "reset_layout":
            self._close_detail_surface()
            self._set_panel("overview")
            self._set_event_strip("Layout reset")

        self.state.add_launcher_history("command", command=command_id, panel=self.state.active_view)

    def _dispatch_command(self, command_id: str) -> None:
        if command_id == "launcher.open":
            self.push_screen(LauncherModal(self), self._handle_launcher_result)
            return
        self._execute_launcher_command(command_id)

    def _handle_launcher_result(self, command_id: str | None) -> None:
        if command_id:
            self._execute_launcher_command(command_id)

    def _ranked_commands(self, *, query: str = "") -> list[RankedCommand]:
        context = derive_context(active_panel=self.state.active_view, jobs=self.jobs.list_jobs())
        return rank_and_filter_commands(context, query=query, launcher_history=self.state.launcher_history(limit=25))

    def _on_job_event(self, event: object) -> None:
        self.call_from_thread(self._handle_job_event, event)

    def _handle_job_event(self, event: object) -> None:
        for panel in self.query(BasePanel):
            panel.handle_tui_event(event)
        self._persist_job_event(event)

    def _persist_job_event(self, event: object) -> None:
        if not isinstance(event, JobEventBase):
            return
        self.state.apply_job_event(event)

    def action_show_launcher(self) -> None:
        self._dispatch_command("launcher.open")

    def action_show_logs(self) -> None:
        self._dispatch_command("view.logs")

    def action_show_diagnostics(self) -> None:
        self._dispatch_command("view.diagnostics")


def main() -> None:
    PerceptromeTUIApp().run()


if __name__ == "__main__":
    main()
