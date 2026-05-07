"""Textual application entrypoint for Perceptrome."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import ContentSwitcher, Footer, Header, Input, ListItem, ListView, Static, Tab, Tabs

from .diagnostics import capture_diagnostics
from .events import JobEventBase, TUIEvent
from .history import HistoryIndexer
from .job_manager import JobStatus, JobManager
from .launcher import DEFAULT_COMMANDS, RankedCommand, derive_context, rank_and_filter_commands
from .panels import ALL_PANELS, BasePanel
from perceptrome.jobs.spec import JobSpec
from .spec_builders import build_generate_plasmid_spec, build_train_one_spec
from .state_store import StateStore

# Ordered list of (panel_id, short_label) for the tab bar
_PANEL_TABS: tuple[tuple[str, str], ...] = (
    ("overview",    "Overview"),
    ("config",      "Config"),
    ("data",        "Catalogs"),
    ("train",       "Train"),
    ("generate",    "Generate"),
    ("history",     "History"),
    ("artifacts",   "Outputs"),
    ("models",      "Models"),
    ("jobs",        "Jobs"),
    ("metrics",     "Metrics"),
    ("pipeline",    "Pipeline"),
    ("troubleshoot","Trouble"),
    ("events",      "Events"),
    ("diagnostics", "Diag"),
    ("settings",    "Settings"),
)


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

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh_tail)
        self._refresh_tail()

    def _refresh_tail(self) -> None:
        app = self.app
        source = app.resolve_log_source_path() if isinstance(app, PerceptromeTUIApp) else None
        self.set_body(_tail_file(source, fallback="No log source path could be resolved."))


class DiagnosticsDetailSurface(DetailSurface):
    SURFACE = "diagnostics"
    TITLE = "Diagnostics"

    def on_mount(self) -> None:
        self.set_interval(2.0, self._refresh_payload)
        self._refresh_payload()

    def _refresh_payload(self) -> None:
        app = self.app
        if not isinstance(app, PerceptromeTUIApp):
            return
        snapshot = capture_diagnostics()
        app.record_diagnostics_snapshot(snapshot)
        payload = snapshot.as_payload()
        self.set_body(
            "\n".join(
                [
                    f"captured_at: {payload['captured_at']}",
                    f"python: {payload['python_version']}",
                    f"platform: {payload['platform']}",
                    f"cpu_count: {payload['cpu_count']}",
                    f"memory_total_mb: {payload['memory_total_mb']}",
                    f"memory_available_mb: {payload['memory_available_mb']}",
                    f"gpu_present: {payload['gpu_present']}",
                    f"gpu_label: {payload['gpu_label'] or 'n/a'}",
                ]
            )
        )


class ResourcesDetailSurface(DetailSurface):
    SURFACE = "resources"
    TITLE = "Resources"


class TracebackDetailSurface(DetailSurface):
    SURFACE = "traceback"
    TITLE = "Traceback"

    def on_mount(self) -> None:
        self.set_interval(1.5, self._refresh_preview)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        app = self.app
        source = app.resolve_traceback_source_path() if isinstance(app, PerceptromeTUIApp) else None
        self.set_body(_tail_file(source, lines=30, fallback="No traceback source path could be resolved."))


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
    """Main Perceptrome text UI shell."""

    TITLE = "Perceptrome"
    SUB_TITLE = "Genomic ML Toolkit"

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    /* ── Top status bar ────────────────────────────────────────── */
    #top-status {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 2;
        text-style: bold;
    }

    /* ── Tab navigation bar ────────────────────────────────────── */
    #main-tabs {
        height: 3;
        background: $surface;
        border-bottom: tall $primary-darken-3;
        padding: 0 1;
    }
    #main-tabs Tab {
        color: $text-muted;
        padding: 1 2;
    }
    #main-tabs Tab.-active {
        color: $text;
        text-style: bold;
    }

    /* ── Main workspace ────────────────────────────────────────── */
    #workspace {
        height: 1fr;
        layout: horizontal;
    }

    /* Panel switcher fills remaining width */
    #panel-switcher {
        width: 1fr;
        height: 100%;
    }

    /* Each panel container */
    #panel-switcher > Container {
        height: 100%;
        padding: 1 2;
        overflow-y: auto;
    }

    /* ── Detail surface (right drawer) ─────────────────────────── */
    #detail-host {
        dock: right;
        width: 46;
        min-width: 36;
        max-width: 60;
        border-left: tall $primary-darken-3;
        background: $surface;
        display: none;
        padding: 0;
    }
    #detail-host.-active {
        display: block;
    }
    .detail-surface {
        padding: 1 2;
        height: 100%;
        overflow-y: auto;
    }
    .detail-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        border-bottom: solid $panel;
        padding-bottom: 1;
    }
    .detail-body {
        color: $text-muted;
    }

    /* ── Command palette modal ─────────────────────────────────── */
    #launcher-modal {
        width: 80%;
        max-width: 90;
        height: 70%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    .launcher-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #launcher-list {
        height: 1fr;
        margin-top: 1;
    }

    /* ── Bottom status bar ─────────────────────────────────────── */
    #bottom-status {
        height: 1;
        background: $primary-darken-2;
        color: $text-muted;
        padding: 0 2;
    }

    /* ── Panel-specific styles ─────────────────────────────────── */
    .panel-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        border-bottom: solid $panel;
        padding-bottom: 0;
    }

    /* ── Overview stat cards ───────────────────────────────────── */
    #overview-top-row {
        height: auto;
        margin-bottom: 1;
    }
    .overview-stat {
        width: 1fr;
        height: auto;
        padding: 0 2;
        border: round $panel;
        margin: 0 1;
    }
    .stat-label {
        color: $text-muted;
    }
    .stat-value {
        text-style: bold;
        color: $accent;
    }

    /* ── DataTable styling ─────────────────────────────────────── */
    DataTable {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }

    /* ── Sparkline ─────────────────────────────────────────────── */
    Sparkline {
        height: 3;
        margin: 1 0;
    }

    /* ── ProgressBar ───────────────────────────────────────────── */
    ProgressBar {
        height: auto;
        margin: 0 0 1 0;
    }

    /* ── Form layouts ──────────────────────────────────────────── */
    #train-form-r1, #train-form-r2, #train-form-r3,
    #gen-form-r1, #gen-form-r2,
    #data-form-row, #config-form-row,
    #metrics-stats-row {
        height: auto;
        margin-bottom: 1;
    }
    #train-form-r1 > Vertical,
    #train-form-r2 > Vertical,
    #train-form-r3 > Vertical,
    #gen-form-r1 > Vertical,
    #gen-form-r2 > Vertical,
    #data-form-row > Vertical,
    #config-form-row > Vertical {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }

    /* ── Button rows ───────────────────────────────────────────── */
    #train-btn-row, #gen-btn-row, #data-btn-row,
    #config-btn-row, #jobs-action-row {
        height: auto;
        margin: 1 0;
    }
    #train-btn-row > Button, #gen-btn-row > Button,
    #data-btn-row > Button, #config-btn-row > Button,
    #jobs-action-row > Button {
        margin: 0 1 0 0;
    }

    /* ── Metric cards ──────────────────────────────────────────── */
    .metric-card {
        width: 1fr;
        height: auto;
        padding: 0 1;
        border: round $panel;
        margin: 0 1;
    }

    /* ── Rule styling ──────────────────────────────────────────── */
    Rule {
        margin: 1 0;
    }

    /* ── Label styling ─────────────────────────────────────────── */
    Label {
        color: $text-muted;
        margin-bottom: 0;
    }
    """

    BINDINGS = [
        ("ctrl+p",      "show_launcher",   "Launcher"),
        ("ctrl+l",      "show_logs",       "Logs"),
        ("ctrl+d",      "show_diagnostics","Diagnostics"),
        ("ctrl+t",      "show_traceback",  "Traceback"),
        ("tab",         "focus_next",      "Next"),
        ("shift+tab",   "focus_previous",  "Prev"),
        ("escape",      "close_surface",   "Close"),
        ("[",           "prev_panel",      "Prev panel"),
        ("]",           "next_panel",      "Next panel"),
        ("1",           "jump_panel(1)",   "Overview"),
        ("2",           "jump_panel(2)",   "Config"),
        ("3",           "jump_panel(3)",   "Catalogs"),
        ("4",           "jump_panel(4)",   "Train"),
        ("5",           "jump_panel(5)",   "Generate"),
        ("6",           "jump_panel(6)",   "History"),
        ("7",           "jump_panel(7)",   "Outputs"),
        ("8",           "jump_panel(8)",   "Models"),
        ("9",           "jump_panel(9)",   "Jobs"),
        ("q",           "quit",            "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = StateStore()
        self.jobs = JobManager(persist_path=str(self.state.root / "tui_jobs.json"))
        self.history_indexer = HistoryIndexer(self.state)
        self._panel_registry = {panel_cls.PANEL_ID: panel_cls for panel_cls in ALL_PANELS}
        self._panel_order: list[str] = [p for p, _ in _PANEL_TABS if p in {pc.PANEL_ID for pc in ALL_PANELS}]
        self.config_overrides: list[str] = []
        self._active_surface = ""
        self._job_subscription_token: int | None = None
        self._syncing_tab = False  # guard against tab⇄panel sync loops

    def compose(self) -> ComposeResult:
        # Top status bar
        yield Static("Perceptrome  •  ready", id="top-status")

        # Tab navigation bar — one tab per panel
        tab_widgets = [
            Tab(label, id=f"tab-{pid}")
            for pid, label in _PANEL_TABS
            if pid in self._panel_registry
        ]
        yield Tabs(*tab_widgets, id="main-tabs")

        # Main workspace: panel switcher + optional detail drawer
        with Container(id="workspace"):
            with ContentSwitcher(initial="panel-overview", id="panel-switcher"):
                for panel_id, panel_cls in self._panel_registry.items():
                    with Container(id=f"panel-{panel_id}"):
                        yield panel_cls()
            yield Container(id="detail-host")

        # Bottom event strip
        yield Static("Ctrl+P launcher  •  1-9 jump  •  [ / ] cycle  •  q quit", id="bottom-status")

    def on_mount(self) -> None:
        diagnostics = capture_diagnostics()
        self.state.set_value("python", diagnostics.python_version)
        self.state.set_value("platform", diagnostics.platform)
        self.record_diagnostics_snapshot(diagnostics)
        self.jobs.reconnect_on_startup()
        self._job_subscription_token = self.jobs.subscribe(self._on_job_event)
        session = self.state.get_session()
        start_panel = session.last_panel if session.last_panel else "overview"
        self._set_panel(start_panel)
        if session.active_detail_surface:
            self._show_detail_surface(str(session.active_detail_surface), "")

    def on_unmount(self) -> None:
        if self._job_subscription_token is not None:
            self.jobs.unsubscribe(self._job_subscription_token)

    # ── Tab ↔ Panel synchronisation ─────────────────────────────────────────

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Switch content panel when user clicks a tab."""
        if self._syncing_tab or not event.tab or not event.tab.id:
            return
        panel_id = event.tab.id.removeprefix("tab-")
        if panel_id in self._panel_registry and panel_id != self.state.active_view:
            self._set_panel(panel_id, _sync_tab=False)

    def _set_panel(self, panel_id: str, *, _sync_tab: bool = True) -> None:
        if panel_id not in self._panel_registry:
            panel_id = next(iter(self._panel_registry), "overview")
        self.query_one("#panel-switcher", ContentSwitcher).current = f"panel-{panel_id}"
        self.state.set_active_view(panel_id)
        self.query_one("#top-status", Static).update(
            f"Perceptrome  •  {panel_id}"
        )
        if _sync_tab:
            self._syncing_tab = True
            try:
                self.query_one("#main-tabs", Tabs).active = f"tab-{panel_id}"
            except Exception:
                pass
            finally:
                self._syncing_tab = False

    # ── Panel cycling ────────────────────────────────────────────────────────

    def action_prev_panel(self) -> None:
        current = self.state.active_view
        if current in self._panel_order:
            idx = (self._panel_order.index(current) - 1) % len(self._panel_order)
        else:
            idx = 0
        self._set_panel(self._panel_order[idx])

    def action_next_panel(self) -> None:
        current = self.state.active_view
        if current in self._panel_order:
            idx = (self._panel_order.index(current) + 1) % len(self._panel_order)
        else:
            idx = 0
        self._set_panel(self._panel_order[idx])

    def action_jump_panel(self, position: int) -> None:
        idx = position - 1
        if 0 <= idx < len(self._panel_order):
            self._set_panel(self._panel_order[idx])

    # ── Status / event strip ─────────────────────────────────────────────────

    def _set_event_strip(self, message: str) -> None:
        self.query_one("#bottom-status", Static).update(message)

    # ── Detail surface (right drawer) ────────────────────────────────────────

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
        if body:
            card.set_body(body)
        self._set_event_strip(f"Opened {surface}  •  Esc to close")

    def _close_detail_surface(self) -> None:
        self._active_surface = ""
        self.state.set_active_detail_surface(None)
        host = self.query_one("#detail-host", Container)
        host.remove_class("-active")
        host.remove_children()
        self._set_event_strip("Ctrl+P launcher  •  1-9 jump  •  [ / ] cycle  •  q quit")

    # ── Launcher ─────────────────────────────────────────────────────────────

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
            draft = self.state.get_draft_job_spec()
            if draft.get("kind") == "generate_plasmid":
                spec = build_generate_plasmid_spec(
                    output=str(draft.get("output") or "generated.fasta"),
                    length=int(draft.get("length") or 256),
                    config_path=str(draft.get("config_path") or "config/stream_config.yaml"),
                    num_candidates=int(draft.get("num_candidates") or 1),
                    top_k=int(draft.get("top_k") or 4),
                    seed=draft.get("seed"),
                    temperature=float(draft.get("temperature") or 1.0),
                )
            else:
                spec = build_train_one_spec(
                    accession=str(draft.get("accession") or "NC_000913"),
                    config_path=str(draft.get("config_path") or "config/stream_config.yaml"),
                    catalog=draft.get("catalog"),
                    tokenizer=draft.get("tokenizer"),
                    source=draft.get("source"),
                )
            job_id = self.submit_job_spec(spec, title=str(draft.get("title") or ""))
            self._set_event_strip(f"Submitted {spec.kind} ({job_id})")
        elif action == "rerun_job":
            ok = self.rerun_selected_job()
            self._set_event_strip("Rerun submitted" if ok else "No recent job to rerun")
        elif action in {"inspect_active", "show_logs", "toggle_logs"}:
            self._show_detail_surface("logs", "")
        elif action in {"toggle_diagnostics"}:
            self._show_detail_surface("diagnostics", "")
        elif action in {"toggle_resources"}:
            self._show_detail_surface("resources", "CPU/GPU/memory resource snapshot placeholder.")
        elif action in {"toggle_traceback"}:
            self._show_detail_surface("traceback", "")
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

    # ── Job event plumbing ───────────────────────────────────────────────────

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

    # ── Keybinding actions ───────────────────────────────────────────────────

    def action_show_launcher(self) -> None:
        self._dispatch_command("launcher.open")

    def action_show_logs(self) -> None:
        self._dispatch_command("view.logs")

    def action_show_diagnostics(self) -> None:
        self._dispatch_command("view.diagnostics")

    def action_show_traceback(self) -> None:
        self._dispatch_command("view.traceback")

    def action_close_surface(self) -> None:
        if self._active_surface:
            self._close_detail_surface()

    def action_focus_next(self) -> None:
        self.focus_next()

    def action_focus_previous(self) -> None:
        self.focus_previous()

    # ── Job helpers (called by panels) ───────────────────────────────────────

    def submit_job_spec(self, spec, title: str | None = None) -> str:
        job_id = self.jobs.submit(spec, title=title or None)
        self.state.set_selected_job(job_id)
        self.state.set_active_focus(self.state.get_session().active_focus, active_job_id=job_id)
        self.state.append_event(TUIEvent(kind="job", message="submitted", payload={"job_id": job_id, "kind": getattr(spec, "kind", "")}))
        return job_id

    def _selected_job_card(self):
        session = self.state.get_session()
        selected = session.selected_job_id or session.active_job_id
        return next((job for job in self.jobs.list_jobs() if job.id == selected), None)

    def _rebuild_spec_from_card(self, card) -> JobSpec:
        """Reconstruct a JobSpec from a Job card's stored params."""
        return JobSpec(
            kind=card.kind,
            config_path=card.config_path or "config/stream_config.yaml",
            params=dict(card.spec_params),
        )

    def rerun_selected_job(self) -> bool:
        selected = self._selected_job_card()
        if selected is None:
            return False
        spec = self._rebuild_spec_from_card(selected)
        draft = {"kind": selected.kind, **selected.spec_params}
        self.state.set_draft_job_spec(draft)
        self.submit_job_spec(spec, title=f"Rerun {selected.id}")
        return True

    def clone_selected_job_to_draft(self) -> bool:
        selected = self._selected_job_card()
        if selected is None:
            return False
        draft = {"kind": selected.kind, "title": f"Clone of {selected.title}", **selected.spec_params}
        self.state.set_draft_job_spec(draft)
        return True

    def cancel_selected_job(self) -> bool:
        selected = self._selected_job_card()
        return bool(selected and self.jobs.cancel(selected.id))

    def open_artifact(self, path: str | None = None) -> None:
        if path:
            self.state.set_selected_artifact_path(path)
        selected = self.state.get_session().selected_artifact_path or "none"
        self._show_detail_surface("artifact", f"Selected artifact: {selected}")

    def open_manifest_for_selected(self) -> None:
        run_id = self.state.get_session().selected_job_id
        path = self.history_indexer.resolve_manifest_path(run_id=run_id)
        self._show_detail_surface("artifact", f"Manifest: {path or 'not found'}")

    def resolve_log_source_path(self) -> str | None:
        selected_job = self.state.get_session().selected_job_id
        return self.history_indexer.resolve_log_path(run_id=selected_job) or self.history_indexer.resolve_log_path()

    def resolve_traceback_source_path(self) -> str | None:
        failed = self.state.open_last_failed_job()
        if failed and failed.failure_summary and failed.failure_summary.traceback_path:
            from pathlib import Path
            candidate = Path(failed.failure_summary.traceback_path)
            if candidate.exists():
                return str(candidate)
        selected_job = self.state.get_session().selected_job_id
        return self.history_indexer.resolve_traceback_path(run_id=selected_job) or self.history_indexer.resolve_traceback_path()

    def record_diagnostics_snapshot(self, snapshot: object) -> None:
        payload = snapshot.as_payload() if hasattr(snapshot, "as_payload") else {}
        self.state.append_event(TUIEvent(kind="diagnostics", message="resource_snapshot", payload=payload))


def _tail_file(path: str | None, *, lines: int = 40, fallback: str) -> str:
    from pathlib import Path

    if not path:
        return fallback
    target = Path(path)
    if not target.exists() or not target.is_file():
        return f"{path}\n\n(missing on disk)"
    try:
        content = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"{path}\n\n(unable to read: {exc})"
    tail = content[-max(1, lines):] if content else ["(empty file)"]
    return f"{path}\n\n" + "\n".join(tail)


def main() -> None:
    PerceptromeTUIApp().run()


if __name__ == "__main__":
    main()
