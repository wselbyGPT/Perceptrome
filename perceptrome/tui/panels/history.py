from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Rule, Static

from perceptrome.tui.history import HistoryIndexer

from .base import BasePanel


class HistoryPanel(BasePanel):
    PANEL_ID = "history"
    TITLE = "History"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indexed_jobs: list = []

    def compose(self) -> ComposeResult:
        yield Static("[b]Run History[/b]", classes="panel-title")
        with Horizontal():
            yield Button("Refresh", id="hist-refresh", variant="primary")
            yield Button("Rerun selected", id="hist-rerun", variant="warning")
            yield Button("Clone to draft", id="hist-clone")
            yield Button("View manifest", id="hist-manifest")
            yield Button("View artifacts", id="hist-artifacts")
        yield DataTable(id="history-table")
        yield Rule()
        yield Static("[b]Run Detail[/b]")
        yield Static("Select a run above.", id="history-detail")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#history-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Status", "Run ID", "Kind", "Title", "Artifacts", "Updated")
        self._refresh_history()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("history", self._refresh_history)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hist-refresh":
            self._refresh_history()
        elif event.button.id == "hist-rerun":
            ok = self.app.rerun_selected_job()
            self._set_detail("Rerun submitted" if ok else "No job to rerun")
        elif event.button.id == "hist-clone":
            ok = self.app.clone_selected_job_to_draft()
            self._set_detail("Cloned to draft" if ok else "No job to clone")
        elif event.button.id == "hist-manifest":
            self.app.open_manifest_for_selected()
        elif event.button.id == "hist-artifacts":
            self.app._set_panel("artifacts")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self._indexed_jobs):
            row = self._indexed_jobs[event.cursor_row]
            self.app.state.set_selected_job(row.run_id)
            self._show_run_detail(row)

    def on_key(self, event) -> None:
        if event.key == "r":
            self.app.rerun_selected_job()
        elif event.key == "c":
            self.app.clone_selected_job_to_draft()
        elif event.key == "m":
            self.app.open_manifest_for_selected()

    def _refresh_history(self) -> None:
        indexer = HistoryIndexer(self.app.state)
        self._indexed_jobs = indexer.merged_jobs(limit=50)

        table = self.query_one("#history-table", DataTable)
        table.clear()
        for row in self._indexed_jobs:
            sts_map = {
                "healthy": "[green]DONE[/]",
                "completed": "[green]DONE[/]",
                "busy": "[cyan]RUN [/]",
                "failed": "[red]FAIL[/]",
                "canceled": "[yellow]CANC[/]",
                "stalled": "[yellow]STAL[/]",
            }
            sts = sts_map.get(row.status, f"[dim]{row.status}[/]")
            table.add_row(
                sts,
                row.run_id[:20],
                row.kind,
                (row.title or "-")[:20],
                str(len(row.artifacts)),
                row.updated_at[:16] if row.updated_at else "-",
            )

        if not self._indexed_jobs:
            self._set_detail("No run history found. Jobs and manifests will appear here after runs complete.")

    def _show_run_detail(self, row) -> None:
        lines = [
            f"Run ID:    {row.run_id}",
            f"Kind:      {row.kind}",
            f"Title:     {row.title}",
            f"Status:    {row.status}",
            f"Created:   {row.created_at}",
            f"Updated:   {row.updated_at}",
            f"Artifacts: {len(row.artifacts)}",
        ]
        if row.manifest_path:
            lines.append(f"Manifest:  {row.manifest_path}")
            try:
                payload = json.loads(Path(row.manifest_path).read_text(encoding="utf-8"))
                parents = payload.get("run_parents") or payload.get("lineage", {}).get("parents") or []
                children = payload.get("run_children") or payload.get("lineage", {}).get("children") or []
                lines.append(f"Parents:   {len(parents)}")
                lines.append(f"Children:  {len(children)}")
            except Exception:
                pass
        if row.failure_summary:
            lines.append(f"Error:     {row.failure_summary.latest_warning_or_error}")
            if row.failure_summary.traceback_path:
                lines.append(f"Traceback: {row.failure_summary.traceback_path}")
        if row.artifacts:
            lines.append("")
            lines.append("Recent artifacts:")
            for art in row.artifacts[-5:]:
                path = art.get("path", "") if isinstance(art, dict) else str(art)
                lines.append(f"  - {path}")
        self._set_detail("\n".join(lines))

    def _set_detail(self, text: str) -> None:
        self.query_one("#history-detail", Static).update(text)
