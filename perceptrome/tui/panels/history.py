from __future__ import annotations

import json
from pathlib import Path

from textual.events import Key
from textual.widgets import Static

from perceptrome.tui.history import HistoryIndexer

from .base import BasePanel


class HistoryPanel(BasePanel):
    PANEL_ID = "history"
    TITLE = "History"

    def compose(self):
        yield Static(id="history-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render_history()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("history", self._render_history)

    def on_key(self, event: Key) -> None:
        if event.key == "r":
            self.app.rerun_selected_job()
        elif event.key == "c":
            self.app.clone_selected_job_to_draft()
        elif event.key == "m":
            self.app.open_manifest_for_selected()

    def _render_history(self) -> None:
        body = self.query_one("#history-body", Static)
        indexer = HistoryIndexer(self.app.state)
        merged = indexer.merged_jobs(limit=12)
        if not merged:
            body.update("No persisted run history was found.")
            return

        lines = ["Run History (r rerun, c clone to draft, m open manifest)"]
        for row in merged[:8]:
            run_parent_refs = []
            run_children = []
            if row.manifest_path:
                try:
                    payload = json.loads(Path(row.manifest_path).read_text(encoding="utf-8"))
                    run_parent_refs = list(payload.get("run_parents") or payload.get("lineage", {}).get("parents") or [])
                    run_children = list(payload.get("run_children") or payload.get("lineage", {}).get("children") or [])
                except Exception:
                    pass
            lines.append(f"- {row.run_id} [{row.status}] {row.kind} parent={len(run_parent_refs)} child={len(run_children)}")
            if row.manifest_path:
                lines.append(f"  manifest: {row.manifest_path}")
        body.update("\n".join(lines))
