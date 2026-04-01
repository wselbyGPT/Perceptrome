from __future__ import annotations

from perceptrome.tui.history import HistoryIndexer
from textual.widgets import Static

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

    def _render_history(self) -> None:
        body = self.query_one("#history-body", Static)
        indexer = HistoryIndexer(self.app.state)
        merged = indexer.merged_jobs(limit=12)
        if not merged:
            body.update("No persisted run history was found.")
            return

        lines = ["Run History"]
        for row in merged[:8]:
            run_parent_refs = []
            run_children = []
            if row.manifest_path:
                try:
                    import json
                    from pathlib import Path

                    payload = json.loads(Path(row.manifest_path).read_text(encoding="utf-8"))
                    run_parent_refs = list(payload.get("run_parents") or payload.get("lineage", {}).get("parents") or [])
                    run_children = list(payload.get("run_children") or payload.get("lineage", {}).get("children") or [])
                except Exception:
                    run_parent_refs = []
                    run_children = []

            artifact_links = [str(item.get("path") or "") for item in row.artifacts if isinstance(item, dict) and item.get("path")]
            artifact_text = artifact_links[-1] if artifact_links else "none"
            lineage = f"parents={len(run_parent_refs)} children={len(run_children)}"
            lines.extend(
                [
                    f"- {row.run_id} [{row.status}] {row.kind}",
                    f"  Lineage: {lineage}",
                    f"  Latest artifact: {artifact_text}",
                ]
            )
        body.update("\n".join(lines))
