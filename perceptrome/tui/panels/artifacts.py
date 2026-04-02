from __future__ import annotations

from pathlib import Path

from textual.events import Key
from textual.widgets import Static

from perceptrome.tui.history import HistoryIndexer

from .base import BasePanel


class ArtifactsPanel(BasePanel):
    PANEL_ID = "artifacts"
    TITLE = "Artifacts"

    def compose(self):
        yield Static(id="artifacts-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("artifacts", self._render)

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            self.app.open_artifact()

    def _render(self) -> None:
        body = self.query_one("#artifacts-body", Static)
        indexer = HistoryIndexer(self.app.state)
        grouped = indexer.artifacts_grouped(limit_runs=40)
        if not grouped:
            body.update("No artifacts discovered in persisted jobs or manifests.")
            return

        lines = ["Artifacts (Enter pins selected)"]
        for run_id, roles in grouped.items():
            lines.append(f"- {run_id}")
            for role, artifacts in sorted(roles.items()):
                lines.append(f"  {role} ({len(artifacts)})")
                for item in artifacts[:4]:
                    p = Path(item.path)
                    preview = ""
                    if p.exists() and p.suffix in {".json", ".txt", ".fasta", ".faa"}:
                        try:
                            preview = p.read_text(encoding="utf-8", errors="replace").splitlines()[0][:80]
                        except Exception:
                            preview = ""
                    lines.append(f"    {'✓' if item.exists else '!'} {item.path} {preview}")
        body.update("\n".join(lines[:200]))
