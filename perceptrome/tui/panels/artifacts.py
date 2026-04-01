from __future__ import annotations

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

    def _render(self) -> None:
        body = self.query_one("#artifacts-body", Static)
        indexer = HistoryIndexer(self.app.state)
        grouped = indexer.artifacts_grouped(limit_runs=40)
        inspection = indexer.inspect_checkpoint()
        if not grouped:
            body.update("No artifacts discovered in persisted jobs or manifests.")
            return

        lines = ["Artifacts by run and role"]
        for run_id, roles in grouped.items():
            lines.append(f"- {run_id}")
            for role, artifacts in sorted(roles.items()):
                lines.append(f"  {role} ({len(artifacts)})")
                for item in artifacts[:5]:
                    mark = "✓" if item.exists else "!"
                    lines.append(f"    {mark} {item.path}")

        lines.append("")
        lines.append("Checkpoint Inspector")
        if inspection is None:
            lines.append("- No checkpoint artifacts found")
        else:
            lines.append(f"- Path: {inspection.path}")
            lines.append(f"  Exists: {inspection.exists}")
            lines.append(f"  MTime: {inspection.mtime}")
            lines.append(f"  Run kind: {inspection.run_kind}")
            if inspection.manifest_path:
                lines.append(f"  Manifest: {inspection.manifest_path}")
            metadata = inspection.metadata
            if metadata:
                lines.append(
                    "  Metadata: "
                    f"config={metadata.get('config_path') or 'n/a'} "
                    f"git={metadata.get('git_sha') or 'n/a'}"
                )

        body.update("\n".join(lines))
