from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Rule, Select, Static

from perceptrome.tui.events import JobArtifactEmittedEvent, JobCompletedEvent
from perceptrome.tui.history import HistoryIndexer

from .base import BasePanel

_PREVIEW_EXTS = {".json", ".txt", ".fasta", ".faa", ".fa", ".csv", ".tsv", ".yaml", ".yml", ".log"}


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _fmt_mtime(path: Path) -> str:
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%m-%d %H:%M")


def _preview_file(path: str, limit: int = 30) -> str:
    target = Path(path)
    if not target.exists():
        return f"(file not found: {path})"
    if target.suffix not in _PREVIEW_EXTS:
        return f"Path: {path}\nSize: {_fmt_size(target)}\n(binary file, no preview)"
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"(read error: {exc})"
    header = f"Path: {path}\nSize: {_fmt_size(target)}\nLines: {len(lines)}\n{'─' * 50}\n"
    shown = lines[:limit]
    body = "\n".join(shown)
    if len(lines) > limit:
        body += f"\n... ({len(lines) - limit} more lines)"
    return header + body


class ArtifactsPanel(BasePanel):
    PANEL_ID = "artifacts"
    TITLE = "Outputs & Artifacts"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flat_rows: list[dict[str, str]] = []
        self._filter_role: str = "all"

    def compose(self) -> ComposeResult:
        yield Static("[b]Outputs & Artifacts[/b]", classes="panel-title")
        with Horizontal():
            yield Button("Refresh", id="art-refresh", variant="primary")
            yield Select(
                options=[
                    ("All roles", "all"),
                    ("Checkpoints", "checkpoint"),
                    ("Outputs", "output"),
                    ("Logs", "log"),
                    ("Manifests", "manifest"),
                    ("Configs", "config_snapshot"),
                    ("Scorecards", "scorecard"),
                    ("Summaries", "summary"),
                ],
                value="all",
                id="art-role-filter",
            )
        yield DataTable(id="art-table")
        yield Rule()
        yield Static("[b]Preview[/b]")
        yield Static("Select a file above to preview.", id="art-preview")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#art-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Run", "Role", "File", "Size", "Modified", "Exists")
        self._refresh_artifacts()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, (JobArtifactEmittedEvent, JobCompletedEvent)):
            self.schedule_throttled_render("artifacts", self._refresh_artifacts)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "art-refresh":
            self._refresh_artifacts()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "art-role-filter":
            self._filter_role = str(event.value or "all")
            self._refresh_artifacts()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self._flat_rows):
            row = self._flat_rows[event.cursor_row]
            path = row["path"]
            self.app.state.set_selected_artifact_path(path)
            preview = _preview_file(path)
            self.query_one("#art-preview", Static).update(preview)
            self.app._show_detail_surface("artifact", f"Selected: {path}")

    def _refresh_artifacts(self) -> None:
        indexer = HistoryIndexer(self.app.state)
        grouped = indexer.artifacts_grouped(limit_runs=50)

        self._flat_rows = []
        for run_id, by_role in grouped.items():
            for role, artifacts in sorted(by_role.items()):
                if self._filter_role != "all" and role != self._filter_role:
                    continue
                for item in artifacts:
                    p = Path(item.path)
                    self._flat_rows.append({
                        "run_id": run_id[:16],
                        "role": role,
                        "name": p.name,
                        "path": str(item.path),
                        "size": _fmt_size(p),
                        "modified": _fmt_mtime(p),
                        "exists": "yes" if item.exists else "NO",
                    })

        table = self.query_one("#art-table", DataTable)
        table.clear()
        for row in self._flat_rows[:200]:
            exists_display = "[green]yes[/]" if row["exists"] == "yes" else "[red]NO[/]"
            table.add_row(
                row["run_id"], row["role"], row["name"],
                row["size"], row["modified"], exists_display,
            )

        if not self._flat_rows:
            self.query_one("#art-preview", Static).update(
                "No artifacts found. Run a job to generate outputs."
            )
