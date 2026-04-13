from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Rule, Static

from .base import BasePanel

_MODEL_DIRS = ("model", "model/checkpoints", "model/pretrain")
_MODEL_EXTS = {".pt", ".pth", ".ckpt", ".npz", ".safetensors", ".bin", ".json"}


def _fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _discover_models() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for directory in _MODEL_DIRS:
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in _MODEL_EXTS:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            stat = path.stat()
            rows.append({
                "path": key,
                "name": path.name,
                "dir": str(path.parent),
                "size": _fmt_size(stat.st_size),
                "size_bytes": str(stat.st_size),
                "modified": _fmt_mtime(stat.st_mtime),
                "ext": path.suffix,
            })
    rows.sort(key=lambda r: float(r.get("size_bytes", "0")), reverse=True)
    return rows


def _model_detail(path: str) -> str:
    target = Path(path)
    if not target.exists():
        return "(file not found)"
    stat = target.stat()
    lines = [
        f"Path: {path}",
        f"Size: {_fmt_size(stat.st_size)}",
        f"Modified: {_fmt_mtime(stat.st_mtime)}",
    ]
    if target.suffix == ".json":
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                lines.append("")
                lines.append("Contents (keys):")
                for key in list(data.keys())[:20]:
                    val = data[key]
                    val_str = str(val)[:60] if not isinstance(val, (dict, list)) else f"({type(val).__name__}, len={len(val)})"
                    lines.append(f"  {key}: {val_str}")
        except Exception:
            pass
    return "\n".join(lines)


class ModelsPanel(BasePanel):
    PANEL_ID = "models"
    TITLE = "Models"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._models: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Static("[b]Model & Checkpoint Browser[/b]", classes="panel-title")
        with Horizontal():
            yield Button("Refresh", id="models-refresh", variant="primary")
            yield Button("Open in artifacts", id="models-to-artifacts")
        yield DataTable(id="models-table")
        yield Rule()
        yield Static("[b]Details[/b]")
        yield Static("Select a model file above.", id="models-detail")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#models-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Name", "Directory", "Size", "Modified", "Type")
        self._refresh_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "models-refresh":
            self._refresh_models()
        elif event.button.id == "models-to-artifacts":
            self.app._set_panel("artifacts")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self._models):
            model = self._models[event.cursor_row]
            detail = _model_detail(model["path"])
            self.query_one("#models-detail", Static).update(detail)
            self.app.state.set_selected_artifact_path(model["path"])

    def _refresh_models(self) -> None:
        self._models = _discover_models()
        table = self.query_one("#models-table", DataTable)
        table.clear()
        if not self._models:
            self.query_one("#models-detail", Static).update(
                "No model files found.\n\n"
                "Searched: " + ", ".join(_MODEL_DIRS) + "\n"
                "Extensions: " + ", ".join(sorted(_MODEL_EXTS))
            )
            return
        for m in self._models:
            table.add_row(m["name"], m["dir"], m["size"], m["modified"], m["ext"])
