from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Label, Rule, Static

from .base import BasePanel

_WATCHED_DIRS = ("runs", "model", "model/checkpoints", "state/tui", "config")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def _dir_file_count(path: Path) -> int:
    try:
        return sum(1 for entry in path.rglob("*") if entry.is_file())
    except Exception:
        return 0


def _collect_disk_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for d in _WATCHED_DIRS:
        p = Path(d)
        if p.is_dir():
            size = _dir_size(p)
            count = _dir_file_count(p)
            rows.append((d, _fmt_bytes(size), str(count), "ok"))
        else:
            rows.append((d, "-", "-", "missing"))
    return rows


def _collect_packages() -> list[tuple[str, str]]:
    packages = []
    for name in ("torch", "numpy", "pyyaml", "textual", "psutil", "biopython", "h5py", "safetensors"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
            packages.append((name, str(ver)))
        except ImportError:
            packages.append((name, "not installed"))
    return packages


def _health_checks() -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = []
    # Config file
    cfg = Path("config/stream_config.yaml")
    checks.append(("Config file", "[green]found[/]" if cfg.exists() else "[red]missing[/]"))
    # Runs directory
    runs = Path("runs")
    checks.append(("Runs directory", "[green]exists[/]" if runs.is_dir() else "[yellow]not created[/]"))
    # Model directory
    model = Path("model")
    checks.append(("Model directory", "[green]exists[/]" if model.is_dir() else "[yellow]not created[/]"))
    # Disk free space
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        tag = "[green]ok[/]" if free_gb > 1 else "[red]low[/]"
        checks.append(("Disk free", f"{tag} ({free_gb:.1f} GB)"))
    except Exception:
        checks.append(("Disk free", "[dim]unknown[/]"))
    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(("CUDA", f"[green]{torch.cuda.get_device_name(0)}[/]"))
        else:
            checks.append(("CUDA", "[yellow]not available[/]"))
    except Exception:
        checks.append(("CUDA", "[dim]torch not importable[/]"))
    return checks


class DiagnosticsPanel(BasePanel):
    PANEL_ID = "diagnostics"
    TITLE = "Diagnostics"

    def compose(self) -> ComposeResult:
        yield Static("[b]Environment Diagnostics[/b]", classes="panel-title")

        # Environment info
        with Horizontal(id="diag-env-row"):
            with Vertical(classes="overview-stat"):
                yield Label("Python", classes="stat-label")
                yield Static(sys.version.split()[0], id="diag-python", classes="stat-value")
            with Vertical(classes="overview-stat"):
                yield Label("Platform", classes="stat-label")
                yield Static("", id="diag-platform", classes="stat-value")
            with Vertical(classes="overview-stat"):
                yield Label("CPUs", classes="stat-label")
                yield Static(str(os.cpu_count() or "?"), id="diag-cpus", classes="stat-value")
            with Vertical(classes="overview-stat"):
                yield Label("Memory", classes="stat-label")
                yield Static("", id="diag-memory", classes="stat-value")

        yield Rule()
        yield Static("[b]Health Checks[/b]")
        yield DataTable(id="diag-health-table")

        yield Rule()
        yield Static("[b]Disk Usage[/b]")
        yield DataTable(id="diag-disk-table")

        yield Rule()
        yield Static("[b]Packages[/b]")
        yield DataTable(id="diag-pkg-table")

        with Horizontal():
            yield Button("Refresh", id="diag-refresh", variant="primary")

    def on_mount(self) -> None:
        super().on_mount()

        health_table = self.query_one("#diag-health-table", DataTable)
        health_table.cursor_type = "row"
        health_table.add_columns("Check", "Status")

        disk_table = self.query_one("#diag-disk-table", DataTable)
        disk_table.cursor_type = "row"
        disk_table.add_columns("Directory", "Size", "Files", "Status")

        pkg_table = self.query_one("#diag-pkg-table", DataTable)
        pkg_table.cursor_type = "row"
        pkg_table.add_columns("Package", "Version")

        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "diag-refresh":
            self._refresh()

    def _refresh(self) -> None:
        from perceptrome.tui.diagnostics import capture_diagnostics

        snapshot = capture_diagnostics()
        payload = snapshot.as_payload()

        self.query_one("#diag-platform", Static).update(str(payload["platform"]))
        mem_total = payload.get("memory_total_mb")
        mem_avail = payload.get("memory_available_mb")
        if mem_total is not None and mem_avail is not None:
            self.query_one("#diag-memory", Static).update(f"{mem_avail} / {mem_total} MB")
        else:
            self.query_one("#diag-memory", Static).update("unknown")

        # Health checks
        health_table = self.query_one("#diag-health-table", DataTable)
        health_table.clear()
        for label, status in _health_checks():
            health_table.add_row(label, status)

        # Disk
        disk_table = self.query_one("#diag-disk-table", DataTable)
        disk_table.clear()
        for row in _collect_disk_rows():
            disk_table.add_row(*row)

        # Packages
        pkg_table = self.query_one("#diag-pkg-table", DataTable)
        pkg_table.clear()
        for name, ver in _collect_packages():
            pkg_table.add_row(name, ver)
