from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Rule, Static

from ..config_tools import apply_override, deep_copy, load_base_config, validate_effective
from .base import BasePanel


class ConfigPanel(BasePanel):
    PANEL_ID = "config"
    TITLE = "Config"

    def compose(self) -> ComposeResult:
        yield Static("[b]Configuration Editor[/b]", classes="panel-title")

        with Horizontal(id="config-form-row"):
            with Vertical():
                yield Label("Override (key=value)")
                yield Input(placeholder="e.g. training.batch_size=8", id="config-override")
            with Vertical():
                yield Label("Config path")
                yield Input(value="config/stream_config.yaml", id="config-path")

        with Horizontal(id="config-btn-row"):
            yield Button("Apply override", id="cfg-apply", variant="success")
            yield Button("Reset overrides", id="cfg-reset", variant="warning")
            yield Button("Use run config", id="cfg-from-run")
            yield Button("Set draft: Train", id="cfg-train", variant="primary")
            yield Button("Set draft: Generate", id="cfg-generate", variant="primary")

        yield Rule()
        yield Static("[b]Validation[/b]")
        yield DataTable(id="config-checks-table")

        yield Rule()
        yield Static("[b]Active Overrides[/b]")
        yield Static("(none)", id="config-overrides-display")

        yield Rule()
        yield Static("[b]Effective Config[/b]")
        yield Static("", id="config-body")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#config-checks-table", DataTable)
        table.add_columns("Check", "Result")
        self._render_config()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        draft = self.app.state.get_draft_job_spec()
        if event.button.id == "cfg-apply":
            value = self.query_one("#config-override", Input).value.strip()
            if value:
                overrides = list(self.app.config_overrides)
                overrides.append(value)
                self.app.config_overrides = overrides
                self.app.state.set_draft_job_spec({**draft, "overrides": overrides})
        elif event.button.id == "cfg-reset":
            self.app.config_overrides = []
            self.app.state.set_draft_job_spec({k: v for k, v in draft.items() if k != "overrides"})
        elif event.button.id == "cfg-train":
            self.app.state.set_draft_job_spec({**draft, "kind": "train_one"})
        elif event.button.id == "cfg-generate":
            self.app.state.set_draft_job_spec({**draft, "kind": "generate_plasmid"})
        elif event.button.id == "cfg-from-run":
            selected = self.selected_job_context().get("selected_job")
            if selected is not None:
                self.app.state.set_draft_job_spec({**draft, "kind": selected.kind})
        self._render_config()

    def _render_config(self) -> None:
        config_path = self.query_one("#config-path", Input).value.strip()
        base_cfg = load_base_config(config_path or "config/stream_config.yaml")
        overrides = [str(item) for item in getattr(self.app, "config_overrides", [])]
        effective_cfg = deep_copy(base_cfg)
        applied: list[str] = []
        for item in overrides:
            ok, _ = apply_override(effective_cfg, item)
            if ok:
                applied.append(item)
        checks = validate_effective(effective_cfg)
        draft = self.app.state.get_draft_job_spec()

        # Overrides display
        self.query_one("#config-overrides-display", Static).update(
            "\n".join(f"  {o}" for o in applied) if applied else "[dim](none)[/]"
        )

        # Validation table
        table = self.query_one("#config-checks-table", DataTable)
        table.clear()
        for check in checks:
            if check.startswith("PASS"):
                table.add_row("[green]PASS[/]", check.removeprefix("PASS: "))
            else:
                table.add_row("[red]FAIL[/]", check.removeprefix("FAIL: "))

        # Effective config
        lines = [
            f"Source: {config_path or 'config/stream_config.yaml'}",
            f"Draft kind: {draft.get('kind', '(unset)')}",
            "",
        ]
        lines.append(json.dumps(effective_cfg, indent=2, sort_keys=True)[:2000])
        self.query_one("#config-body", Static).update("\n".join(lines))
