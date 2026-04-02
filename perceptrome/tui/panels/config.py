from __future__ import annotations

import json

from textual.containers import Horizontal
from textual.widgets import Button, Input, Static

from ..config_tools import apply_override, deep_copy, load_base_config, validate_effective
from .base import BasePanel


class ConfigPanel(BasePanel):
    PANEL_ID = "config"
    TITLE = "Config"

    def compose(self):
        yield Input(placeholder="override e.g. training.batch_size=8", id="config-override")
        with Horizontal():
            yield Button("Apply", id="cfg-apply", variant="success")
            yield Button("Reset", id="cfg-reset")
            yield Button("Use selected run config", id="cfg-from-run")
            yield Button("Prepare train", id="cfg-train")
            yield Button("Prepare generate", id="cfg-generate")
        yield Static(id="config-body")

    def on_mount(self) -> None:
        super().on_mount()
        self.render_config()

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
        self.render_config()

    def render_config(self) -> None:
        body = self.query_one("#config-body", Static)
        base_cfg = load_base_config("config/stream_config.yaml")
        overrides = [str(item) for item in getattr(self.app, "config_overrides", [])]
        effective_cfg = deep_copy(base_cfg)
        applied: list[str] = []
        for item in overrides:
            ok, _ = apply_override(effective_cfg, item)
            if ok:
                applied.append(item)
        checks = validate_effective(effective_cfg)
        draft = self.app.state.get_draft_job_spec()
        lines = [
            "Config source: config/stream_config.yaml",
            f"Draft kind: {draft.get('kind', '(unset)')}",
            f"Overrides ({len(applied)}): {', '.join(applied) if applied else '(none)'}",
            "",
            "Validation:",
            *(f" - {line}" for line in checks),
            "",
            "Effective (truncated):",
            json.dumps(effective_cfg, indent=2, sort_keys=True)[:1200],
        ]
        body.update("\n".join(lines))
