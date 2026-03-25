from __future__ import annotations

import json
from typing import Any

from textual.widgets import Static

from ..config_tools import apply_override, deep_copy, load_base_config, validate_effective
from ..state_store import StateStore
from .base import BasePanel


class ConfigPanel(BasePanel):
    PANEL_ID = "config"
    TITLE = "Config"

    def compose(self):
        yield Static(id="config-body")

    def on_mount(self) -> None:
        super().on_mount()
        self.render_config()

    def render_config(self) -> None:
        body = self.query_one("#config-body", Static)
        base_cfg = load_base_config("config/stream_config.yaml")
        overrides = self._load_cli_overrides()
        effective_cfg = deep_copy(base_cfg)
        applied: list[str] = []
        for item in overrides:
            ok, _ = apply_override(effective_cfg, item)
            if ok:
                applied.append(item)
        checks = validate_effective(effective_cfg)

        base_preview = json.dumps(base_cfg, indent=2, sort_keys=True)[:900]
        effective_preview = json.dumps(effective_cfg, indent=2, sort_keys=True)[:900]
        lines = [
            "Config source: config/stream_config.yaml",
            f"CLI overrides ({len(applied)} applied): {', '.join(applied) if applied else '(none)'}",
            "",
            "Validation / preflight:",
            *(f" - {line}" for line in checks),
            "",
            "Base values (truncated):",
            base_preview,
            "",
            "Effective values (truncated):",
            effective_preview,
        ]
        body.update("\n".join(lines))

    def _load_cli_overrides(self) -> list[str]:
        app_overrides = getattr(self.app, "config_overrides", None)
        if isinstance(app_overrides, list):
            return [str(item) for item in app_overrides]
        store: StateStore | None = getattr(self.app, "state", None)
        if store is None:
            return []
        session = store.get_session()
        raw = session.drawer_toggles.get("config_overrides")
        if isinstance(raw, str):
            return [item.strip() for item in raw.split() if item.strip()]
        return []
