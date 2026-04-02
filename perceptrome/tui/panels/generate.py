from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import Button, Input, Select, Static

from ..spec_builders import build_generate_plasmid_spec, build_generate_protein_spec
from .base import BasePanel


class GeneratePanel(BasePanel):
    PANEL_ID = "generate"
    TITLE = "Generate"

    def compose(self):
        yield Select(options=[("generate_plasmid", "generate_plasmid"), ("generate_protein", "generate_protein")], value="generate_plasmid", id="gen-kind")
        yield Input(placeholder="output path", id="gen-output", value="generated.fasta")
        yield Input(placeholder="length", id="gen-length", value="256")
        yield Input(placeholder="num_candidates", id="gen-n", value="1")
        yield Input(placeholder="top_k", id="gen-topk", value="4")
        yield Input(placeholder="seed (optional)", id="gen-seed")
        yield Input(placeholder="temperature / latent-scale", id="gen-temp", value="1.0")
        with Horizontal():
            yield Button("Submit", id="gen-submit", variant="success")
            yield Button("Reset", id="gen-reset")
            yield Button("Open outputs", id="gen-open")
        yield Static(id="generate-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render("Ready")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gen-reset":
            self.query_one("#gen-output", Input).value = "generated.fasta"
            self.query_one("#gen-length", Input).value = "256"
            self.query_one("#gen-n", Input).value = "1"
            self.query_one("#gen-topk", Input).value = "4"
            self.query_one("#gen-temp", Input).value = "1.0"
            self._render("Reset form")
            return
        if event.button.id == "gen-open":
            self.app._set_panel("artifacts")
            self.app.open_artifact()
            return

        kind = str(self.query_one("#gen-kind", Select).value)
        output = self.query_one("#gen-output", Input).value.strip()
        length = int(self.query_one("#gen-length", Input).value.strip() or "0")
        num_candidates = int(self.query_one("#gen-n", Input).value.strip() or "1")
        top_k = int(self.query_one("#gen-topk", Input).value.strip() or "1")
        seed_raw = self.query_one("#gen-seed", Input).value.strip()
        temperature = float(self.query_one("#gen-temp", Input).value.strip() or "1.0")
        seed = int(seed_raw) if seed_raw else None

        if kind == "generate_protein":
            spec = build_generate_protein_spec(
                output=output,
                length=length,
                num_candidates=num_candidates,
                top_k=top_k,
                seed=seed,
                latent_scale=temperature,
            )
        else:
            spec = build_generate_plasmid_spec(
                output=output,
                length=length,
                num_candidates=num_candidates,
                top_k=top_k,
                seed=seed,
                temperature=temperature,
            )
        self.app.state.set_draft_job_spec({"kind": kind, "output": output, "length": length, "num_candidates": num_candidates, "top_k": top_k, "seed": seed, "temperature": temperature})
        job_id = self.app.submit_job_spec(spec, title=f"{kind} {output}")
        self._render(f"Submitted {job_id}")

    def _render(self, status: str) -> None:
        body = self.query_one("#generate-body", Static)
        selected = self.selected_job_context().get("selected_job")
        lines = [f"Status: {status}"]
        if selected is not None:
            lines.extend(
                [
                    f"Selected job: {selected.id} [{selected.status.value}]",
                    f"Message: {selected.message}",
                    "Recent artifacts:",
                    *[f"- {path}" for path in selected.artifacts[-5:]],
                    "Shortcuts: Ctrl+L logs, Ctrl+T traceback, open outputs button",
                ]
            )
        body.update("\n".join(lines))
