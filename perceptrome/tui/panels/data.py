from __future__ import annotations

from pathlib import Path

from textual.containers import Horizontal
from textual.widgets import Button, Input, Select, Static

from .base import BasePanel


class DataPanel(BasePanel):
    PANEL_ID = "data"
    TITLE = "Data"

    def compose(self):
        yield Input(placeholder="accession", id="data-accession")
        yield Input(placeholder="catalog path", id="data-catalog")
        yield Select(options=[("fasta", "fasta"), ("genbank", "genbank")], value="fasta", id="data-source")
        yield Select(options=[("base", "base"), ("codon", "codon"), ("aa", "aa")], value="base", id="data-tokenizer")
        with Horizontal():
            yield Button("Prepare train", id="data-train", variant="success")
            yield Button("Prepare stream", id="data-stream")
            yield Button("Prepare validate", id="data-validate")
        yield Static(id="data-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        accession = self.query_one("#data-accession", Input).value.strip()
        catalog = self.query_one("#data-catalog", Input).value.strip()
        source = str(self.query_one("#data-source", Select).value or "fasta")
        tokenizer = str(self.query_one("#data-tokenizer", Select).value or "base")
        kind = "train_one"
        if event.button.id == "data-stream":
            kind = "stream"
        elif event.button.id == "data-validate":
            kind = "validate_plasmid"
        self.app.state.set_draft_job_spec(
            {
                **self.app.state.get_draft_job_spec(),
                "kind": kind,
                "accession": accession,
                "catalog": catalog,
                "source": source,
                "tokenizer": tokenizer,
            }
        )
        self._render()

    def _render(self) -> None:
        body = self.query_one("#data-body", Static)
        catalog_files = sorted(str(p) for p in Path("config").glob("*catalog*"))
        draft = self.app.state.get_draft_job_spec()
        lines = [
            f"Draft kind: {draft.get('kind', '(unset)')}",
            f"Draft accession: {draft.get('accession', '(none)')}",
            f"Draft catalog: {draft.get('catalog', '(none)')}",
            "",
            "Catalog-like files:",
            *([f"- {path}" for path in catalog_files] or ["- none found"]),
            "",
            f"Cache FASTA dir exists: {Path('cache/fasta').exists()}",
            f"Cache GenBank dir exists: {Path('cache/genbank').exists()}",
        ]
        body.update("\n".join(lines))
