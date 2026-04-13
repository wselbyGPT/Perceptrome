from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Rule, Select, Static

from .base import BasePanel

_CATALOG_DIRS = ("accessions", "config")
_CATALOG_GLOBS = ("*.txt",)


def _discover_catalogs() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for directory in _CATALOG_DIRS:
        root = Path(directory)
        if not root.is_dir():
            continue
        for pattern in _CATALOG_GLOBS:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    count = len([ln for ln in lines if ln.strip() and not ln.strip().startswith("#")])
                except Exception:
                    count = 0
                rows.append({"path": str(path), "name": path.name, "dir": directory, "count": count})
    return rows


def _preview_catalog(path: str, limit: int = 20) -> str:
    target = Path(path)
    if not target.exists():
        return "(file not found)"
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except Exception as exc:
        return f"(read error: {exc})"
    accessions = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    total = len(accessions)
    shown = accessions[:limit]
    preview = "\n".join(f"  {acc}" for acc in shown)
    if total > limit:
        preview += f"\n  ... and {total - limit} more"
    return f"Accessions ({total}):\n{preview}"


class DataPanel(BasePanel):
    PANEL_ID = "data"
    TITLE = "Catalogs & Data"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._catalogs: list[dict] = []
        self._selected_catalog: str = ""

    def compose(self) -> ComposeResult:
        yield Static("[b]Catalog Browser[/b]", classes="panel-title")
        yield DataTable(id="catalog-table")
        yield Rule()
        yield Static("[b]Preview[/b]")
        yield Static("Select a catalog above to preview accessions.", id="catalog-preview")
        yield Rule()
        yield Static("[b]Prepare Job[/b]")
        with Horizontal(id="data-form-row"):
            with Vertical():
                yield Label("Accession (single)")
                yield Input(placeholder="e.g. NC_000913", id="data-accession")
            with Vertical():
                yield Label("Source format")
                yield Select(
                    options=[("FASTA", "fasta"), ("GenBank", "genbank")],
                    value="fasta",
                    id="data-source",
                )
            with Vertical():
                yield Label("Tokenizer")
                yield Select(
                    options=[("Base (ACGT)", "base"), ("Codon (3-mer)", "codon"), ("Amino acid", "aa")],
                    value="base",
                    id="data-tokenizer",
                )
        with Horizontal(id="data-btn-row"):
            yield Button("Train single", id="data-train", variant="success")
            yield Button("Stream catalog", id="data-stream", variant="primary")
            yield Button("Validate plasmid", id="data-validate", variant="warning")
            yield Button("Pretrain", id="data-pretrain")
        yield Static("", id="data-status")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#catalog-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Directory", "File", "Accessions")
        self._refresh_catalogs()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one("#catalog-table", DataTable)
        row_idx = event.cursor_row
        if 0 <= row_idx < len(self._catalogs):
            cat = self._catalogs[row_idx]
            self._selected_catalog = str(cat["path"])
            preview = _preview_catalog(self._selected_catalog)
            self.query_one("#catalog-preview", Static).update(preview)
            # Pre-populate draft
            draft = self.app.state.get_draft_job_spec()
            draft["catalog"] = self._selected_catalog
            self.app.state.set_draft_job_spec(draft)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        accession = self.query_one("#data-accession", Input).value.strip()
        source = str(self.query_one("#data-source", Select).value or "fasta")
        tokenizer = str(self.query_one("#data-tokenizer", Select).value or "base")
        catalog = self._selected_catalog

        draft = self.app.state.get_draft_job_spec()
        draft.update({
            "accession": accession,
            "catalog": catalog,
            "source": source,
            "tokenizer": tokenizer,
        })

        status = self.query_one("#data-status", Static)

        if event.button.id == "data-train":
            if not accession:
                status.update("[red]Enter an accession to train on.[/]")
                return
            draft["kind"] = "train_one"
            self.app.state.set_draft_job_spec(draft)
            from ..spec_builders import build_train_one_spec
            spec = build_train_one_spec(
                accession=accession, source=source, tokenizer=tokenizer,
            )
            job_id = self.app.submit_job_spec(spec, title=f"Train {accession}")
            status.update(f"[green]Submitted train_one: {job_id}[/]")

        elif event.button.id == "data-stream":
            if not catalog:
                status.update("[red]Select a catalog file first.[/]")
                return
            draft["kind"] = "stream"
            self.app.state.set_draft_job_spec(draft)
            from ..spec_builders import build_stream_spec
            spec = build_stream_spec(
                catalog=catalog, source=source, tokenizer=tokenizer,
            )
            job_id = self.app.submit_job_spec(spec, title=f"Stream {Path(catalog).stem}")
            status.update(f"[green]Submitted stream: {job_id}[/]")

        elif event.button.id == "data-validate":
            if not accession:
                status.update("[red]Enter an accession to validate.[/]")
                return
            draft["kind"] = "validate_plasmid"
            self.app.state.set_draft_job_spec(draft)
            from ..spec_builders import build_validate_plasmid_spec
            spec = build_validate_plasmid_spec(accession=accession)
            job_id = self.app.submit_job_spec(spec, title=f"Validate {accession}")
            status.update(f"[green]Submitted validate: {job_id}[/]")

        elif event.button.id == "data-pretrain":
            if not catalog:
                status.update("[red]Select a catalog/dataset file first.[/]")
                return
            draft["kind"] = "pretrain"
            self.app.state.set_draft_job_spec(draft)
            from ..spec_builders import build_pretrain_spec
            spec = build_pretrain_spec(dataset=catalog)
            job_id = self.app.submit_job_spec(spec, title=f"Pretrain {Path(catalog).stem}")
            status.update(f"[green]Submitted pretrain: {job_id}[/]")

    def _refresh_catalogs(self) -> None:
        self._catalogs = _discover_catalogs()
        table = self.query_one("#catalog-table", DataTable)
        table.clear()
        for cat in self._catalogs:
            table.add_row(str(cat["dir"]), str(cat["name"]), str(cat["count"]))

        # Also show cache stats
        fasta_count = len(list(Path("cache/fasta").glob("*"))) if Path("cache/fasta").is_dir() else 0
        genbank_count = len(list(Path("cache/genbank").glob("*"))) if Path("cache/genbank").is_dir() else 0
        encoded_count = len(list(Path("cache/encoded").glob("*"))) if Path("cache/encoded").is_dir() else 0
        self.query_one("#data-status", Static).update(
            f"Cache: {fasta_count} FASTA, {genbank_count} GenBank, {encoded_count} encoded"
        )
