from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ProgressBar,
    Rule,
    Select,
    Sparkline,
    Static,
)

from perceptrome.tui.events import (
    JobCompletedEvent,
    JobFailedEvent,
    JobMetricUpdatedEvent,
    JobStageUpdatedEvent,
    JobStartedEvent,
)
from perceptrome.tui.job_manager import JobStatus

from .base import BasePanel

_JOB_KINDS = [
    ("Train single accession", "train_one"),
    ("Stream over catalog", "stream"),
    ("Generate plasmid", "generate_plasmid"),
    ("Generate protein", "generate_protein"),
    ("Validate plasmid", "validate_plasmid"),
    ("Pretrain (MLM/SME)", "pretrain"),
    ("Design loop", "design_loop"),
]


class TrainPanel(BasePanel):
    PANEL_ID = "train"
    TITLE = "Train"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_data: dict[str, list[float]] = {}

    def compose(self) -> ComposeResult:
        yield Static("[b]Submit Training Job[/b]", classes="panel-title")

        # Row 1: Job kind + config
        with Horizontal(id="train-form-r1"):
            with Vertical():
                yield Label("Job kind")
                yield Select(options=_JOB_KINDS, value="train_one", id="train-kind")
            with Vertical():
                yield Label("Config")
                yield Input(value="config/stream_config.yaml", id="train-config")

        # Row 2: Context-sensitive parameters
        with Horizontal(id="train-form-r2"):
            with Vertical():
                yield Label("Accession / seed")
                yield Input(placeholder="NC_000913 or ATGC...", id="train-accession")
            with Vertical():
                yield Label("Catalog path")
                yield Input(placeholder="accessions/plasmid_accessions.txt", id="train-catalog")
            with Vertical():
                yield Label("Output path")
                yield Input(placeholder="generated.fasta", id="train-output", value="generated.fasta")

        # Row 3: Hyperparameters
        with Horizontal(id="train-form-r3"):
            with Vertical():
                yield Label("Tokenizer")
                yield Select(
                    options=[("base", "base"), ("codon", "codon"), ("aa", "aa")],
                    value="base",
                    id="train-tokenizer",
                )
            with Vertical():
                yield Label("Length")
                yield Input(value="256", id="train-length")
            with Vertical():
                yield Label("Candidates")
                yield Input(value="1", id="train-candidates")
            with Vertical():
                yield Label("Temperature")
                yield Input(value="1.0", id="train-temp")

        # Submit row
        with Horizontal(id="train-btn-row"):
            yield Button("Submit Job", id="train-submit", variant="success")
            yield Button("Reset Form", id="train-reset")
            yield Button("Load from draft", id="train-load-draft", variant="primary")
        yield Static("", id="train-form-status")

        yield Rule()

        # Live monitor section
        yield Static("[b]Live Monitor[/b]")
        yield ProgressBar(total=100, show_eta=False, id="train-progress")
        yield Static("", id="train-active-status")

        yield Rule()
        yield Static("[b]Loss Curve[/b]")
        yield Sparkline([], id="train-sparkline")
        yield Static("", id="train-loss-stats")

        yield Rule()
        yield Static("[b]Running / Recent Jobs[/b]")
        yield DataTable(id="train-jobs-table")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#train-jobs-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Sts", "Job ID", "Kind", "Stage", "Progress", "Loss")
        self._refresh_monitor()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobMetricUpdatedEvent):
            self._loss_data.setdefault(event.job_id, []).append(float(event.latest_value))
            self._loss_data[event.job_id] = self._loss_data[event.job_id][-80:]
        if isinstance(event, (JobStartedEvent, JobStageUpdatedEvent, JobMetricUpdatedEvent,
                              JobCompletedEvent, JobFailedEvent)):
            self.schedule_throttled_render("train-monitor", self._refresh_monitor)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train-reset":
            self._reset_form()
            return
        if event.button.id == "train-load-draft":
            self._load_draft()
            return
        if event.button.id == "train-submit":
            self._submit_job()
            return

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        jobs = self.current_jobs()
        if 0 <= event.cursor_row < len(jobs):
            job = jobs[event.cursor_row]
            self.app.state.set_selected_job(job.id)
            self._refresh_monitor()

    def on_key(self, event) -> None:
        if event.key == "r":
            self.app.rerun_selected_job()
        elif event.key == "c":
            self.app.cancel_selected_job()
        elif event.key == "l":
            self.app.action_show_logs()

    def _submit_job(self) -> None:
        kind = str(self.query_one("#train-kind", Select).value or "train_one")
        config_path = self.query_one("#train-config", Input).value.strip()
        accession = self.query_one("#train-accession", Input).value.strip()
        catalog = self.query_one("#train-catalog", Input).value.strip()
        output = self.query_one("#train-output", Input).value.strip()
        tokenizer = str(self.query_one("#train-tokenizer", Select).value or "base")
        length_str = self.query_one("#train-length", Input).value.strip()
        candidates_str = self.query_one("#train-candidates", Input).value.strip()
        temp_str = self.query_one("#train-temp", Input).value.strip()

        status = self.query_one("#train-form-status", Static)

        try:
            length = int(length_str) if length_str else 256
            candidates = int(candidates_str) if candidates_str else 1
            temperature = float(temp_str) if temp_str else 1.0
        except ValueError as exc:
            status.update(f"[red]Invalid number: {exc}[/]")
            return

        from ..spec_builders import (
            build_design_loop_spec,
            build_generate_plasmid_spec,
            build_generate_protein_spec,
            build_pretrain_spec,
            build_stream_spec,
            build_train_one_spec,
            build_validate_plasmid_spec,
        )

        try:
            if kind == "train_one":
                if not accession:
                    status.update("[red]Accession required for train_one[/]")
                    return
                spec = build_train_one_spec(
                    accession=accession, config_path=config_path,
                    tokenizer=tokenizer, catalog=catalog or None,
                )
                title = f"Train {accession}"
            elif kind == "stream":
                if not catalog:
                    status.update("[red]Catalog path required for stream[/]")
                    return
                spec = build_stream_spec(
                    catalog=catalog, config_path=config_path, tokenizer=tokenizer,
                )
                title = f"Stream {catalog}"
            elif kind == "generate_plasmid":
                spec = build_generate_plasmid_spec(
                    output=output or "generated.fasta", length=length,
                    config_path=config_path, num_candidates=candidates,
                    temperature=temperature,
                )
                title = f"GenPlasmid {output}"
            elif kind == "generate_protein":
                spec = build_generate_protein_spec(
                    output=output or "generated.faa", length=length,
                    config_path=config_path, num_candidates=candidates,
                    latent_scale=temperature,
                )
                title = f"GenProtein {output}"
            elif kind == "validate_plasmid":
                if not accession:
                    status.update("[red]Accession required for validate[/]")
                    return
                spec = build_validate_plasmid_spec(
                    accession=accession, config_path=config_path,
                )
                title = f"Validate {accession}"
            elif kind == "pretrain":
                if not catalog:
                    status.update("[red]Dataset path required for pretrain[/]")
                    return
                spec = build_pretrain_spec(
                    dataset=catalog, config_path=config_path,
                )
                title = f"Pretrain {catalog}"
            elif kind == "design_loop":
                if not accession:
                    status.update("[red]Seed sequence required for design_loop[/]")
                    return
                spec = build_design_loop_spec(
                    seed_sequence=accession, config_path=config_path,
                )
                title = f"DesignLoop"
            else:
                status.update(f"[red]Unknown kind: {kind}[/]")
                return

            job_id = self.app.submit_job_spec(spec, title=title)
            status.update(f"[green]Submitted {kind}: {job_id}[/]")
        except Exception as exc:
            status.update(f"[red]{exc}[/]")

    def _reset_form(self) -> None:
        self.query_one("#train-kind", Select).value = "train_one"
        self.query_one("#train-config", Input).value = "config/stream_config.yaml"
        self.query_one("#train-accession", Input).value = ""
        self.query_one("#train-catalog", Input).value = ""
        self.query_one("#train-output", Input).value = "generated.fasta"
        self.query_one("#train-tokenizer", Select).value = "base"
        self.query_one("#train-length", Input).value = "256"
        self.query_one("#train-candidates", Input).value = "1"
        self.query_one("#train-temp", Input).value = "1.0"
        self.query_one("#train-form-status", Static).update("Form reset.")

    def _load_draft(self) -> None:
        draft = self.app.state.get_draft_job_spec()
        if not draft:
            self.query_one("#train-form-status", Static).update("[dim]No draft available[/]")
            return
        if draft.get("kind"):
            self.query_one("#train-kind", Select).value = str(draft["kind"])
        if draft.get("accession"):
            self.query_one("#train-accession", Input).value = str(draft["accession"])
        if draft.get("catalog"):
            self.query_one("#train-catalog", Input).value = str(draft["catalog"])
        if draft.get("output"):
            self.query_one("#train-output", Input).value = str(draft["output"])
        if draft.get("tokenizer"):
            self.query_one("#train-tokenizer", Select).value = str(draft["tokenizer"])
        if draft.get("length"):
            self.query_one("#train-length", Input).value = str(draft["length"])
        if draft.get("config_path"):
            self.query_one("#train-config", Input).value = str(draft["config_path"])
        self.query_one("#train-form-status", Static).update("[green]Loaded from draft[/]")

    def _refresh_monitor(self) -> None:
        jobs = self.current_jobs()
        active = [j for j in jobs if getattr(j, "status", None) == JobStatus.BUSY]

        # Progress bar + status for first active job
        bar = self.query_one("#train-progress", ProgressBar)
        active_status = self.query_one("#train-active-status", Static)
        if active:
            job = active[0]
            pct = getattr(getattr(job, "progress", None), "percent", None)
            if isinstance(pct, (int, float)):
                bar.update(progress=min(100, int(pct * 100)))
            else:
                bar.update(progress=0)
            step = getattr(getattr(job, "progress", None), "step", None) or "-"
            total = getattr(getattr(job, "progress", None), "total_steps", None) or "-"
            epoch = getattr(getattr(job, "progress", None), "epoch", None)
            epoch_str = f"  epoch {epoch}" if epoch is not None else ""
            loss = getattr(getattr(job, "metrics", None), "latest_loss", None)
            loss_str = f"  loss={loss:.6f}" if isinstance(loss, float) else ""
            active_status.update(
                f"  [cyan]{job.title}[/]  |  {job.current_stage or '-'}  |  "
                f"step {step}/{total}{epoch_str}{loss_str}"
            )
        else:
            bar.update(progress=0)
            active_status.update("  [dim]No active jobs[/]")

        # Sparkline
        spark = self.query_one("#train-sparkline", Sparkline)
        loss_stats = self.query_one("#train-loss-stats", Static)
        # Show selected or first active job's loss
        ctx = self.selected_job_context()
        sel_id = ctx.get("selected_job_id") or (active[0].id if active else None)
        vals = list(self._loss_data.get(sel_id, [])) if sel_id else []
        spark.data = vals if vals else [0.0]
        if vals:
            rolling = sum(vals[-10:]) / min(10, len(vals))
            loss_stats.update(
                f"  current={vals[-1]:.6f}  min={min(vals):.6f}  max={max(vals):.6f}  "
                f"rolling(10)={rolling:.6f}  samples={len(vals)}"
            )
        else:
            loss_stats.update("  [dim]No loss data yet[/]")

        # Jobs table
        table = self.query_one("#train-jobs-table", DataTable)
        table.clear()
        for job in jobs[:10]:
            pct = getattr(getattr(job, "progress", None), "percent", None)
            pct_str = f"{pct * 100:.0f}%" if isinstance(pct, (int, float)) else "-"
            loss = getattr(getattr(job, "metrics", None), "latest_loss", None)
            loss_str = f"{loss:.5f}" if isinstance(loss, float) else "-"
            sts = {
                JobStatus.BUSY: "[cyan]RUN[/]",
                JobStatus.HEALTHY: "[green]OK [/]",
                JobStatus.FAILED: "[red]ERR[/]",
                JobStatus.STALLED: "[yellow]STL[/]",
                JobStatus.WARNING: "[yellow]WRN[/]",
            }.get(job.status, str(job.status.value))
            table.add_row(
                sts,
                job.id[:20],
                job.kind,
                getattr(job, "current_stage", "") or "-",
                pct_str,
                loss_str,
            )
