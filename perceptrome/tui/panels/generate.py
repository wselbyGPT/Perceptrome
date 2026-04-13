from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, ProgressBar, Rule, Select, Static

from perceptrome.tui.events import (
    JobArtifactEmittedEvent,
    JobCompletedEvent,
    JobFailedEvent,
    JobMetricUpdatedEvent,
    JobStageUpdatedEvent,
    JobStartedEvent,
)
from perceptrome.tui.job_manager import JobStatus

from ..spec_builders import build_generate_plasmid_spec, build_generate_protein_spec
from .base import BasePanel


class GeneratePanel(BasePanel):
    PANEL_ID = "generate"
    TITLE = "Generate"

    def compose(self) -> ComposeResult:
        yield Static("[b]Sequence Generation[/b]", classes="panel-title")

        with Horizontal(id="gen-form-r1"):
            with Vertical():
                yield Label("Kind")
                yield Select(
                    options=[("Plasmid", "generate_plasmid"), ("Protein", "generate_protein")],
                    value="generate_plasmid",
                    id="gen-kind",
                )
            with Vertical():
                yield Label("Output path")
                yield Input(placeholder="output path", id="gen-output", value="generated.fasta")
            with Vertical():
                yield Label("Length")
                yield Input(placeholder="length", id="gen-length", value="256")

        with Horizontal(id="gen-form-r2"):
            with Vertical():
                yield Label("Candidates")
                yield Input(placeholder="num_candidates", id="gen-n", value="1")
            with Vertical():
                yield Label("Top-K")
                yield Input(placeholder="top_k", id="gen-topk", value="4")
            with Vertical():
                yield Label("Seed")
                yield Input(placeholder="seed (optional)", id="gen-seed")
            with Vertical():
                yield Label("Temperature")
                yield Input(placeholder="temperature", id="gen-temp", value="1.0")

        with Horizontal(id="gen-btn-row"):
            yield Button("Generate", id="gen-submit", variant="success")
            yield Button("Reset", id="gen-reset")
            yield Button("Open outputs", id="gen-open")

        yield Static("", id="gen-status")
        yield Rule()

        yield Static("[b]Generation Progress[/b]")
        yield ProgressBar(total=100, show_eta=False, id="gen-progress")
        yield Static("", id="gen-active-info")
        yield Rule()

        yield Static("[b]Recent Generation Jobs[/b]")
        yield DataTable(id="gen-jobs-table")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#gen-jobs-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Status", "Job ID", "Kind", "Stage", "Artifacts")
        self._refresh_jobs()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, (JobStartedEvent, JobStageUpdatedEvent,
                              JobCompletedEvent, JobFailedEvent,
                              JobArtifactEmittedEvent)):
            self.schedule_throttled_render("generate", self._refresh_jobs)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gen-reset":
            self.query_one("#gen-output", Input).value = "generated.fasta"
            self.query_one("#gen-length", Input).value = "256"
            self.query_one("#gen-n", Input).value = "1"
            self.query_one("#gen-topk", Input).value = "4"
            self.query_one("#gen-seed", Input).value = ""
            self.query_one("#gen-temp", Input).value = "1.0"
            self.query_one("#gen-status", Static).update("Form reset.")
            return
        if event.button.id == "gen-open":
            self.app._set_panel("artifacts")
            self.app.open_artifact()
            return
        if event.button.id != "gen-submit":
            return

        status = self.query_one("#gen-status", Static)
        try:
            kind = str(self.query_one("#gen-kind", Select).value)
            output = self.query_one("#gen-output", Input).value.strip()
            length = int(self.query_one("#gen-length", Input).value.strip() or "256")
            num_candidates = int(self.query_one("#gen-n", Input).value.strip() or "1")
            top_k = int(self.query_one("#gen-topk", Input).value.strip() or "4")
            seed_raw = self.query_one("#gen-seed", Input).value.strip()
            temperature = float(self.query_one("#gen-temp", Input).value.strip() or "1.0")
            seed = int(seed_raw) if seed_raw else None
        except ValueError as exc:
            status.update(f"[red]Invalid input: {exc}[/]")
            return

        if kind == "generate_protein":
            spec = build_generate_protein_spec(
                output=output, length=length, num_candidates=num_candidates,
                top_k=top_k, seed=seed, latent_scale=temperature,
            )
        else:
            spec = build_generate_plasmid_spec(
                output=output, length=length, num_candidates=num_candidates,
                top_k=top_k, seed=seed, temperature=temperature,
            )

        self.app.state.set_draft_job_spec({
            "kind": kind, "output": output, "length": length,
            "num_candidates": num_candidates, "top_k": top_k,
            "seed": seed, "temperature": temperature,
        })
        job_id = self.app.submit_job_spec(spec, title=f"{kind} {output}")
        status.update(f"[green]Submitted {job_id}[/]")

    def _refresh_jobs(self) -> None:
        jobs = self.current_jobs()
        gen_jobs = [j for j in jobs if j.kind.startswith("generate")]

        # Progress for active gen job
        bar = self.query_one("#gen-progress", ProgressBar)
        info = self.query_one("#gen-active-info", Static)
        active = [j for j in gen_jobs if j.status == JobStatus.BUSY]
        if active:
            job = active[0]
            pct = getattr(getattr(job, "progress", None), "percent", None)
            bar.update(progress=min(100, int(pct * 100)) if isinstance(pct, (int, float)) else 0)
            info.update(f"  {job.title}  |  {job.current_stage or '-'}  |  {job.message or ''}")
        else:
            bar.update(progress=0)
            info.update("  [dim]No active generation jobs[/]")

        # Table
        table = self.query_one("#gen-jobs-table", DataTable)
        table.clear()
        for job in gen_jobs[:10]:
            sts = {
                JobStatus.BUSY: "[cyan]RUN[/]",
                JobStatus.HEALTHY: "[green]OK[/]",
                JobStatus.FAILED: "[red]ERR[/]",
            }.get(job.status, str(job.status.value))
            table.add_row(
                sts, job.id[:20], job.kind,
                getattr(job, "current_stage", "") or "-",
                str(len(getattr(job, "artifacts", []))),
            )
