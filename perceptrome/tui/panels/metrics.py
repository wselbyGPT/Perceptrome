from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Label, Rule, Sparkline, Static

from perceptrome.tui.events import JobMetricUpdatedEvent
from perceptrome.tui.job_manager import JobStatus

from .base import BasePanel


def _spark_text(values: list[float]) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(bars[min(len(bars) - 1, int((v - lo) / span * (len(bars) - 1)))] for v in values[-40:])


class MetricsPanel(BasePanel):
    PANEL_ID = "metrics"
    TITLE = "Metrics"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._losses: dict[str, list[float]] = {}

    def compose(self) -> ComposeResult:
        yield Static("[b]Training Metrics[/b]", classes="panel-title")

        # Primary sparkline for selected job
        yield Static("[b]Selected Job Loss Curve[/b]")
        yield Sparkline([], id="metrics-sparkline")
        with Horizontal(id="metrics-stats-row"):
            with Vertical(classes="metric-card"):
                yield Label("Current")
                yield Static("-", id="m-current")
            with Vertical(classes="metric-card"):
                yield Label("Min")
                yield Static("-", id="m-min")
            with Vertical(classes="metric-card"):
                yield Label("Max")
                yield Static("-", id="m-max")
            with Vertical(classes="metric-card"):
                yield Label("Rolling(10)")
                yield Static("-", id="m-rolling")
            with Vertical(classes="metric-card"):
                yield Label("Samples")
                yield Static("-", id="m-samples")

        yield Rule()

        # All jobs summary table
        yield Static("[b]All Jobs Loss Summary[/b]")
        yield DataTable(id="metrics-table")

        yield Rule()

        # Per-job text sparklines
        yield Static("[b]Per-Job Sparklines[/b]")
        yield Static("", id="metrics-all-sparks")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#metrics-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Job ID", "Status", "Current", "Min", "Max", "Rolling(10)", "Trend")
        self._render()

    def handle_tui_event(self, event: object) -> None:
        if isinstance(event, JobMetricUpdatedEvent):
            self._losses.setdefault(event.job_id, []).append(float(event.latest_value))
            self._losses[event.job_id] = self._losses[event.job_id][-80:]
            self.schedule_throttled_render("metrics", self._render)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        job_ids = list(self._losses.keys())
        if 0 <= event.cursor_row < len(job_ids):
            self.app.state.set_selected_job(job_ids[event.cursor_row])
            self._render()

    def _render(self) -> None:
        ctx = self.selected_job_context()
        selected = ctx.get("selected_job_id")
        vals = list(self._losses.get(selected, [])) if selected else []

        # Primary sparkline
        spark = self.query_one("#metrics-sparkline", Sparkline)
        spark.data = vals if vals else [0.0]

        # Stats cards
        if vals:
            rolling = sum(vals[-10:]) / min(10, len(vals))
            self.query_one("#m-current", Static).update(f"{vals[-1]:.6f}")
            self.query_one("#m-min", Static).update(f"{min(vals):.6f}")
            self.query_one("#m-max", Static).update(f"{max(vals):.6f}")
            self.query_one("#m-rolling", Static).update(f"{rolling:.6f}")
            self.query_one("#m-samples", Static).update(str(len(vals)))
        else:
            for wid in ("#m-current", "#m-min", "#m-max", "#m-rolling", "#m-samples"):
                self.query_one(wid, Static).update("-")

        # All-jobs table
        table = self.query_one("#metrics-table", DataTable)
        table.clear()
        jobs = self.current_jobs()
        job_status_map = {j.id: j.status for j in jobs}
        for jid, data in self._losses.items():
            if not data:
                continue
            rolling = sum(data[-10:]) / min(10, len(data))
            status = job_status_map.get(jid, JobStatus.HEALTHY)
            sts = {
                JobStatus.BUSY: "[cyan]RUN[/]",
                JobStatus.HEALTHY: "[green]OK[/]",
                JobStatus.FAILED: "[red]ERR[/]",
            }.get(status, str(status.value) if hasattr(status, "value") else str(status))
            table.add_row(
                jid[:20], sts,
                f"{data[-1]:.5f}", f"{min(data):.5f}", f"{max(data):.5f}",
                f"{rolling:.5f}", _spark_text(data),
            )

        # Per-job text sparklines
        lines: list[str] = []
        for jid, data in self._losses.items():
            if not data:
                continue
            marker = " > " if jid == selected else "   "
            lines.append(f"{marker}{jid[:16]}  {_spark_text(data)}  cur={data[-1]:.5f}")
        self.query_one("#metrics-all-sparks", Static).update(
            "\n".join(lines) if lines else "[dim]No metric data yet[/]"
        )
