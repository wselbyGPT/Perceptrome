from __future__ import annotations

from textual.widgets import Static

from perceptrome.tui.job_manager import JobStatus

from .base import BasePanel


class OverviewPanel(BasePanel):
    PANEL_ID = "overview"
    TITLE = "Overview"

    def compose(self):
        yield Static(id="overview-body")

    def on_mount(self) -> None:
        super().on_mount()
        self._render_overview()

    def handle_tui_event(self, event: object) -> None:
        if hasattr(event, "job_id"):
            self.schedule_throttled_render("overview", self._render_overview)

    def _render_overview(self) -> None:
        body = self.query_one("#overview-body", Static)
        jobs = self.current_jobs()
        if not jobs:
            body.update("No active runs yet. Start a job from the launcher to populate overview metrics.")
            return

        active_count = sum(1 for job in jobs if getattr(job, "status", None) == JobStatus.BUSY)
        failed_count = sum(1 for job in jobs if getattr(job, "status", None) == JobStatus.FAILED)

        latest = jobs[0]
        latest_artifact = "none"
        artifacts = list(getattr(latest, "artifacts", []) or [])
        if artifacts:
            latest_artifact = str(artifacts[-1])

        def _badge(status: object) -> str:
            if status == JobStatus.HEALTHY:
                return "✅ healthy"
            if status == JobStatus.BUSY:
                return "🟦 busy"
            if status == JobStatus.WARNING:
                return "🟨 warning"
            if status == JobStatus.STALLED:
                return "🟧 stalled"
            if status == JobStatus.FAILED:
                return "🟥 failed"
            return f"⬜ {status}"

        health = []
        for job in jobs[:5]:
            health.append(f"- {job.id}: {_badge(getattr(job, 'status', 'unknown'))}")

        body.update(
            "\n".join(
                [
                    "Dashboard Summary",
                    f"Active count: {active_count}",
                    f"Failed count: {failed_count}",
                    f"Latest artifact: {latest_artifact}",
                    "",
                    "Health badges",
                    *health,
                ]
            )
        )
