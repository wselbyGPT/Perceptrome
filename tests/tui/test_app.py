from __future__ import annotations

from perceptrome.jobs.engine import JobSpec
from perceptrome.tui.app import PerceptromeTUIApp


def test_app_has_submission_helpers() -> None:
    app = PerceptromeTUIApp()
    assert hasattr(app, "submit_job_spec")
    assert hasattr(app, "rerun_selected_job")
    assert hasattr(app, "clone_selected_job_to_draft")
    assert hasattr(app, "cancel_selected_job")


def test_submit_job_spec_sets_selected(monkeypatch) -> None:
    app = PerceptromeTUIApp()

    def fake_submit(spec, **kwargs):
        _ = kwargs
        assert isinstance(spec, JobSpec)
        return "job-1"

    monkeypatch.setattr(app.jobs, "submit", fake_submit)
    app.submit_job_spec(JobSpec(kind="train_one", params={"accession": "NC_000913"}))
    assert app.state.get_session().selected_job_id == "job-1"
