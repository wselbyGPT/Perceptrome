from __future__ import annotations

from types import SimpleNamespace

from perceptrome.tui.job_manager import JobStatus
from perceptrome.tui.launcher import derive_context, rank_commands
from perceptrome.tui.config_tools import apply_override, validate_effective


def _fake_job(status: JobStatus):
    return SimpleNamespace(status=status)


def test_rank_commands_running_prioritizes_stop_inspect_logs() -> None:
    context = derive_context(active_panel="overview", jobs=[_fake_job(JobStatus.BUSY)])
    ranked = rank_commands(context)
    top_ids = [row.command_id for row in ranked[:4]]
    assert "job.stop" in top_ids
    assert "inspect.logs" in top_ids


def test_rank_commands_failed_prioritizes_troubleshooting() -> None:
    context = derive_context(active_panel="overview", jobs=[_fake_job(JobStatus.FAILED)])
    ranked = rank_commands(context)
    top_ids = [row.command_id for row in ranked[:5]]
    assert "failure.troubleshoot" in top_ids
    assert "view.traceback" in top_ids


def test_rank_commands_idle_prioritizes_start_and_rerun() -> None:
    context = derive_context(active_panel="overview", jobs=[])
    ranked = rank_commands(context)
    top_ids = [row.command_id for row in ranked[:5]]
    assert "job.start" in top_ids


def test_config_override_and_validation() -> None:
    config = {
        "training": {"window_size": 9, "stride": 3, "batch_size": 16, "tokenizer": "codon"},
        "io": {"checkpoints_dir": "model/checkpoints"},
    }
    ok, _ = apply_override(config, "training.batch_size=64")
    assert ok
    assert config["training"]["batch_size"] == 64
    checks = validate_effective(config)
    assert all(not item.startswith("FAIL") for item in checks)


