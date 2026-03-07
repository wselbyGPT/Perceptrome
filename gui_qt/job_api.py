from __future__ import annotations

from perceptrome.jobs import JobSpec


def build_stream_spec(config_path: str, catalog: str, model_type: str, steps_per_plasmid: int, batch_size: int) -> JobSpec:
    return JobSpec(
        kind="stream",
        config_path=config_path,
        params={
            "catalog": catalog,
            "model_type": model_type,
            "steps_per_plasmid": int(steps_per_plasmid),
            "batch_size": int(batch_size),
        },
    )


def build_generate_plasmid_spec(config_path: str, length_bp: int, output: str) -> JobSpec:
    return JobSpec(
        kind="generate_plasmid",
        config_path=config_path,
        params={"length_bp": int(length_bp), "output": output},
    )
