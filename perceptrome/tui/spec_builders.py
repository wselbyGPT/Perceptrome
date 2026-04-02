from __future__ import annotations

from typing import Any

from perceptrome.jobs.engine import JobSpec


class SpecValidationError(ValueError):
    """Raised when a job form cannot be converted into a valid JobSpec."""


def _required(params: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if params.get(key) in (None, "")]
    if missing:
        raise SpecValidationError(f"Missing required parameters: {', '.join(missing)}")


def _spec(kind: str, config_path: str, **params: Any) -> JobSpec:
    return JobSpec(kind=kind, config_path=config_path or "config/stream_config.yaml", params=params)


def build_train_one_spec(*, accession: str, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"accession": accession}, "accession")
    return _spec("train_one", config_path, accession=str(accession), **params)


def build_stream_spec(*, catalog: str, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"catalog": catalog}, "catalog")
    return _spec("stream", config_path, catalog=str(catalog), **params)


def build_generate_plasmid_spec(*, output: str, length: int, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"output": output, "length": length}, "output", "length")
    return _spec("generate_plasmid", config_path, output=str(output), length=int(length), **params)


def build_generate_protein_spec(*, output: str, length: int, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"output": output, "length": length}, "output", "length")
    return _spec("generate_protein", config_path, output=str(output), length=int(length), **params)


def build_validate_plasmid_spec(*, accession: str, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"accession": accession}, "accession")
    return _spec("validate_plasmid", config_path, accession=str(accession), **params)


def build_pretrain_spec(*, dataset: str, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"dataset": dataset}, "dataset")
    return _spec("pretrain", config_path, dataset=str(dataset), **params)


def build_design_loop_spec(*, seed_sequence: str, config_path: str = "config/stream_config.yaml", **params: Any) -> JobSpec:
    _required({"seed_sequence": seed_sequence}, "seed_sequence")
    return _spec("design_loop", config_path, seed_sequence=str(seed_sequence), **params)
