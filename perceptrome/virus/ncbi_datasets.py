from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence


ENV_DATASETS_BIN = "PERCEPTROME_DATASETS_BIN"


class DatasetsBinaryNotFoundError(FileNotFoundError):
    """Raised when the NCBI datasets executable cannot be located."""


class DatasetsCommandError(RuntimeError):
    """Raised when `datasets` exits with a non-zero return code."""


class DatasetsJSONDecodeError(ValueError):
    """Raised when JSON output from `datasets` cannot be parsed."""


@dataclass(frozen=True)
class DatasetsCommandResult:
    argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    parsed_json: Any | None = None


def resolve_datasets_binary(datasets_bin: str | None) -> str:
    """Resolve the `datasets` executable using CLI arg, env var, then PATH."""
    if datasets_bin:
        if os.path.isfile(datasets_bin) and os.access(datasets_bin, os.X_OK):
            return datasets_bin
        raise DatasetsBinaryNotFoundError(f"datasets binary is not executable: {datasets_bin}")

    env_bin = os.environ.get(ENV_DATASETS_BIN)
    if env_bin:
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin
        raise DatasetsBinaryNotFoundError(
            f"Environment variable {ENV_DATASETS_BIN} is set but not executable: {env_bin}"
        )

    discovered = shutil.which("datasets")
    if discovered:
        return discovered

    raise DatasetsBinaryNotFoundError(
        "Unable to locate `datasets` executable. Provide datasets_bin, set "
        f"{ENV_DATASETS_BIN}, or add `datasets` to PATH."
    )


def _run_datasets(argv: list[str], *, parse_json: bool) -> DatasetsCommandResult:
    proc = subprocess.run(argv, check=False, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise DatasetsCommandError(
            f"datasets command failed with return code {proc.returncode}: {' '.join(argv)}\n{stderr.strip()}"
        )

    parsed: Any | None = None
    if parse_json:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise DatasetsJSONDecodeError(
                f"datasets output was not valid JSON for command {' '.join(argv)}"
            ) from exc

    return DatasetsCommandResult(
        argv=tuple(argv),
        return_code=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        parsed_json=parsed,
    )


def summary_virus_genome_by_taxon(
    *,
    taxon: str,
    datasets_bin: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> DatasetsCommandResult:
    resolved_bin = resolve_datasets_binary(datasets_bin)
    argv = [resolved_bin, "summary", "virus", "genome", "taxon", str(taxon), "--as-json"]
    if extra_args:
        argv.extend(str(arg) for arg in extra_args)
    return _run_datasets(argv, parse_json=True)


def summary_virus_genome_by_accession(
    *,
    accession: str,
    datasets_bin: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> DatasetsCommandResult:
    resolved_bin = resolve_datasets_binary(datasets_bin)
    argv = [resolved_bin, "summary", "virus", "genome", "accession", str(accession), "--as-json"]
    if extra_args:
        argv.extend(str(arg) for arg in extra_args)
    return _run_datasets(argv, parse_json=True)


def download_virus_genome_by_taxon(
    *,
    taxon: str,
    output_path: str,
    datasets_bin: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> DatasetsCommandResult:
    resolved_bin = resolve_datasets_binary(datasets_bin)
    argv = [
        resolved_bin,
        "download",
        "virus",
        "genome",
        "taxon",
        str(taxon),
        "--filename",
        str(output_path),
    ]
    if extra_args:
        argv.extend(str(arg) for arg in extra_args)
    return _run_datasets(argv, parse_json=False)


def download_virus_genome_by_accession(
    *,
    accession: str,
    output_path: str,
    datasets_bin: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> DatasetsCommandResult:
    resolved_bin = resolve_datasets_binary(datasets_bin)
    argv = [
        resolved_bin,
        "download",
        "virus",
        "genome",
        "accession",
        str(accession),
        "--filename",
        str(output_path),
    ]
    if extra_args:
        argv.extend(str(arg) for arg in extra_args)
    return _run_datasets(argv, parse_json=False)
