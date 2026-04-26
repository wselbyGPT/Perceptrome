from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from perceptrome.encoding.parse import parse_fasta_sequence


ENV_ALPHAFOLD3_BIN = "PERCEPTROME_ALPHAFOLD3_BIN"
ENV_ALPHAFOLD3_MODEL_DIR = "PERCEPTROME_ALPHAFOLD3_MODEL_DIR"
ENV_ALPHAFOLD3_DB_DIR = "PERCEPTROME_ALPHAFOLD3_DB_DIR"


_JOB_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class AlphaFold3RunResult:
    return_code: int
    command: List[str]
    stdout_log_path: str
    stderr_log_path: str
    json_input_path: str
    job_name: str


def sanitize_job_name(name: str) -> str:
    cleaned = _JOB_NAME_SAFE.sub("_", str(name)).strip("_")
    return cleaned or "job"


def resolve_alphafold3_binary(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        if os.path.exists(explicit_path) and (os.path.isfile(explicit_path) or os.access(explicit_path, os.X_OK)):
            return explicit_path
        raise FileNotFoundError(f"AlphaFold 3 entrypoint not found: {explicit_path}")

    env_bin = os.environ.get(ENV_ALPHAFOLD3_BIN)
    if env_bin:
        if os.path.exists(env_bin):
            return env_bin
        raise FileNotFoundError(
            f"Environment variable {ENV_ALPHAFOLD3_BIN} is set but target does not exist: {env_bin}"
        )

    for candidate in ("run_alphafold.py", "run_alphafold", "alphafold3"):
        path = shutil.which(candidate)
        if path:
            return path

    raise FileNotFoundError(
        "Unable to locate AlphaFold 3 entrypoint. Provide --alphafold3-bin, set "
        f"{ENV_ALPHAFOLD3_BIN}, or add run_alphafold.py to PATH."
    )


def resolve_alphafold3_model_dir(explicit_path: Optional[str]) -> str:
    if explicit_path:
        if os.path.isdir(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"AlphaFold 3 model dir not found: {explicit_path}")
    env_dir = os.environ.get(ENV_ALPHAFOLD3_MODEL_DIR)
    if env_dir:
        if os.path.isdir(env_dir):
            return env_dir
        raise FileNotFoundError(
            f"{ENV_ALPHAFOLD3_MODEL_DIR} is set but directory does not exist: {env_dir}"
        )
    raise FileNotFoundError(
        "AlphaFold 3 model dir not provided. Use --alphafold3-model-dir or set "
        f"{ENV_ALPHAFOLD3_MODEL_DIR}."
    )


def resolve_alphafold3_db_dir(explicit_path: Optional[str]) -> str:
    if explicit_path:
        if os.path.isdir(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"AlphaFold 3 db dir not found: {explicit_path}")
    env_dir = os.environ.get(ENV_ALPHAFOLD3_DB_DIR)
    if env_dir:
        if os.path.isdir(env_dir):
            return env_dir
        raise FileNotFoundError(
            f"{ENV_ALPHAFOLD3_DB_DIR} is set but directory does not exist: {env_dir}"
        )
    raise FileNotFoundError(
        "AlphaFold 3 database dir not provided. Use --alphafold3-db-dir or set "
        f"{ENV_ALPHAFOLD3_DB_DIR}."
    )


def build_alphafold3_protein_job(
    *,
    fasta_path: str,
    job_name: str,
    model_seeds: Optional[List[int]] = None,
) -> dict:
    sequence = parse_fasta_sequence(fasta_path)
    seeds = list(model_seeds) if model_seeds else [1]
    return {
        "name": sanitize_job_name(job_name),
        "modelSeeds": [int(s) for s in seeds],
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
        ],
        "dialect": "alphafold3",
        "version": 1,
    }


def write_alphafold3_input_json(payload: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def run_alphafold3_monomer(
    *,
    fasta_path: str,
    output_dir: str,
    alphafold3_bin: str,
    model_dir: str,
    db_dir: str,
    stdout_log_path: str,
    stderr_log_path: str,
    job_name: Optional[str] = None,
    num_seeds: int = 1,
    num_diffusion_samples: int = 5,
    json_input_path: Optional[str] = None,
) -> AlphaFold3RunResult:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(stdout_log_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(stderr_log_path) or ".", exist_ok=True)

    resolved_job = sanitize_job_name(job_name or os.path.splitext(os.path.basename(fasta_path))[0])
    seeds = list(range(1, max(int(num_seeds), 1) + 1))
    payload = build_alphafold3_protein_job(
        fasta_path=fasta_path,
        job_name=resolved_job,
        model_seeds=seeds,
    )

    input_path = json_input_path or os.path.join(output_dir, f"{resolved_job}.input.json")
    write_alphafold3_input_json(payload, input_path)

    entrypoint = str(alphafold3_bin)
    cmd: List[str]
    if entrypoint.endswith(".py"):
        cmd = ["python", entrypoint]
    else:
        cmd = [entrypoint]
    cmd += [
        f"--json_path={input_path}",
        f"--output_dir={output_dir}",
        f"--model_dir={model_dir}",
        f"--db_dir={db_dir}",
        f"--num_diffusion_samples={int(num_diffusion_samples)}",
    ]

    with open(stdout_log_path, "w", encoding="utf-8") as out, open(stderr_log_path, "w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, check=False, stdout=out, stderr=err)

    return AlphaFold3RunResult(
        return_code=int(proc.returncode),
        command=cmd,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        json_input_path=input_path,
        job_name=resolved_job,
    )
