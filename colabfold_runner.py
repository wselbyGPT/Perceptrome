from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


ENV_COLABFOLD_BIN = "PERCEPTROME_COLABFOLD_BIN"


@dataclass(frozen=True)
class ColabFoldRunResult:
    return_code: int
    command: List[str]
    stdout_log_path: str
    stderr_log_path: str


def resolve_colabfold_binary(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        if os.path.exists(explicit_path) and os.access(explicit_path, os.X_OK):
            return explicit_path
        raise FileNotFoundError(f"ColabFold binary not executable: {explicit_path}")

    env_bin = os.environ.get(ENV_COLABFOLD_BIN)
    if env_bin:
        if os.path.exists(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin
        raise FileNotFoundError(
            f"Environment variable {ENV_COLABFOLD_BIN} is set but target is not executable: {env_bin}"
        )

    for candidate in ("colabfold_batch", "colabfold_batch.sh"):
        path = shutil.which(candidate)
        if path:
            return path

    raise FileNotFoundError(
        "Unable to locate ColabFold executable. Provide --colabfold-bin, set "
        f"{ENV_COLABFOLD_BIN}, or add colabfold_batch to PATH."
    )


def run_colabfold_monomer(
    *,
    fasta_path: str,
    output_dir: str,
    num_recycle: int,
    num_models: int,
    colabfold_bin: str,
    stdout_log_path: str,
    stderr_log_path: str,
) -> ColabFoldRunResult:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(stdout_log_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(stderr_log_path) or ".", exist_ok=True)

    cmd = [
        str(colabfold_bin),
        "--num-recycle",
        str(int(num_recycle)),
        "--num-models",
        str(int(num_models)),
        str(fasta_path),
        str(output_dir),
    ]

    with open(stdout_log_path, "w", encoding="utf-8") as out, open(stderr_log_path, "w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, check=False, stdout=out, stderr=err)

    return ColabFoldRunResult(
        return_code=int(proc.returncode),
        command=cmd,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )
