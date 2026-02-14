import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import IOConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(io_cfg: IOConfig, run_id: str) -> str:
    return os.path.join(io_cfg.model_dir, run_id, "manifest.json")


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def file_fingerprint(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
    }
    if not out["exists"]:
        return out
    st = os.stat(path)
    out["size_bytes"] = int(st.st_size)
    out["mtime_epoch"] = float(st.st_mtime)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    out["sha256"] = h.hexdigest()
    return out


def dataset_list_checksum(accessions: List[str]) -> str:
    normalized = "\n".join(accessions).encode("utf-8")
    return _sha256_bytes(normalized)


def _git_commit_hash() -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return p.stdout.strip() or None
    except Exception:
        return None


def _environment_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def init_run_manifest(
    io_cfg: IOConfig,
    run_id: str,
    config_path: str,
    merged_config: Dict[str, Any],
    source_inputs: Dict[str, Any],
    dataset_accessions: List[str],
) -> str:
    run_dir = os.path.join(io_cfg.model_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = _manifest_path(io_cfg, run_id)
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "git_commit": _git_commit_hash(),
        "config": {
            "config_path": config_path,
            "effective_merged": merged_config,
        },
        "dataset": {
            "accessions": dataset_accessions,
            "list_sha256": dataset_list_checksum(dataset_accessions),
            "count": len(dataset_accessions),
        },
        "environment": _environment_metadata(),
        "source_inputs": source_inputs,
        "cache_fingerprints": {
            "encode": [],
            "train": [],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_manifest(io_cfg: IOConfig, run_id: str) -> Dict[str, Any]:
    path = _manifest_path(io_cfg, run_id)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(io_cfg: IOConfig, run_id: str, manifest: Dict[str, Any]) -> str:
    path = _manifest_path(io_cfg, run_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)
    return path


def record_source_accession(
    io_cfg: IOConfig,
    run_id: str,
    accession: str,
    source: str,
    source_path: Optional[str],
) -> None:
    m = load_manifest(io_cfg, run_id)
    entries = m.setdefault("source_inputs", {}).setdefault("accessions", [])
    row: Dict[str, Any] = {"accession": accession, "source": source}
    if source_path:
        row["source_file"] = file_fingerprint(source_path)
    entries.append(row)
    save_manifest(io_cfg, run_id, m)


def record_cache_fingerprint(
    io_cfg: IOConfig,
    run_id: str,
    stage: str,
    accession: str,
    cache_path: str,
) -> None:
    if stage not in ("encode", "train"):
        raise ValueError("stage must be encode|train")
    m = load_manifest(io_cfg, run_id)
    rec = {
        "at": _utc_now(),
        "accession": accession,
        "cache_file": file_fingerprint(cache_path),
    }
    m.setdefault("cache_fingerprints", {}).setdefault(stage, []).append(rec)
    save_manifest(io_cfg, run_id, m)
