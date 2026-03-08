from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import uuid
from typing import Any, Dict, Optional

from perceptrome.jobs.manifest_schema import empty_run_manifest


def iso_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        sha = out.decode("utf-8").strip()
        return sha or None
    except Exception:
        return None


def config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def sidecar_manifest_path(path: str) -> str:
    return f"{path}.manifest.json"


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def build_run_manifest(
    *,
    run_kind: str,
    config_path: str,
    config_hash_value: Optional[str],
    run_id: Optional[str] = None,
    created_at: Optional[str] = None,
    dataset_catalog_manifest: Optional[Dict[str, Any]] = None,
    tokenizer_encoding_config: Optional[Dict[str, Any]] = None,
    model_objective_config: Optional[Dict[str, Any]] = None,
    checkpoints: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    training_metrics: Optional[Dict[str, Any]] = None,
    pretraining_metrics: Optional[Dict[str, Any]] = None,
    generated_sequences: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    evolution_history: Optional[Dict[str, Any]] = None,
    provenance_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = empty_run_manifest(
        run_kind=run_kind,
        config_path=config_path,
        config_hash=config_hash_value,
        run_id=run_id or new_run_id(run_kind),
        created_at=created_at or iso_now(),
        git_sha=get_git_sha(),
    )
    manifest["dataset_catalog_manifest"] = dataset_catalog_manifest
    manifest["tokenizer_encoding_config"] = tokenizer_encoding_config
    manifest["model_objective_config"] = model_objective_config
    manifest["checkpoints"] = checkpoints
    manifest["metrics"] = metrics
    manifest["training_metrics"] = training_metrics
    manifest["pretraining_metrics"] = pretraining_metrics
    manifest["generated_sequences"] = generated_sequences
    manifest["validation_results"] = validation_results
    manifest["evolution_history"] = evolution_history
    if provenance_metadata:
        merged = dict(manifest.get("provenance_metadata") or {})
        merged.update(provenance_metadata)
        manifest["provenance_metadata"] = merged
    return manifest


def write_sidecar_run_manifest(*, target_path: str, **kwargs: Any) -> str:
    payload = build_run_manifest(**kwargs)
    return _write_json(sidecar_manifest_path(target_path), payload)


def write_experiment_run_manifest(*, logs_dir: str, experiment_id: str, **kwargs: Any) -> str:
    payload = build_run_manifest(run_id=experiment_id, **kwargs)
    out_path = os.path.join(logs_dir, "experiments", f"{experiment_id}.json")
    return _write_json(out_path, payload)
