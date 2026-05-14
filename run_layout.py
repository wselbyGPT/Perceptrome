from __future__ import annotations

import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

from perceptrome.jobs.artifact_index import append_artifact_entries


CANONICAL_MANIFEST_SECTIONS = (
    "generated_sequences",
    "validation_results",
    "training_metrics",
    "pretraining_metrics",
    "checkpoints",
    "evolution_history",
)


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    root: str
    inputs_dir: str
    artifacts_dir: str
    metrics_dir: str
    outputs_dir: str
    provenance_dir: str
    manifest_path: str


def _new_run_id(prefix: str = "run") -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_run_layout(run_id: Optional[str] = None, base_dir: str = "runs") -> RunLayout:
    env_id = os.environ.get("PERCEPTROME_RUN_ID")
    run_id = str(run_id or env_id or _new_run_id())
    root = os.path.join(base_dir, run_id)
    layout = RunLayout(
        run_id=run_id,
        root=root,
        inputs_dir=os.path.join(root, "inputs"),
        artifacts_dir=os.path.join(root, "artifacts"),
        metrics_dir=os.path.join(root, "metrics"),
        outputs_dir=os.path.join(root, "outputs"),
        provenance_dir=os.path.join(root, "provenance"),
        manifest_path=os.path.join(root, "manifest.json"),
    )
    for d in (layout.root, layout.inputs_dir, layout.artifacts_dir, layout.metrics_dir, layout.outputs_dir, layout.provenance_dir):
        os.makedirs(d, exist_ok=True)

    os.environ["PERCEPTROME_RUN_ID"] = layout.run_id
    os.environ["PERCEPTROME_RUN_ROOT"] = layout.root

    if not os.path.exists(layout.manifest_path):
        payload: Dict[str, Any] = {
            "run_id": layout.run_id,
            "created_at": _utc_now(),
            "layout": {
                "root": layout.root,
                "inputs": layout.inputs_dir,
                "artifacts": layout.artifacts_dir,
                "metrics": layout.metrics_dir,
                "outputs": layout.outputs_dir,
                "provenance": layout.provenance_dir,
            },
            "paths": {},
            "metrics": {},
            "provenance": {},
            "artifacts": [],
        }
        for key in CANONICAL_MANIFEST_SECTIONS:
            payload[key] = {}
        with open(layout.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    return layout


def path_in_run(layout: RunLayout, section: str, filename: str) -> str:
    section_map = {
        "inputs": layout.inputs_dir,
        "artifacts": layout.artifacts_dir,
        "metrics": layout.metrics_dir,
        "outputs": layout.outputs_dir,
        "provenance": layout.provenance_dir,
    }
    if section not in section_map:
        raise ValueError(f"Unknown run section: {section}")
    return os.path.join(section_map[section], filename)


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def update_run_manifest(
    layout: RunLayout,
    *,
    paths: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    generated_sequences: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    training_metrics: Optional[Dict[str, Any]] = None,
    pretraining_metrics: Optional[Dict[str, Any]] = None,
    checkpoints: Optional[Dict[str, Any]] = None,
    evolution_history: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Iterable[Dict[str, Any]]] = None,
) -> str:
    with open(layout.manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if paths:
        _deep_merge(payload.setdefault("paths", {}), paths)
    if metrics:
        _deep_merge(payload.setdefault("metrics", {}), metrics)
    if provenance:
        _deep_merge(payload.setdefault("provenance", {}), provenance)
    if generated_sequences:
        _deep_merge(payload.setdefault("generated_sequences", {}), generated_sequences)
    if validation_results:
        _deep_merge(payload.setdefault("validation_results", {}), validation_results)
    if training_metrics:
        _deep_merge(payload.setdefault("training_metrics", {}), training_metrics)
    if pretraining_metrics:
        _deep_merge(payload.setdefault("pretraining_metrics", {}), pretraining_metrics)
    if checkpoints:
        _deep_merge(payload.setdefault("checkpoints", {}), checkpoints)
    if evolution_history:
        _deep_merge(payload.setdefault("evolution_history", {}), evolution_history)
    if artifacts:
        append_artifact_entries(payload, artifacts)
    payload["updated_at"] = _utc_now()

    with open(layout.manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return layout.manifest_path


def layout_as_dict(layout: RunLayout) -> Dict[str, str]:
    d = asdict(layout)
    return {k: str(v) for k, v in d.items()}
