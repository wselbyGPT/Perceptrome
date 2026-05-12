from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ModelVersion, ModelVersionArtifact, RegisteredModel, Run, User
from ..schemas import ModelArtifactOut, ModelRegistrySummaryOut, ModelVersionOut, RegisteredModelOut
from . import audit_service, run_service

MODEL_VISIBILITIES = {"private", "team", "public"}
MODEL_STATUSES = {"active", "archived", "deprecated"}
VERSION_STATUSES = {"candidate", "stable", "deprecated", "archived"}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_obj(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _clean_tags(tags: list[str] | None) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for tag in tags or []:
        value = str(tag).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _validate_choice(value: str, allowed: set[str], field: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported {field}: {value}")
    return normalized


def model_artifact_download_url(model_id: str, version_id: str, artifact_id: int) -> str:
    return f"/api/models/{model_id}/versions/{version_id}/artifacts/{artifact_id}/download"


def can_view_model(model: RegisteredModel, user: User) -> bool:
    return user.role == "admin" or model.owner_user_id == user.id or model.visibility in {"team", "public"}


def assert_model_view(model: RegisteredModel, user: User) -> None:
    if not can_view_model(model, user):
        raise HTTPException(status_code=403, detail="Forbidden")


def assert_model_manage(model: RegisteredModel, user: User) -> None:
    if user.role != "admin" and model.owner_user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")


def get_model(db: Session, model_id: str) -> RegisteredModel | None:
    return db.get(RegisteredModel, model_id)


def artifact_to_out(model_id: str, version_id: str, artifact: ModelVersionArtifact) -> ModelArtifactOut:
    return ModelArtifactOut(
        id=artifact.id,
        role=artifact.role,
        path=artifact.path,
        label=artifact.label,
        download_url=model_artifact_download_url(model_id, version_id, artifact.id),
        created_at=artifact.created_at,
    )


def version_to_out(version: ModelVersion) -> ModelVersionOut:
    return ModelVersionOut(
        id=version.id,
        model_id=version.model_id,
        source_run_id=version.source_run_id,
        version_label=version.version_label,
        status=version.status,
        architecture=version.architecture,
        tokenizer=version.tokenizer,
        checkpoint_path=version.checkpoint_path,
        config_snapshot_path=version.config_snapshot_path,
        manifest_path=version.manifest_path,
        metrics=_json_obj(version.metrics_json),
        metadata=_json_obj(version.metadata_json),
        created_at=version.created_at,
        promoted_at=version.promoted_at,
        artifacts=[artifact_to_out(version.model_id, version.id, artifact) for artifact in sorted(version.artifacts, key=lambda item: item.created_at)],
    )


def model_to_out(model: RegisteredModel) -> RegisteredModelOut:
    versions = sorted(model.versions, key=lambda item: item.created_at, reverse=True)
    version_out = [version_to_out(version) for version in versions]
    current = next((item for item in version_out if item.id == model.current_version_id), None)
    if current is None and version_out:
        current = version_out[0]
    return RegisteredModelOut(
        id=model.id,
        owner_user_id=model.owner_user_id,
        name=model.name,
        description=model.description,
        visibility=model.visibility,
        status=model.status,
        tags=_json_list(model.tags_json),
        current_version_id=model.current_version_id,
        created_at=model.created_at,
        updated_at=model.updated_at,
        versions=version_out,
        current_version=current,
    )


def list_models(
    db: Session,
    user: User,
    *,
    search: str | None = None,
    architecture: str | None = None,
    status: str | None = None,
    visibility: str | None = None,
    limit: int = 100,
) -> list[RegisteredModelOut]:
    rows = db.execute(select(RegisteredModel).order_by(RegisteredModel.updated_at.desc()).limit(max(1, min(limit, 500)))).scalars().all()
    needle = str(search or "").strip().lower()
    arch = str(architecture or "").strip().lower()
    stat = str(status or "").strip().lower()
    vis = str(visibility or "").strip().lower()
    out: list[RegisteredModelOut] = []
    for model in rows:
        if not can_view_model(model, user):
            continue
        dto = model_to_out(model)
        text = " ".join([dto.name, dto.description or "", " ".join(dto.tags)]).lower()
        if needle and needle not in text:
            continue
        if stat and dto.status != stat:
            continue
        if vis and dto.visibility != vis:
            continue
        if arch and not any((version.architecture or "").lower() == arch for version in dto.versions):
            continue
        out.append(dto)
    return out


def registry_summary(db: Session, user: User) -> ModelRegistrySummaryOut:
    models = list_models(db, user, limit=500)
    architecture_counts: Counter[str] = Counter()
    tokenizer_counts: Counter[str] = Counter()
    version_count = 0
    for model in models:
        for version in model.versions:
            version_count += 1
            if version.architecture:
                architecture_counts[version.architecture] += 1
            if version.tokenizer:
                tokenizer_counts[version.tokenizer] += 1
    return ModelRegistrySummaryOut(
        total_models=len(models),
        total_versions=version_count,
        architecture_counts=dict(architecture_counts),
        tokenizer_counts=dict(tokenizer_counts),
    )


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _config_params(run: Run) -> dict[str, Any]:
    cfg = run_service.json_loads(run.config_json)
    params = cfg.get("params")
    if isinstance(params, dict):
        merged = {k: v for k, v in cfg.items() if k != "params"}
        merged.update(params)
        return merged
    return cfg


def _extract_config_snapshot_path(result_payload: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    snap = result_payload.get("config_snapshot")
    if isinstance(snap, dict):
        value = snap.get("path")
        if isinstance(value, str) and value:
            return value
    provenance = manifest.get("provenance_metadata") if isinstance(manifest.get("provenance_metadata"), dict) else {}
    snap = provenance.get("config_snapshot") if isinstance(provenance.get("config_snapshot"), dict) else None
    if snap and isinstance(snap.get("path"), str):
        return str(snap["path"])
    provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    snap = provenance.get("config_snapshot") if isinstance(provenance.get("config_snapshot"), dict) else None
    if snap and isinstance(snap.get("path"), str):
        return str(snap["path"])
    return None


def _find_checkpoint_path(run: Run, result_payload: dict[str, Any], manifest: dict[str, Any], config_snapshot_path: str | None) -> str | None:
    for value in (result_payload.get("checkpoint_path"), result_payload.get("checkpoint"), result_payload.get("model_checkpoint")):
        if isinstance(value, str) and value:
            return value

    checkpoints = manifest.get("checkpoints")
    if isinstance(checkpoints, dict):
        for key in ("latest", "path", "checkpoint", "model", "index_json"):
            value = checkpoints.get(key)
            if isinstance(value, str) and value:
                return value

    for artifact in manifest.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        role = str(artifact.get("role") or artifact.get("type") or artifact.get("id") or "").lower()
        path = artifact.get("path")
        if isinstance(path, str) and path and ("checkpoint" in role or path.endswith((".pt", ".pth", ".ckpt"))):
            return path

    for artifact in run.artifacts:
        text = " ".join([artifact.phase or "", artifact.label or "", artifact.path]).lower()
        if "checkpoint" in text or artifact.path.endswith((".pt", ".pth", ".ckpt")):
            return artifact.path

    if config_snapshot_path:
        candidate = Path(config_snapshot_path).expanduser().resolve().parent / "checkpoints" / "latest.pt"
        if candidate.exists():
            return str(candidate)
    return None


def _add_artifact_if_file(
    *,
    version: ModelVersion,
    role: str,
    path: str | None,
    label: str,
    seen: set[str],
) -> None:
    if not path:
        return
    resolved = Path(path).expanduser().resolve()
    if str(resolved) in seen or not resolved.exists() or not resolved.is_file():
        return
    seen.add(str(resolved))
    version.artifacts.append(ModelVersionArtifact(role=role, path=str(resolved), label=label))


def _version_payload_from_run(run: Run) -> dict[str, Any]:
    result_payload = run_service.json_loads(run.result_json)
    manifest_path = _first_str(result_payload.get("manifest_path"), run_service.manifest_path_for_run(run))
    manifest = run_service.load_manifest(manifest_path)
    params = _config_params(run)
    model_cfg = manifest.get("model_objective_config") if isinstance(manifest.get("model_objective_config"), dict) else {}
    tokenizer_cfg = manifest.get("tokenizer_encoding_config") if isinstance(manifest.get("tokenizer_encoding_config"), dict) else {}
    config_snapshot_path = _extract_config_snapshot_path(result_payload, manifest)
    checkpoint_path = _find_checkpoint_path(run, result_payload, manifest, config_snapshot_path)

    architecture = _first_str(model_cfg.get("model_type"), params.get("model_type"), params.get("model_family"))
    tokenizer = _first_str(tokenizer_cfg.get("tokenizer"), params.get("tokenizer"))
    metrics = {
        "metrics": manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {},
        "training_metrics": manifest.get("training_metrics") if isinstance(manifest.get("training_metrics"), dict) else {},
        "pretraining_metrics": manifest.get("pretraining_metrics") if isinstance(manifest.get("pretraining_metrics"), dict) else {},
        "result": {
            key: value
            for key, value in result_payload.items()
            if key in {"length", "last_total_loss", "processed_accessions", "scorecard_path", "summary_path", "output"}
        },
    }
    metadata = {
        "source_run": {
            "run_id": run.run_id,
            "kind": run.kind,
            "state": run.state,
            "submitted_at": run.submitted_at.isoformat() if run.submitted_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        },
        "model_objective_config": model_cfg,
        "tokenizer_encoding_config": tokenizer_cfg,
        "config": run_service.json_loads(run.config_json),
    }
    return {
        "architecture": architecture,
        "tokenizer": tokenizer,
        "checkpoint_path": checkpoint_path,
        "config_snapshot_path": config_snapshot_path,
        "manifest_path": manifest_path,
        "metrics": metrics,
        "metadata": metadata,
    }


def register_from_run(
    db: Session,
    user: User,
    *,
    run_id: str,
    model_id: str | None,
    name: str | None,
    description: str | None,
    visibility: str | None,
    tags: list[str] | None,
    version_label: str | None,
    version_status: str,
) -> RegisteredModelOut:
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run_service.assert_run_access(run, user)
    if run.state != "completed":
        raise HTTPException(status_code=409, detail="Only completed runs can be registered as model versions")

    version_status_value = _validate_choice(version_status, VERSION_STATUSES, "version_status")
    now = audit_service.utcnow()

    if model_id:
        model = get_model(db, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        assert_model_manage(model, user)
    else:
        visibility_value = _validate_choice(visibility or "private", MODEL_VISIBILITIES, "visibility")
        model = RegisteredModel(
            owner_user_id=user.id,
            name=(name or f"{run.kind} {run.run_id}").strip()[:160],
            description=description,
            visibility=visibility_value,
            status="active",
            tags_json=_json_dumps(_clean_tags(tags)),
        )
        db.add(model)
        db.flush()

    if name and model_id:
        model.name = name.strip()[:160]
    if description is not None and model_id:
        model.description = description
    if tags is not None and model_id:
        model.tags_json = _json_dumps(_clean_tags(tags))
    if visibility is not None and model_id:
        model.visibility = _validate_choice(visibility, MODEL_VISIBILITIES, "visibility")

    payload = _version_payload_from_run(run)
    version = ModelVersion(
        model_id=model.id,
        source_run_id=run.run_id,
        version_label=(version_label or f"v{len(model.versions) + 1}").strip()[:80],
        status=version_status_value,
        architecture=payload["architecture"],
        tokenizer=payload["tokenizer"],
        checkpoint_path=payload["checkpoint_path"],
        config_snapshot_path=payload["config_snapshot_path"],
        manifest_path=payload["manifest_path"],
        metrics_json=_json_dumps(payload["metrics"]),
        metadata_json=_json_dumps(payload["metadata"]),
        promoted_at=now if version_status_value == "stable" else None,
    )
    db.add(version)
    db.flush()

    seen: set[str] = set()
    _add_artifact_if_file(version=version, role="checkpoint", path=version.checkpoint_path, label="Model checkpoint", seen=seen)
    _add_artifact_if_file(version=version, role="manifest", path=version.manifest_path, label="Run manifest", seen=seen)
    _add_artifact_if_file(version=version, role="config_snapshot", path=version.config_snapshot_path, label="Resolved config", seen=seen)
    for artifact in run.artifacts:
        _add_artifact_if_file(version=version, role=artifact.phase or "run_artifact", path=artifact.path, label=artifact.label or "Run artifact", seen=seen)

    model.current_version_id = version.id
    model.updated_at = now
    db.commit()
    db.refresh(model)
    return model_to_out(model)


def update_model(db: Session, user: User, model_id: str, payload: dict[str, Any]) -> RegisteredModelOut:
    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    assert_model_manage(model, user)

    if payload.get("name") is not None:
        name = str(payload["name"]).strip()
        if not name:
            raise HTTPException(status_code=400, detail="Model name is required")
        model.name = name[:160]
    if "description" in payload:
        model.description = None if payload["description"] is None else str(payload["description"])
    if payload.get("visibility") is not None:
        model.visibility = _validate_choice(str(payload["visibility"]), MODEL_VISIBILITIES, "visibility")
    if payload.get("status") is not None:
        model.status = _validate_choice(str(payload["status"]), MODEL_STATUSES, "status")
    if payload.get("tags") is not None:
        raw_tags = payload["tags"]
        if not isinstance(raw_tags, list):
            raise HTTPException(status_code=400, detail="tags must be a list")
        model.tags_json = _json_dumps(_clean_tags([str(item) for item in raw_tags]))
    if payload.get("current_version_id") is not None:
        version_id = str(payload["current_version_id"])
        if not any(version.id == version_id for version in model.versions):
            raise HTTPException(status_code=400, detail="current_version_id does not belong to this model")
        model.current_version_id = version_id
    model.updated_at = audit_service.utcnow()
    db.commit()
    db.refresh(model)
    return model_to_out(model)


def update_version(db: Session, user: User, model_id: str, version_id: str, payload: dict[str, Any]) -> RegisteredModelOut:
    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    assert_model_manage(model, user)
    version = next((item for item in model.versions if item.id == version_id), None)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")

    if payload.get("version_label") is not None:
        label = str(payload["version_label"]).strip()
        if not label:
            raise HTTPException(status_code=400, detail="version_label is required")
        version.version_label = label[:80]
    if payload.get("status") is not None:
        version.status = _validate_choice(str(payload["status"]), VERSION_STATUSES, "status")
    if bool(payload.get("promote_current", False)):
        model.current_version_id = version.id
        version.status = "stable"
        version.promoted_at = audit_service.utcnow()
    model.updated_at = audit_service.utcnow()
    db.commit()
    db.refresh(model)
    return model_to_out(model)


def download_model_artifact(db: Session, user: User, model_id: str, version_id: str, artifact_id: int):
    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    assert_model_view(model, user)
    version = next((item for item in model.versions if item.id == version_id), None)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    artifact = next((item for item in version.artifacts if item.id == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    target = Path(artifact.path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)
