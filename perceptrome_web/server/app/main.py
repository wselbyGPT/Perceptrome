# server/app/main.py
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import logging
import smtplib
import threading
from dataclasses import dataclass, field
from collections import Counter, deque
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from .config import settings
from .auth_rate_limit import login_attempt_store
from .db import Base, engine, SessionLocal
from .deps import get_db, get_current_user, get_current_user_strict, require_role
from .models import AuthToken, Run, RunArtifact, User, UserSession
from .schemas import (
    RegisterRequest,
    LoginRequest,
    VerifyEmailRequest,
    ResendVerificationRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    ChangePasswordRequest,
    AdminCreateUserRequest,
    UserOut,
    MessageOut,
    RunArtifactOut,
    RunOut,
    RunStartRequest,
    LineageNodeOut,
    LineageEdgeOut,
    RunLineageOut,
    DatasetCatalogItemOut,
    DatasetDetailOut,
    DatasetPreviewOut,
    DatasetSplitOut,
    RunSummaryOut,
    RunsBoardOut,
)
from perceptrome.jobs import JobEngine, JobEvent, JobSpec

from .security import (
    hash_password,
    verify_password,
    make_session_token,
    hash_session_token,
    password_complexity_error,
)

app = FastAPI(title=settings.app_name)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_auth_logger = logging.getLogger("perceptrome.auth")
_auth_metrics: Counter[str] = Counter()

RunState = Literal["queued", "running", "completed", "failed", "canceled"]


@dataclass(slots=True)
class RunRecord:
    run_id: str
    spec: JobSpec
    state: RunState = "queued"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


_run_lock = threading.Lock()
_runs: dict[str, RunRecord] = {}

VALID_JOB_KINDS = {"train_one", "stream", "generate_plasmid", "generate_protein", "validate_plasmid", "pretrain"}
REPLAY_DESCRIPTOR_VERSION = 1


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _artifact_download_url(run_id: str, artifact_id: int) -> str:
    return f"/api/runs/{run_id}/artifacts/{artifact_id}/download"


def _run_to_out(run: Run) -> RunOut:
    artifacts = [
        RunArtifactOut(
            id=a.id,
            phase=a.phase,
            path=a.path,
            label=a.label,
            download_url=_artifact_download_url(run.run_id, a.id),
            created_at=a.created_at,
        )
        for a in sorted(run.artifacts, key=lambda item: item.created_at)
    ]
    result_obj = _json_loads(run.result_json)
    return RunOut(
        run_id=run.run_id,
        user_id=run.user_id,
        kind=run.kind,
        state=run.state,
        message=run.message,
        config=_json_loads(run.config_json),
        result=result_obj or None,
        submitted_at=run.submitted_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        artifacts=artifacts,
    )


def _find_run(db: Session, run_id: str) -> Run | None:
    return db.execute(select(Run).where(Run.run_id == run_id)).scalar_one_or_none()


def _save_run_submission(db: Session, user: User, run_id: str, spec: JobSpec, cfg: dict[str, Any]) -> Run:
    run = _find_run(db, run_id)
    if run and run.state in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
    if not run:
        run = Run(run_id=run_id, user_id=user.id, kind=spec.kind)
        db.add(run)
    run.kind = spec.kind
    run.state = "queued"
    run.config_json = _json_dumps(cfg)
    run.result_json = None
    run.message = "queued"
    run.submitted_at = _utcnow()
    run.started_at = None
    run.finished_at = None
    db.commit()
    db.refresh(run)
    return run


def _mark_run_started(db: Session, run_id: str):
    run = _find_run(db, run_id)
    if not run:
        return
    run.state = "running"
    run.started_at = _utcnow()
    run.message = "running"
    db.commit()


def _record_artifact(db: Session, run_id: str, path: str, phase: str | None = None, label: str | None = None) -> RunArtifact | None:
    run = _find_run(db, run_id)
    if not run:
        return None
    artifact = RunArtifact(run_id=run.id, path=path, phase=phase, label=label)
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def _finalize_run(db: Session, run_id: str, state: RunState, result: dict[str, Any], message: str | None = None):
    run = _find_run(db, run_id)
    if not run:
        return
    run.state = state
    run.message = message or state
    run.result_json = _json_dumps(result)
    run.finished_at = _utcnow()
    db.commit()


def _assert_run_access(run: Run, user: User):
    if user.role != "admin" and run.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")




def _scoped_runs_query(user: User):
    q = select(Run)
    if user.role != "admin":
        q = q.where(Run.user_id == user.id)
    return q

def _extract_manifest_uri(data: dict[str, Any], run_id: str) -> str | None:
    manifest_path = data.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        return f"/api/runs/{run_id}/artifacts/download-by-path?path={quote(manifest_path)}"
    return None


def _parse_job_spec(config: dict[str, Any]) -> tuple[str, JobSpec]:
    cfg = config or {}
    run_id = str(cfg.get("run_id") or cfg.get("manifest_id") or f"web_{uuid4().hex}")
    kind = str(cfg.get("kind", "generate_plasmid"))
    config_path = str(cfg.get("config_path", "config/stream_config.yaml"))
    if kind not in VALID_JOB_KINDS:
        raise HTTPException(status_code=400, detail=f"Unsupported run kind: {kind}")
    if not config_path:
        raise HTTPException(status_code=400, detail="config_path is required")

    reserved = {"run_id", "manifest_id", "kind", "config_path", "params"}
    params = {k: v for k, v in cfg.items() if k not in reserved}
    params.update(dict(cfg.get("params") or {}))
    params.setdefault("run_id", run_id)
    params.setdefault("manifest_id", run_id)

    spec = JobSpec(kind=kind, config_path=config_path, params=params)
    return run_id, spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _extract_seed_info(params: dict[str, Any]) -> dict[str, Any]:
    seed_keys = {
        "seed",
        "random_seed",
        "rng_seed",
        "np_seed",
        "torch_seed",
        "python_seed",
    }
    return {k: params[k] for k in sorted(seed_keys) if k in params}


def _extract_required_parent_artifacts(manifest_path: Path) -> list[dict[str, Any]]:
    payload = _load_json_file(manifest_path)
    required: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def _append_parent(ref: dict[str, Any]):
        raw_path = ref.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return
        resolved = Path(raw_path).expanduser().resolve()
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        exists = resolved.exists() and resolved.is_file()
        required.append(
            {
                "path": str(resolved),
                "sha256": _sha256_file(resolved) if exists else None,
                "size_bytes": resolved.stat().st_size if exists else None,
                "artifact_id": ref.get("artifact_id"),
                "relation": ref.get("relation"),
            }
        )

    run = payload.get("run")
    if isinstance(run, dict):
        for parent in run.get("parents") or []:
            if isinstance(parent, dict):
                _append_parent(parent)

    for artifact in payload.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        for parent in artifact.get("parents") or []:
            if isinstance(parent, dict):
                _append_parent(parent)
    return required


def _build_replay_descriptor(*, cfg: dict[str, Any], spec: JobSpec, result_data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    manifest_path_value = result_data.get("manifest_path")
    manifest_path = Path(str(manifest_path_value)).expanduser().resolve() if isinstance(manifest_path_value, str) and manifest_path_value else None
    explicit_params = dict(spec.params)
    descriptor: dict[str, Any] = {
        "schema_version": REPLAY_DESCRIPTOR_VERSION,
        "run_kind": spec.kind,
        "config_path": spec.config_path,
        "resolved_config_snapshot": result_data.get("config_snapshot"),
        "explicit_params": explicit_params,
        "seed_info": _extract_seed_info(explicit_params),
        "required_input_artifacts": _extract_required_parent_artifacts(manifest_path) if manifest_path and manifest_path.exists() else [],
        "source_run_config": cfg,
    }
    descriptor_path: str | None = None
    if manifest_path:
        descriptor_path = str(manifest_path.with_name(f"{manifest_path.stem}.replay.json"))
        target = Path(descriptor_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(descriptor, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return descriptor, descriptor_path


def _descriptor_hash(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_required_inputs(required_inputs: list[dict[str, Any]]) -> None:
    for item in required_inputs:
        path_value = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise HTTPException(status_code=400, detail="Replay descriptor has an invalid required artifact path")
        target = Path(path_value).expanduser().resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=409, detail=f"Required artifact missing for replay: {target}")
        if isinstance(expected_hash, str) and expected_hash:
            actual_hash = _sha256_file(target)
            if actual_hash != expected_hash:
                raise HTTPException(status_code=409, detail=f"Required artifact hash mismatch for replay: {target}")


def _update_manifest_metadata(manifest_path: str, metadata: dict[str, Any]) -> None:
    if not manifest_path:
        return
    target = Path(manifest_path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return
    payload = _load_json_file(target)
    provenance = payload.get("provenance_metadata")
    if not isinstance(provenance, dict):
        provenance = {}
    replay_meta = provenance.get("replay")
    if not isinstance(replay_meta, dict):
        replay_meta = {}
    replay_meta.update(metadata)
    provenance["replay"] = replay_meta
    payload["provenance_metadata"] = provenance
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _execute_run(*, cfg: dict[str, Any], user: User, db: Session, manifest_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    run_id, spec = _parse_job_spec(cfg)
    record = _upsert_run(run_id, spec)
    _save_run_submission(db, user, run_id, spec, cfg)
    record.state = "running"
    _mark_run_started(db, run_id)
    result = JobEngine(cancel_event=record.cancel_event).run(spec)
    final_state: RunState = "completed" if result.ok else ("canceled" if result.exit_code == 130 else "failed")
    result_data = dict(result.data or {})
    manifest_path = result_data.get("manifest_path") if isinstance(result_data.get("manifest_path"), str) else None
    manifest_uri = _extract_manifest_uri(result_data, run_id)
    descriptor, descriptor_path = _build_replay_descriptor(cfg=cfg, spec=spec, result_data=result_data)
    descriptor_hash = _descriptor_hash(descriptor)
    final_payload = {
        "run_id": run_id,
        "ok": bool(result.ok),
        "state": final_state,
        "message": result.message,
        "manifest_path": manifest_path,
        "manifest_uri": manifest_uri,
        "replay_descriptor": descriptor,
        "replay_descriptor_path": descriptor_path,
        "replay_descriptor_hash": descriptor_hash,
        **result_data,
    }
    with _run_lock:
        record.state = final_state
        record.result = final_payload
    if manifest_path:
        _record_artifact(db, run_id, manifest_path, phase="manifest", label="Run manifest")
    if descriptor_path:
        _record_artifact(db, run_id, descriptor_path, phase="provenance", label="Replay descriptor")
    if manifest_path and manifest_metadata:
        _update_manifest_metadata(manifest_path, manifest_metadata)
    _finalize_run(db, run_id, final_state, final_payload, result.message)
    return {"ok": result.ok, "message": result.message, "user_id": user.id, "role": user.role, "result": final_payload}


def _load_run_replay_descriptor(source_run: Run) -> tuple[dict[str, Any], str]:
    result_payload = _json_loads(source_run.result_json)
    descriptor = result_payload.get("replay_descriptor")
    descriptor_path = result_payload.get("replay_descriptor_path")
    if isinstance(descriptor, dict):
        return descriptor, _descriptor_hash(descriptor)
    if isinstance(descriptor_path, str) and descriptor_path:
        descriptor_file = Path(descriptor_path).expanduser().resolve()
        if not descriptor_file.exists() or not descriptor_file.is_file():
            raise HTTPException(status_code=400, detail="Replay descriptor file is missing")
        descriptor_payload = _load_json_file(descriptor_file)
        return descriptor_payload, _descriptor_hash(descriptor_payload)
    raise HTTPException(status_code=400, detail="Replay descriptor is unavailable for this run")




def _manifest_path_for_run(run: Run) -> str | None:
    for artifact in sorted(run.artifacts, key=lambda item: item.created_at):
        if artifact.phase == "manifest" and artifact.path:
            return artifact.path
    result_payload = _json_loads(run.result_json)
    manifest_path = result_payload.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        return manifest_path
    return None


def _load_manifest(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return {}
    return _load_json_file(target)


def _lineage_ref_node_id(ref: dict[str, Any]) -> str:
    artifact_id = str(ref.get("artifact_id") or "").strip()
    path_value = str(ref.get("path") or "").strip()
    if artifact_id:
        return f"artifact:{artifact_id}"
    if path_value:
        return f"path:{Path(path_value).expanduser().resolve()}"
    return "lineage:unknown"


def _build_lineage_graph(*, run: Run, depth_limit: int, accessible_runs: list[Run]) -> tuple[list[LineageNodeOut], list[LineageEdgeOut]]:
    run_by_id = {item.run_id: item for item in accessible_runs}
    run_by_manifest_path: dict[str, Run] = {}
    for item in accessible_runs:
        mpath = _manifest_path_for_run(item)
        if not mpath:
            continue
        run_by_manifest_path[str(Path(mpath).expanduser().resolve())] = item

    nodes: dict[str, LineageNodeOut] = {}
    edges: dict[tuple[str, str, str], LineageEdgeOut] = {}

    def ensure_run_node(target_run: Run, depth: int) -> str:
        node_id = f"run:{target_run.run_id}"
        if node_id in nodes:
            if depth < nodes[node_id].depth:
                nodes[node_id].depth = depth
            return node_id
        result_payload = _json_loads(target_run.result_json)
        run_manifest = _load_manifest(_manifest_path_for_run(target_run))
        provenance = run_manifest.get("provenance_metadata") if isinstance(run_manifest.get("provenance_metadata"), dict) else {}
        cfg = provenance.get("config") if isinstance(provenance.get("config"), dict) else {}
        snapshot = None
        cfg_path = cfg.get("path")
        cfg_hash = cfg.get("sha256")
        if isinstance(cfg_path, str) and cfg_path and isinstance(cfg_hash, str) and cfg_hash:
            snapshot = {"path": cfg_path, "sha256": cfg_hash, "format": "json"}
        replay_hash = result_payload.get("replay_descriptor_hash")
        node_hash = replay_hash if isinstance(replay_hash, str) else (cfg_hash if isinstance(cfg_hash, str) else None)
        nodes[node_id] = LineageNodeOut(
            id=node_id,
            kind="run",
            label=target_run.run_id,
            depth=depth,
            run_id=target_run.run_id,
            run_state=target_run.state,
            hash=node_hash,
            config_snapshot=snapshot,
            payload={"kind": target_run.kind, "message": target_run.message, "result": result_payload},
        )
        return node_id

    def ensure_artifact_node(artifact: dict[str, Any], depth: int) -> str:
        artifact_id = str(artifact.get("id") or artifact.get("path") or "").strip()
        node_id = f"artifact:{artifact_id}"
        if node_id in nodes:
            if depth < nodes[node_id].depth:
                nodes[node_id].depth = depth
            return node_id
        nodes[node_id] = LineageNodeOut(
            id=node_id,
            kind="artifact",
            label=str(artifact.get("id") or Path(str(artifact.get("path") or "artifact")).name),
            depth=depth,
            artifact_id=str(artifact.get("id") or "") or None,
            artifact_type=str(artifact.get("type") or artifact.get("role") or "") or None,
            path=str(artifact.get("path") or "") or None,
            hash=str(artifact.get("sha256") or "") or None,
            payload=dict(artifact),
        )
        return node_id

    def ensure_ref_node(ref: dict[str, Any], depth: int) -> str:
        node_id = _lineage_ref_node_id(ref)
        if node_id in nodes:
            if depth < nodes[node_id].depth:
                nodes[node_id].depth = depth
            return node_id
        path_value = str(ref.get("path") or "").strip()
        artifact_id = str(ref.get("artifact_id") or "").strip()
        nodes[node_id] = LineageNodeOut(
            id=node_id,
            kind="artifact_ref",
            label=artifact_id or Path(path_value).name or "lineage_ref",
            depth=depth,
            artifact_id=artifact_id or None,
            path=path_value or None,
            relation=str(ref.get("relation") or "") or None,
            payload=dict(ref),
        )
        return node_id

    queue = deque([(run.run_id, 0)])
    visited: set[str] = set()
    while queue:
        current_run_id, depth = queue.popleft()
        if current_run_id in visited or depth > depth_limit:
            continue
        visited.add(current_run_id)
        current_run = run_by_id.get(current_run_id)
        if not current_run:
            continue
        current_run_node = ensure_run_node(current_run, depth)
        manifest = _load_manifest(_manifest_path_for_run(current_run))
        run_section = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}

        for parent in run_section.get("parents") or []:
            if not isinstance(parent, dict):
                continue
            source_node = ensure_ref_node(parent, depth + 1)
            relation = str(parent.get("relation") or "parent")
            edges[(source_node, current_run_node, relation)] = LineageEdgeOut(source=source_node, target=current_run_node, relation=relation)
            parent_path = parent.get("path")
            if isinstance(parent_path, str) and parent_path:
                parent_run = run_by_manifest_path.get(str(Path(parent_path).expanduser().resolve()))
                if parent_run:
                    parent_run_node = ensure_run_node(parent_run, depth + 1)
                    edges[(parent_run_node, current_run_node, relation)] = LineageEdgeOut(source=parent_run_node, target=current_run_node, relation=relation)
                    if depth + 1 <= depth_limit:
                        queue.append((parent_run.run_id, depth + 1))

        for artifact in manifest.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            artifact_node = ensure_artifact_node(artifact, depth + 1)
            edges[(current_run_node, artifact_node, "emits.artifact")] = LineageEdgeOut(source=current_run_node, target=artifact_node, relation="emits.artifact")
            for parent in artifact.get("parents") or []:
                if not isinstance(parent, dict):
                    continue
                source_node = ensure_ref_node(parent, depth + 2)
                relation = str(parent.get("relation") or "artifact_parent")
                edges[(source_node, artifact_node, relation)] = LineageEdgeOut(source=source_node, target=artifact_node, relation=relation)
                parent_path = parent.get("path")
                if isinstance(parent_path, str) and parent_path:
                    parent_run = run_by_manifest_path.get(str(Path(parent_path).expanduser().resolve()))
                    if parent_run:
                        parent_run_node = ensure_run_node(parent_run, depth + 1)
                        edges[(parent_run_node, artifact_node, relation)] = LineageEdgeOut(source=parent_run_node, target=artifact_node, relation=relation)
                        if depth + 1 <= depth_limit:
                            queue.append((parent_run.run_id, depth + 1))

    return list(nodes.values()), list(edges.values())


def _filter_lineage_graph(
    *,
    nodes: list[LineageNodeOut],
    edges: list[LineageEdgeOut],
    root_run_id: str,
    artifact_type_filter: str | None,
    run_state_filter: set[str],
) -> tuple[list[LineageNodeOut], list[LineageEdgeOut]]:
    filtered: dict[str, LineageNodeOut] = {}
    artifact_filter = (artifact_type_filter or "").strip().lower()
    for node in nodes:
        keep = True
        if node.kind == "artifact" and artifact_filter:
            keep = (node.artifact_type or "").lower() == artifact_filter
        if node.kind == "run" and run_state_filter and node.run_id != root_run_id:
            keep = (node.run_state or "").lower() in run_state_filter
        if node.run_id == root_run_id:
            keep = True
        if keep:
            filtered[node.id] = node

    filtered_edges: list[LineageEdgeOut] = []
    for edge in edges:
        if edge.source in filtered and edge.target in filtered:
            filtered_edges.append(edge)

    connected = {f"run:{root_run_id}"}
    for edge in filtered_edges:
        connected.add(edge.source)
        connected.add(edge.target)
    return [node for node in filtered.values() if node.id in connected], filtered_edges




def _dataset_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config"


def _count_dataset_sequences(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _dataset_split_metadata(dataset_id: str, sequence_count: int) -> list[DatasetSplitOut]:
    if sequence_count <= 0:
        return []
    if "_" in dataset_id:
        prefix = dataset_id.split("_", 1)[0]
        siblings = sorted(_dataset_config_dir().glob(f"{prefix}_*.txt"))
        if len(siblings) >= 2:
            return [DatasetSplitOut(name="catalog_group", count=len(siblings))]
    train_count = int(round(sequence_count * 0.8))
    val_count = int(round(sequence_count * 0.1))
    test_count = max(sequence_count - train_count - val_count, 0)
    return [
        DatasetSplitOut(name="train", count=train_count),
        DatasetSplitOut(name="validation", count=val_count),
        DatasetSplitOut(name="test", count=test_count),
    ]


def _build_dataset_catalog_item(path: Path) -> DatasetCatalogItemOut:
    dataset_id = path.stem
    source = dataset_id.split("_", 1)[0]
    sequence_count = _count_dataset_sequences(path)
    split_metadata = _dataset_split_metadata(dataset_id, sequence_count)
    tags = sorted({source, f"size:{sequence_count}", "manifest", "config"})
    last_updated_hash = _sha256_file(path)
    return DatasetCatalogItemOut(
        dataset_id=dataset_id,
        source=source,
        sequence_count=sequence_count,
        split_metadata=split_metadata,
        tags=tags,
        last_updated_hash=last_updated_hash,
    )


def _load_dataset_catalog() -> list[DatasetCatalogItemOut]:
    cfg = _dataset_config_dir()
    if not cfg.exists() or not cfg.is_dir():
        return []
    items: list[DatasetCatalogItemOut] = []
    for path in sorted(cfg.glob("*.txt")):
        items.append(_build_dataset_catalog_item(path))
    return items


def _find_dataset_manifest(dataset_id: str) -> Path:
    target = (_dataset_config_dir() / f"{dataset_id}.txt").resolve()
    cfg_root = _dataset_config_dir().resolve()
    if cfg_root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid dataset id")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return target


def _dataset_preview_rows(path: Path, limit: int = 25) -> tuple[list[str], int]:
    rows: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            total += 1
            if len(rows) < max(1, min(limit, 100)):
                rows.append(value)
    return rows, total


def _upsert_run(run_id: str, spec: JobSpec) -> RunRecord:
    with _run_lock:
        existing = _runs.get(run_id)
        if existing and existing.state in {"queued", "running"}:
            raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
        record = RunRecord(run_id=run_id, spec=spec)
        _runs[run_id] = record
    return record


def _stop_run_record(run_id: str) -> bool:
    with _run_lock:
        record = _runs.get(run_id)
        if not record:
            return False
        record.cancel_event.set()
        if record.state == "queued":
            record.state = "canceled"
            record.result = {"run_id": run_id, "canceled": True, "manifest_path": None, "manifest_uri": None}
    return True


def _utcnow() -> datetime:
    return datetime.utcnow()


def _set_session_cookie(response: Response, raw_token: str):
    response.set_cookie(
        key=settings.session_cookie_name,
        value=raw_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.session_ttl_hours * 3600,
        path="/",
        domain=settings.cookie_domain,
    )


def _clear_session_cookie(response: Response):
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        domain=settings.cookie_domain,
    )


def _issue_session(db: Session, user: User, request: Request) -> str:
    raw = make_session_token()
    token_hash = hash_session_token(raw)
    expires_at = _utcnow() + timedelta(hours=settings.session_ttl_hours)

    sess = UserSession(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(sess)
    user.last_login_at = _utcnow()
    db.commit()
    return raw


def _build_verification_link(raw_token: str) -> str:
    return f"{settings.email_verification_base_url}?{urlencode({'token': raw_token})}"


def _build_password_reset_link(raw_token: str) -> str:
    return f"{settings.password_reset_base_url}?{urlencode({'token': raw_token})}"


def _send_email(recipient: str, subject: str, body: str):
    if settings.mail_provider.lower() == "smtp":
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.mail_from_email
        msg["To"] = recipient
        msg.set_content(body)

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
            if settings.smtp_use_tls:
                smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password or "")
            smtp.send_message(msg)
        return

    print(f"[auth] {subject} to {recipient}: {body}")


def _send_verification_email(recipient: str, raw_token: str):
    link = _build_verification_link(raw_token)
    _send_email(
        recipient,
        "Verify your Perceptrome account",
        "Welcome to Perceptrome!\n\n"
        "Use the following link to verify your email:\n"
        f"{link}\n\n"
        "If you did not sign up, you can ignore this email.",
    )


def _send_password_reset_email(recipient: str, raw_token: str):
    link = _build_password_reset_link(raw_token)
    _send_email(
        recipient,
        "Reset your Perceptrome password",
        "We received a request to reset your Perceptrome password.\n\n"
        "Use the following link to reset your password:\n"
        f"{link}\n\n"
        "If you did not request this, you can ignore this email.",
    )


def _issue_auth_token(db: Session, user: User, purpose: str, ttl_minutes: int) -> str:
    raw = make_session_token()
    token = AuthToken(
        user_id=user.id,
        purpose=purpose,
        token_hash=hash_session_token(raw),
        expires_at=_utcnow() + timedelta(minutes=ttl_minutes),
    )
    db.add(token)
    db.commit()
    return raw


def _issue_email_verification_token(db: Session, user: User) -> str:
    raw = _issue_auth_token(db, user, purpose="email_verification", ttl_minutes=settings.email_verification_token_ttl_minutes)
    user.email_verification_sent_at = _utcnow()
    db.commit()
    return raw


def _issue_password_reset_token(db: Session, user: User) -> str:
    return _issue_auth_token(db, user, purpose="password_reset", ttl_minutes=settings.password_reset_token_ttl_minutes)


def _revoke_session_by_cookie(db: Session, raw_cookie: str | None):
    if not raw_cookie:
        return
    token_hash = hash_session_token(raw_cookie)
    stmt = (
        select(UserSession)
        .where(UserSession.token_hash == token_hash)
        .where(UserSession.revoked_at.is_(None))
    )
    sess = db.execute(stmt).scalar_one_or_none()
    if sess:
        sess.revoked_at = _utcnow()
        db.commit()


def _metric_inc(name: str):
    _auth_metrics[name] += 1


def _structured_auth_log(event: str, **fields):
    payload = {"event": event, **fields}
    _auth_logger.warning(json.dumps(payload, sort_keys=True))


def _raise_auth_429(retry_after_seconds: int, reason: str):
    detail = {
        "message": "Too many authentication attempts. Please retry later.",
        "retry_after_seconds": retry_after_seconds,
        "reason": reason,
    }
    raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after_seconds)})


def _ensure_users_columns():
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return

    cols = {c["name"] for c in insp.get_columns("users")}
    statements = {
        "must_change_password": "ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT 0",
        "email_verified_at": "ALTER TABLE users ADD COLUMN email_verified_at DATETIME NULL",
        "email_verification_sent_at": "ALTER TABLE users ADD COLUMN email_verification_sent_at DATETIME NULL",
        "failed_login_count": "ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0",
        "locked_until": "ALTER TABLE users ADD COLUMN locked_until DATETIME NULL",
    }

    for name, stmt in statements.items():
        if name not in cols:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            print(f"[auth] migrated users.{name}")


def _bootstrap_admin():
    if not settings.bootstrap_admin_email or not settings.bootstrap_admin_password:
        return

    db = SessionLocal()
    try:
        email = settings.bootstrap_admin_email.lower().strip()
        existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if existing:
            return

        admin = User(
            email=email,
            password_hash=hash_password(settings.bootstrap_admin_password),
            role="admin",
            is_active=True,
            must_change_password=True,
            email_verified_at=_utcnow(),
        )
        db.add(admin)
        db.commit()
        print(f"[auth] bootstrapped admin user: {admin.email}")
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    _ensure_users_columns()
    _bootstrap_admin()


@app.get("/api/health")
def health():
    return {"ok": True, "service": "perceptrome-api"}


@app.post("/api/auth/register", response_model=UserOut)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if not settings.allow_self_register:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Self registration is disabled")

    email = payload.email.lower().strip()
    username = payload.username.strip() if payload.username else None

    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    if username and db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=email,
        username=username,
        password_hash=hash_password(payload.password),
        role="user",
        is_active=True,
        must_change_password=False,
        email_verified_at=None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    raw_token = _issue_email_verification_token(db, user)
    _send_verification_email(user.email, raw_token)

    db.refresh(user)
    return UserOut.from_model(user)


@app.post("/api/auth/verify-email", response_model=MessageOut)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    token_hash = hash_session_token(payload.token)
    token = db.execute(
        select(AuthToken)
        .where(AuthToken.purpose == "email_verification")
        .where(AuthToken.token_hash == token_hash)
    ).scalar_one_or_none()

    if not token:
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Invalid verification token")

    if token.used_at is not None:
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Verification token already used")

    if token.expires_at <= _utcnow():
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Verification token expired")

    user = db.get(User, token.user_id)
    if not user:
        _metric_inc("verification_failures")
        raise HTTPException(status_code=400, detail="Invalid verification token")

    token.used_at = _utcnow()
    if user.email_verified_at is not None:
        db.commit()
        return MessageOut(message="Email already verified")

    user.email_verified_at = _utcnow()
    db.commit()
    return MessageOut(message="Email verified")


@app.post("/api/auth/resend-verification", response_model=MessageOut)
def resend_verification(payload: ResendVerificationRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if not user:
        return MessageOut(message="If this email is registered, a verification email has been sent")

    if user.email_verified_at is not None:
        return MessageOut(message="Email already verified")

    if user.email_verification_sent_at is not None:
        elapsed = (_utcnow() - user.email_verification_sent_at).total_seconds()
        if elapsed < settings.email_verification_resend_cooldown_seconds:
            _raise_auth_429(max(1, int(settings.email_verification_resend_cooldown_seconds - elapsed)), "verification_resend_cooldown")

    raw_token = _issue_email_verification_token(db, user)
    _send_verification_email(user.email, raw_token)
    return MessageOut(message="Verification email sent")


@app.post("/api/auth/forgot-password", response_model=MessageOut)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if user:
        raw_token = _issue_password_reset_token(db, user)
        _send_password_reset_email(user.email, raw_token)

    return MessageOut(message="If this email is registered, a password reset email has been sent")


@app.post("/api/auth/reset-password", response_model=MessageOut)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash = hash_session_token(payload.token)
    token = db.execute(
        select(AuthToken)
        .where(AuthToken.purpose == "password_reset")
        .where(AuthToken.token_hash == token_hash)
    ).scalar_one_or_none()

    if not token:
        raise HTTPException(status_code=400, detail="Invalid password reset token")

    if token.used_at is not None:
        raise HTTPException(status_code=400, detail="Password reset token already used")

    now = _utcnow()
    if token.expires_at <= now:
        raise HTTPException(status_code=400, detail="Password reset token expired")

    user = db.get(User, token.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid password reset token")

    complexity_error = password_complexity_error(payload.new_password)
    if complexity_error:
        raise HTTPException(status_code=400, detail=complexity_error)

    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    token.used_at = now

    active_sessions = db.execute(
        select(UserSession)
        .where(UserSession.user_id == user.id)
        .where(UserSession.revoked_at.is_(None))
        .where(UserSession.expires_at > now)
    ).scalars().all()
    for sess in active_sessions:
        sess.revoked_at = now

    db.commit()
    return MessageOut(message="Password reset successful")


@app.post("/api/auth/login", response_model=UserOut)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    ip = request.client.host if request.client else "unknown"
    now = _utcnow()
    rl_status = login_attempt_store.check_and_record(db=db, ip=ip, email=email, now=now)
    if rl_status.limited:
        _structured_auth_log("login_rate_limited", ip=ip, email=email, scope=rl_status.scope, retry_after_seconds=rl_status.retry_after_seconds)
        _metric_inc("lockouts")
        _raise_auth_429(rl_status.retry_after_seconds, f"rate_limit_{rl_status.scope}")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if user and user.locked_until and user.locked_until > now:
        retry_after = max(1, int((user.locked_until - now).total_seconds()))
        _structured_auth_log("login_user_locked", ip=ip, email=email, user_id=user.id, retry_after_seconds=retry_after)
        _metric_inc("lockouts")
        _raise_auth_429(retry_after, "user_locked")

    if not user or not user.is_active or not verify_password(payload.password, user.password_hash):
        if user:
            user.failed_login_count = (user.failed_login_count or 0) + 1
            if user.failed_login_count >= settings.login_lockout_threshold:
                user.locked_until = now + timedelta(seconds=settings.login_lockout_seconds)
                _metric_inc("lockouts")
            elif user.failed_login_count >= 2:
                backoff = min(
                    settings.login_backoff_max_seconds,
                    settings.login_backoff_base_seconds * (2 ** (user.failed_login_count - 2)),
                )
                user.locked_until = now + timedelta(seconds=backoff)
            db.commit()
            _structured_auth_log(
                "login_failed",
                ip=ip,
                email=email,
                user_id=user.id,
                failed_login_count=user.failed_login_count,
                locked_until=user.locked_until.isoformat() if user.locked_until else None,
            )
        else:
            _structured_auth_log("login_failed", ip=ip, email=email, user_id=None)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.email_verified_at is None:
        _structured_auth_log("login_email_unverified", ip=ip, email=email, user_id=user.id)
        raise HTTPException(status_code=403, detail="Email verification required")

    if user.failed_login_count or user.locked_until is not None:
        user.failed_login_count = 0
        user.locked_until = None
        db.commit()
        _metric_inc("resets")
        _structured_auth_log("login_failure_state_reset", ip=ip, email=email, user_id=user.id)

    _revoke_session_by_cookie(db, request.cookies.get(settings.session_cookie_name))

    raw_session = _issue_session(db, user, request)
    _set_session_cookie(response, raw_session)
    return UserOut.from_model(user)


@app.post("/api/auth/logout", response_model=MessageOut)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    _revoke_session_by_cookie(db, request.cookies.get(settings.session_cookie_name))
    _clear_session_cookie(response)
    return MessageOut(message="Logged out")


@app.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.from_model(user)


@app.post("/api/auth/change-password", response_model=MessageOut)
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different")

    complexity_error = password_complexity_error(payload.new_password)
    if complexity_error:
        raise HTTPException(status_code=400, detail=complexity_error)

    now = _utcnow()
    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False

    current_cookie = request.cookies.get(settings.session_cookie_name)
    current_session_hash = hash_session_token(current_cookie) if current_cookie else None
    active_sessions = db.execute(
        select(UserSession)
        .where(UserSession.user_id == user.id)
        .where(UserSession.revoked_at.is_(None))
        .where(UserSession.expires_at > now)
    ).scalars().all()
    for sess in active_sessions:
        if sess.token_hash != current_session_hash:
            sess.revoked_at = now

    db.commit()
    return MessageOut(message="Password changed")


@app.post("/api/runs/start")
def start_run(payload: RunStartRequest | None = None, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    cfg = dict((payload.config if payload else {}) or {})
    return _execute_run(cfg=cfg, user=user, db=db)


@app.post("/api/runs/{run_id}/replay")
def replay_run(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    source_run = _find_run(db, run_id)
    if not source_run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(source_run, user)

    descriptor, descriptor_hash = _load_run_replay_descriptor(source_run)
    required_inputs = descriptor.get("required_input_artifacts")
    if not isinstance(required_inputs, list):
        raise HTTPException(status_code=400, detail="Replay descriptor is invalid: required_input_artifacts must be a list")
    if any(not isinstance(item, dict) for item in required_inputs):
        raise HTTPException(status_code=400, detail="Replay descriptor is invalid: each required_input_artifacts entry must be an object")
    _validate_required_inputs(required_inputs)

    explicit_params = descriptor.get("explicit_params")
    if not isinstance(explicit_params, dict):
        raise HTTPException(status_code=400, detail="Replay descriptor is invalid: explicit_params must be an object")

    replay_run_id = f"replay_{uuid4().hex}"
    replay_cfg = {
        "run_id": replay_run_id,
        "manifest_id": replay_run_id,
        "kind": descriptor.get("run_kind"),
        "config_path": descriptor.get("config_path"),
        "params": explicit_params,
    }
    metadata = {"replayed_from_run_id": run_id, "descriptor_hash": descriptor_hash}
    return _execute_run(cfg=replay_cfg, user=user, db=db, manifest_metadata=metadata)


@app.get("/api/runs", response_model=list[RunOut])
def list_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 50):
    q = _scoped_runs_query(user).order_by(Run.submitted_at.desc()).limit(max(1, min(limit, 200)))
    runs = db.execute(q).scalars().all()
    return [_run_to_out(run) for run in runs]




@app.get("/api/runs/summary", response_model=RunSummaryOut)
def runs_summary(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    runs = db.execute(_scoped_runs_query(user).order_by(Run.submitted_at.desc()).limit(1000)).scalars().all()
    state_counts = Counter(run.state for run in runs)
    latest_failed = next((run for run in runs if run.state == "failed"), None)
    return RunSummaryOut(
        total_runs=len(runs),
        state_counts=dict(state_counts),
        queued=state_counts.get("queued", 0),
        running=state_counts.get("running", 0),
        completed=state_counts.get("completed", 0),
        failed=state_counts.get("failed", 0),
        canceled=state_counts.get("canceled", 0),
        latest_failed_run_id=latest_failed.run_id if latest_failed else None,
        latest_failed_at=latest_failed.finished_at if latest_failed else None,
    )


@app.get("/api/runs/active", response_model=RunsBoardOut)
def active_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 12):
    q = (
        _scoped_runs_query(user)
        .where(Run.state.in_(["queued", "running"]))
        .order_by(Run.submitted_at.desc())
        .limit(max(1, min(limit, 100)))
    )
    runs = db.execute(q).scalars().all()
    return RunsBoardOut(generated_at=_utcnow(), runs=[_run_to_out(run) for run in runs])


@app.get("/api/runs/failures", response_model=RunsBoardOut)
def failed_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 12):
    q = (
        _scoped_runs_query(user)
        .where(Run.state == "failed")
        .order_by(Run.finished_at.desc(), Run.submitted_at.desc())
        .limit(max(1, min(limit, 100)))
    )
    runs = db.execute(q).scalars().all()
    return RunsBoardOut(generated_at=_utcnow(), runs=[_run_to_out(run) for run in runs])

@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    return _run_to_out(run)




@app.get("/api/runs/{run_id}/lineage", response_model=RunLineageOut)
def get_run_lineage(
    run_id: str,
    depth: int = 2,
    artifact_type: str | None = None,
    run_state: str | None = None,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)

    q = _scoped_runs_query(user).order_by(Run.submitted_at.desc()).limit(400)
    accessible_runs = db.execute(q).scalars().all()

    depth_limit = max(0, min(depth, 6))
    nodes, edges = _build_lineage_graph(run=run, depth_limit=depth_limit, accessible_runs=accessible_runs)
    states = {item.strip().lower() for item in (run_state or "").split(",") if item.strip()}
    nodes, edges = _filter_lineage_graph(
        nodes=nodes,
        edges=edges,
        root_run_id=run_id,
        artifact_type_filter=artifact_type,
        run_state_filter=states,
    )

    return RunLineageOut(
        run_id=run_id,
        depth_limit=depth_limit,
        artifact_type_filter=artifact_type,
        run_state_filter=sorted(states),
        nodes=nodes,
        edges=edges,
    )


@app.get("/api/runs/{run_id}/artifacts", response_model=list[RunArtifactOut])
def list_run_artifacts(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    return _run_to_out(run).artifacts


@app.get("/api/runs/{run_id}/artifacts/{artifact_id}/download")
def download_artifact(run_id: str, artifact_id: int, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    artifact = db.execute(select(RunArtifact).where(RunArtifact.id == artifact_id).where(RunArtifact.run_id == run.id)).scalar_one_or_none()
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    target = Path(artifact.path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)


@app.get("/api/runs/{run_id}/artifacts/download-by-path")
def download_artifact_by_path(run_id: str, path: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = _find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    _assert_run_access(run, user)
    target = Path(path).expanduser().resolve()
    known_paths = {Path(a.path).expanduser().resolve() for a in run.artifacts}
    if target not in known_paths:
        raise HTTPException(status_code=403, detail="Artifact path not permitted")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)


@app.get("/api/datasets", response_model=list[DatasetCatalogItemOut])
def list_datasets(user: User = Depends(get_current_user_strict)):
    _ = user
    return _load_dataset_catalog()


@app.get("/api/datasets/{dataset_id}", response_model=DatasetDetailOut)
def get_dataset(dataset_id: str, user: User = Depends(get_current_user_strict)):
    _ = user
    path = _find_dataset_manifest(dataset_id)
    item = _build_dataset_catalog_item(path)
    return DatasetDetailOut(**item.model_dump(), manifest_path=str(path))


@app.get("/api/datasets/{dataset_id}/preview", response_model=DatasetPreviewOut)
def preview_dataset(dataset_id: str, limit: int = 25, user: User = Depends(get_current_user_strict)):
    _ = user
    path = _find_dataset_manifest(dataset_id)
    preview, total_rows = _dataset_preview_rows(path, limit=limit)
    return DatasetPreviewOut(
        dataset_id=dataset_id,
        source=dataset_id.split("_", 1)[0],
        preview=preview,
        total_rows=total_rows,
    )


@app.get("/api/admin/users")
def list_users(
    _admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    users = db.execute(select(User).order_by(User.created_at.desc())).scalars().all()
    return [UserOut.from_model(u).model_dump() for u in users]


@app.post("/api/admin/users", response_model=UserOut)
def create_user_admin(
    payload: AdminCreateUserRequest,
    _admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    email = payload.email.lower().strip()
    username = payload.username.strip() if payload.username else None
    role = payload.role.strip().lower()

    if role not in {"admin", "user"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    if username and db.execute(select(User).where(User.username == username)).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=email,
        username=username,
        password_hash=hash_password(payload.password),
        role=role,
        is_active=payload.is_active,
        must_change_password=payload.must_change_password,
        email_verified_at=_utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut.from_model(user)


def _ws_auth_user(websocket: WebSocket, db: Session) -> User | None:
    raw = websocket.cookies.get(settings.session_cookie_name)
    if not raw:
        return None

    token_hash = hash_session_token(raw)
    now = _utcnow()

    sess = (
        db.execute(
            select(UserSession)
            .where(UserSession.token_hash == token_hash)
            .where(UserSession.revoked_at.is_(None))
            .where(UserSession.expires_at > now)
        )
        .scalar_one_or_none()
    )
    if not sess:
        return None

    user = db.get(User, sess.user_id)
    if not user or not user.is_active:
        return None
    return user


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    db = SessionLocal()
    try:
        user = _ws_auth_user(websocket, db)
        if not user:
            await websocket.close(code=4401)
            return

        if user.must_change_password:
            await websocket.close(code=4403)  # password change required
            return

        await websocket.accept()

        await websocket.send_text(
            json.dumps(
                {
                    "type": "status",
                    "status": f"authenticated as {user.email}",
                    "user_id": user.id,
                    "role": user.role,
                }
            )
        )

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "start_run":
                cfg = dict(msg.get("config", {}) or {})
                run_id, spec = _parse_job_spec(cfg)
                try:
                    record = _upsert_run(run_id, spec)
                    _save_run_submission(db, user, run_id, spec, cfg)
                except HTTPException as exc:
                    await websocket.send_text(json.dumps({"type": "error", "run_id": run_id, "state": "failed", "message": str(exc.detail)}))
                    continue

                await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": "queued", "status": "queued", "phase": "queued", "progress": 0.0}))

                queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _sink(ev: JobEvent):
                    payload: dict[str, Any] = {"type": "log", "run_id": run_id, "phase": ev.stage, "line": ev.message, "data": ev.data}
                    if ev.stage == "train" and "loss" in ev.data:
                        payload = {"type": "metric", "run_id": run_id, "phase": ev.stage, "name": "loss", "value": float(ev.data["loss"]), "step": ev.data.get("epoch")}
                    if ev.stage == "validate" and ev.data:
                        payload = {"type": "validation-summary", "run_id": run_id, "phase": ev.stage, "summary": ev.data}
                    if "progress" in ev.data and isinstance(ev.data.get("progress"), (int, float)):
                        payload = {"type": "progress", "run_id": run_id, "state": "running", "phase": ev.stage, "progress": float(ev.data["progress"])}
                    elif ev.stage in {"start", "done", "stream_step", "generate", "validate", "pretrain", "encode", "train", "manifest"}:
                        payload = {"type": "phase", "run_id": run_id, "state": "running", "phase": ev.stage, "status": ev.message, "data": ev.data}
                    if "path" in ev.data and isinstance(ev.data.get("path"), str):
                        with SessionLocal() as tx:
                            artifact = _record_artifact(tx, run_id, ev.data["path"], phase=ev.stage)
                        artifact_payload = {"path": ev.data["path"]}
                        if artifact:
                            artifact_payload["download_url"] = _artifact_download_url(run_id, artifact.id)
                            loop.call_soon_threadsafe(queue.put_nowait, {"type": "checkpoint", "run_id": run_id, "phase": ev.stage, "path": ev.data["path"], "download_url": artifact_payload["download_url"]})
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "artifact-available", "run_id": run_id, "state": "running", "artifact": artifact_payload, "phase": ev.stage})
                    loop.call_soon_threadsafe(queue.put_nowait, payload)

                async def _forward_events(task: asyncio.Task[Any]):
                    while True:
                        if task.done() and queue.empty():
                            break
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        await websocket.send_text(json.dumps(item))

                record.state = "running"
                _mark_run_started(db, run_id)
                await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": "running", "status": "running", "phase": "start", "progress": 0.01}))

                run_task = asyncio.create_task(asyncio.to_thread(lambda: JobEngine(event_sink=_sink, cancel_event=record.cancel_event).run(spec)))
                forward_task = asyncio.create_task(_forward_events(run_task))
                result = await run_task
                await forward_task

                final_state: RunState = "completed" if result.ok else ("canceled" if result.exit_code == 130 else "failed")
                result_data = dict(result.data or {})
                manifest_path = result_data.get("manifest_path") if isinstance(result_data.get("manifest_path"), str) else None
                manifest_uri = _extract_manifest_uri(result_data, run_id)
                final_result = {
                    "run_id": run_id,
                    "ok": bool(result.ok),
                    "state": final_state,
                    "message": result.message,
                    "manifest_path": manifest_path,
                    "manifest_uri": manifest_uri,
                    **result_data,
                }
                with _run_lock:
                    record.state = final_state
                    record.result = final_result

                if manifest_path:
                    _record_artifact(db, run_id, manifest_path, phase="manifest", label="Run manifest")
                _finalize_run(db, run_id, final_state, final_result, result.message)

                await websocket.send_text(json.dumps({"type": "result", "run_id": run_id, "state": final_state, "result": final_result}))
                await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": final_state, "status": final_state, "phase": "done", "progress": (1.0 if final_state == "completed" else 0.0)}))

            elif msg.get("type") == "stop_run":
                run_id = str(msg.get("run_id") or "")
                if not run_id:
                    with _run_lock:
                        active = [rid for rid, rec in _runs.items() if rec.state in {"queued", "running"}]
                    run_id = active[-1] if active else ""
                if not run_id or not _stop_run_record(run_id):
                    await websocket.send_text(json.dumps({"type": "error", "state": "failed", "message": "run_id not found"}))
                else:
                    _finalize_run(db, run_id, "canceled", {"run_id": run_id, "canceled": True}, "Cancellation requested")
                    await websocket.send_text(json.dumps({"type": "run_stopped", "run_id": run_id, "state": "canceled", "message": "Cancellation requested"}))

            elif msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "ts": msg.get("ts")}))

            else:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "status": f"echo ({user.role})",
                            "data": msg,
                        }
                    )
                )

    except WebSocketDisconnect:
        pass
    finally:
        db.close()
