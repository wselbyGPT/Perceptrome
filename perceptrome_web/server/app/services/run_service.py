import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import threading
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException, WebSocket
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from perceptrome.jobs import JobEngine, JobEvent, JobSpec

from perceptrome.encoding.bio_ast_export import export_filenames

from ..core.db import SessionLocal
from ..models import Run, RunArtifact, User
from ..schemas import BioASTVisualizationBundleOut, LineageEdgeOut, LineageNodeOut, RunArtifactOut, RunLineageOut, RunOut, RunsBoardOut, RunSummaryOut
from . import audit_service

RunState = Literal["queued", "running", "completed", "failed", "canceled"]
VALID_JOB_KINDS = {"train_one", "stream", "generate_plasmid", "generate_protein", "validate_plasmid", "pretrain"}
REPLAY_DESCRIPTOR_VERSION = 1


@dataclass(slots=True)
class RunRecord:
    run_id: str
    spec: JobSpec
    state: RunState = "queued"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


_run_lock = threading.Lock()
_runs: dict[str, RunRecord] = {}


def utcnow():
    return audit_service.utcnow()


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def artifact_download_url(run_id: str, artifact_id: int) -> str:
    return f"/api/runs/{run_id}/artifacts/{artifact_id}/download"


def run_to_out(run: Run) -> RunOut:
    artifacts = [
        RunArtifactOut(id=a.id, phase=a.phase, path=a.path, label=a.label, download_url=artifact_download_url(run.run_id, a.id), created_at=a.created_at)
        for a in sorted(run.artifacts, key=lambda item: item.created_at)
    ]
    result_obj = json_loads(run.result_json)
    return RunOut(run_id=run.run_id, user_id=run.user_id, kind=run.kind, state=run.state, message=run.message, config=json_loads(run.config_json), result=result_obj or None, submitted_at=run.submitted_at, started_at=run.started_at, finished_at=run.finished_at, artifacts=artifacts)


def find_run(db: Session, run_id: str) -> Run | None:
    return db.execute(select(Run).where(Run.run_id == run_id)).scalar_one_or_none()


def save_run_submission(db: Session, user: User, run_id: str, spec: JobSpec, cfg: dict[str, Any]) -> Run:
    run = find_run(db, run_id)
    if run and run.state in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
    if not run:
        run = Run(run_id=run_id, user_id=user.id, kind=spec.kind)
        db.add(run)
    run.kind = spec.kind
    run.state = "queued"
    run.config_json = json_dumps(cfg)
    run.result_json = None
    run.message = "queued"
    run.submitted_at = utcnow()
    run.started_at = None
    run.finished_at = None
    db.commit()
    db.refresh(run)
    return run


def mark_run_started(db: Session, run_id: str):
    run = find_run(db, run_id)
    if not run:
        return
    run.state = "running"
    run.started_at = utcnow()
    run.message = "running"
    db.commit()


def record_artifact(db: Session, run_id: str, path: str, phase: str | None = None, label: str | None = None) -> RunArtifact | None:
    run = find_run(db, run_id)
    if not run:
        return None
    artifact = RunArtifact(run_id=run.id, path=path, phase=phase, label=label)
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def finalize_run(db: Session, run_id: str, state: RunState, result: dict[str, Any], message: str | None = None):
    run = find_run(db, run_id)
    if not run:
        return
    run.state = state
    run.message = message or state
    run.result_json = json_dumps(result)
    run.finished_at = utcnow()
    db.commit()


def assert_run_access(run: Run, user: User):
    if user.role != "admin" and run.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")


def scoped_runs_query(user: User):
    q = select(Run)
    if user.role != "admin":
        q = q.where(Run.user_id == user.id)
    return q


def extract_manifest_uri(data: dict[str, Any], run_id: str) -> str | None:
    manifest_path = data.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path:
        return f"/api/runs/{run_id}/artifacts/download-by-path?path={quote(manifest_path)}"
    return None


def parse_job_spec(config: dict[str, Any]) -> tuple[str, JobSpec]:
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
    return run_id, JobSpec(kind=kind, config_path=config_path, params=params)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def extract_seed_info(params: dict[str, Any]) -> dict[str, Any]:
    seed_keys = {"seed", "random_seed", "rng_seed", "np_seed", "torch_seed", "python_seed"}
    return {k: params[k] for k in sorted(seed_keys) if k in params}


def extract_required_parent_artifacts(manifest_path: Path) -> list[dict[str, Any]]:
    payload = load_json_file(manifest_path)
    required: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def append_parent(ref: dict[str, Any]):
        raw_path = ref.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return
        resolved = Path(raw_path).expanduser().resolve()
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        exists = resolved.exists() and resolved.is_file()
        required.append({"path": str(resolved), "sha256": sha256_file(resolved) if exists else None, "size_bytes": resolved.stat().st_size if exists else None, "artifact_id": ref.get("artifact_id"), "relation": ref.get("relation")})

    run = payload.get("run")
    if isinstance(run, dict):
        for parent in run.get("parents") or []:
            if isinstance(parent, dict):
                append_parent(parent)
    for artifact in payload.get("artifacts") or []:
        if isinstance(artifact, dict):
            for parent in artifact.get("parents") or []:
                if isinstance(parent, dict):
                    append_parent(parent)
    return required


def build_replay_descriptor(*, cfg: dict[str, Any], spec: JobSpec, result_data: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    manifest_path_value = result_data.get("manifest_path")
    manifest_path = Path(str(manifest_path_value)).expanduser().resolve() if isinstance(manifest_path_value, str) and manifest_path_value else None
    explicit_params = dict(spec.params)
    descriptor = {
        "schema_version": REPLAY_DESCRIPTOR_VERSION,
        "run_kind": spec.kind,
        "config_path": spec.config_path,
        "resolved_config_snapshot": result_data.get("config_snapshot"),
        "explicit_params": explicit_params,
        "seed_info": extract_seed_info(explicit_params),
        "required_input_artifacts": extract_required_parent_artifacts(manifest_path) if manifest_path and manifest_path.exists() else [],
        "source_run_config": cfg,
    }
    descriptor_path = None
    if manifest_path:
        descriptor_path = str(manifest_path.with_name(f"{manifest_path.stem}.replay.json"))
        target = Path(descriptor_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(descriptor, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return descriptor, descriptor_path


def descriptor_hash(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_required_inputs(required_inputs: list[dict[str, Any]]) -> None:
    for item in required_inputs:
        path_value = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise HTTPException(status_code=400, detail="Replay descriptor has an invalid required artifact path")
        target = Path(path_value).expanduser().resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=409, detail=f"Required artifact missing for replay: {target}")
        if isinstance(expected_hash, str) and expected_hash and sha256_file(target) != expected_hash:
            raise HTTPException(status_code=409, detail=f"Required artifact hash mismatch for replay: {target}")


def update_manifest_metadata(manifest_path: str, metadata: dict[str, Any]) -> None:
    if not manifest_path:
        return
    target = Path(manifest_path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        return
    payload = load_json_file(target)
    provenance = payload.get("provenance_metadata") if isinstance(payload.get("provenance_metadata"), dict) else {}
    replay_meta = provenance.get("replay") if isinstance(provenance.get("replay"), dict) else {}
    replay_meta.update(metadata)
    provenance["replay"] = replay_meta
    payload["provenance_metadata"] = provenance
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def upsert_run(run_id: str, spec: JobSpec) -> RunRecord:
    with _run_lock:
        existing = _runs.get(run_id)
        if existing and existing.state in {"queued", "running"}:
            raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
        record = RunRecord(run_id=run_id, spec=spec)
        _runs[run_id] = record
    return record


def stop_run_record(run_id: str) -> bool:
    with _run_lock:
        record = _runs.get(run_id)
        if not record:
            return False
        record.cancel_event.set()
        if record.state == "queued":
            record.state = "canceled"
            record.result = {"run_id": run_id, "canceled": True, "manifest_path": None, "manifest_uri": None}
    return True


def execute_run(*, cfg: dict[str, Any], user: User, db: Session, manifest_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    run_id, spec = parse_job_spec(cfg)
    record = upsert_run(run_id, spec)
    save_run_submission(db, user, run_id, spec, cfg)
    record.state = "running"
    mark_run_started(db, run_id)
    result = JobEngine(cancel_event=record.cancel_event).run(spec)
    final_state: RunState = "completed" if result.ok else ("canceled" if result.exit_code == 130 else "failed")
    result_data = dict(result.data or {})
    manifest_path = result_data.get("manifest_path") if isinstance(result_data.get("manifest_path"), str) else None
    manifest_uri = extract_manifest_uri(result_data, run_id)
    descriptor, descriptor_path = build_replay_descriptor(cfg=cfg, spec=spec, result_data=result_data)
    final_payload = {"run_id": run_id, "ok": bool(result.ok), "state": final_state, "message": result.message, "manifest_path": manifest_path, "manifest_uri": manifest_uri, "replay_descriptor": descriptor, "replay_descriptor_path": descriptor_path, "replay_descriptor_hash": descriptor_hash(descriptor), **result_data}
    with _run_lock:
        record.state = final_state
        record.result = final_payload
    if manifest_path:
        record_artifact(db, run_id, manifest_path, phase="manifest", label="Run manifest")
    if descriptor_path:
        record_artifact(db, run_id, descriptor_path, phase="provenance", label="Replay descriptor")
    if manifest_path and manifest_metadata:
        update_manifest_metadata(manifest_path, manifest_metadata)
    finalize_run(db, run_id, final_state, final_payload, result.message)
    return {"ok": result.ok, "message": result.message, "user_id": user.id, "role": user.role, "result": final_payload}


def load_run_replay_descriptor(source_run: Run) -> tuple[dict[str, Any], str]:
    result_payload = json_loads(source_run.result_json)
    descriptor = result_payload.get("replay_descriptor")
    descriptor_path = result_payload.get("replay_descriptor_path")
    if isinstance(descriptor, dict):
        return descriptor, descriptor_hash(descriptor)
    if isinstance(descriptor_path, str) and descriptor_path:
        descriptor_payload = load_json_file(Path(descriptor_path).expanduser().resolve())
        return descriptor_payload, descriptor_hash(descriptor_payload)
    raise HTTPException(status_code=400, detail="Replay descriptor is unavailable for this run")


def manifest_path_for_run(run: Run) -> str | None:
    for artifact in sorted(run.artifacts, key=lambda item: item.created_at):
        if artifact.phase == "manifest" and artifact.path:
            return artifact.path
    manifest_path = json_loads(run.result_json).get("manifest_path")
    return manifest_path if isinstance(manifest_path, str) and manifest_path else None


def load_manifest(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path).expanduser().resolve()
    return load_json_file(target) if target.exists() and target.is_file() else {}


def lineage_ref_node_id(ref: dict[str, Any]) -> str:
    artifact_id = str(ref.get("artifact_id") or "").strip()
    path_value = str(ref.get("path") or "").strip()
    if artifact_id:
        return f"artifact:{artifact_id}"
    if path_value:
        return f"path:{Path(path_value).expanduser().resolve()}"
    return "lineage:unknown"


def build_lineage_graph(*, run: Run, depth_limit: int, accessible_runs: list[Run]) -> tuple[list[LineageNodeOut], list[LineageEdgeOut]]:
    run_by_id = {item.run_id: item for item in accessible_runs}
    run_by_manifest_path = {str(Path(mpath).expanduser().resolve()): item for item in accessible_runs if (mpath := manifest_path_for_run(item))}
    nodes: dict[str, LineageNodeOut] = {}
    edges: dict[tuple[str, str, str], LineageEdgeOut] = {}

    def ensure_run_node(target_run: Run, depth: int) -> str:
        node_id = f"run:{target_run.run_id}"
        if node_id in nodes:
            nodes[node_id].depth = min(nodes[node_id].depth, depth)
            return node_id
        result_payload = json_loads(target_run.result_json)
        run_manifest = load_manifest(manifest_path_for_run(target_run))
        provenance = run_manifest.get("provenance_metadata") if isinstance(run_manifest.get("provenance_metadata"), dict) else {}
        cfg = provenance.get("config") if isinstance(provenance.get("config"), dict) else {}
        cfg_path = cfg.get("path")
        cfg_hash = cfg.get("sha256")
        snapshot = {"path": cfg_path, "sha256": cfg_hash, "format": "json"} if isinstance(cfg_path, str) and cfg_path and isinstance(cfg_hash, str) and cfg_hash else None
        replay_hash = result_payload.get("replay_descriptor_hash")
        node_hash = replay_hash if isinstance(replay_hash, str) else (cfg_hash if isinstance(cfg_hash, str) else None)
        nodes[node_id] = LineageNodeOut(id=node_id, kind="run", label=target_run.run_id, depth=depth, run_id=target_run.run_id, run_state=target_run.state, hash=node_hash, config_snapshot=snapshot, payload={"kind": target_run.kind, "message": target_run.message, "result": result_payload})
        return node_id

    def ensure_artifact_node(artifact: dict[str, Any], depth: int) -> str:
        node_id = f"artifact:{str(artifact.get('id') or artifact.get('path') or '').strip()}"
        if node_id in nodes:
            nodes[node_id].depth = min(nodes[node_id].depth, depth)
            return node_id
        nodes[node_id] = LineageNodeOut(id=node_id, kind="artifact", label=str(artifact.get("id") or Path(str(artifact.get("path") or "artifact")).name), depth=depth, artifact_id=str(artifact.get("id") or "") or None, artifact_type=str(artifact.get("type") or artifact.get("role") or "") or None, path=str(artifact.get("path") or "") or None, hash=str(artifact.get("sha256") or "") or None, payload=dict(artifact))
        return node_id

    def ensure_ref_node(ref: dict[str, Any], depth: int) -> str:
        node_id = lineage_ref_node_id(ref)
        if node_id in nodes:
            nodes[node_id].depth = min(nodes[node_id].depth, depth)
            return node_id
        path_value = str(ref.get("path") or "").strip()
        artifact_id = str(ref.get("artifact_id") or "").strip()
        nodes[node_id] = LineageNodeOut(id=node_id, kind="artifact_ref", label=artifact_id or Path(path_value).name or "lineage_ref", depth=depth, artifact_id=artifact_id or None, path=path_value or None, relation=str(ref.get("relation") or "") or None, payload=dict(ref))
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
        manifest = load_manifest(manifest_path_for_run(current_run))
        run_section = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
        for parent in run_section.get("parents") or []:
            if isinstance(parent, dict):
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
            if isinstance(artifact, dict):
                artifact_node = ensure_artifact_node(artifact, depth + 1)
                edges[(current_run_node, artifact_node, "emits.artifact")] = LineageEdgeOut(source=current_run_node, target=artifact_node, relation="emits.artifact")
                for parent in artifact.get("parents") or []:
                    if isinstance(parent, dict):
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


def filter_lineage_graph(*, nodes: list[LineageNodeOut], edges: list[LineageEdgeOut], root_run_id: str, artifact_type_filter: str | None, run_state_filter: set[str]) -> tuple[list[LineageNodeOut], list[LineageEdgeOut]]:
    filtered = {}
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
    filtered_edges = [edge for edge in edges if edge.source in filtered and edge.target in filtered]
    connected = {f"run:{root_run_id}"}
    for edge in filtered_edges:
        connected.add(edge.source)
        connected.add(edge.target)
    return [node for node in filtered.values() if node.id in connected], filtered_edges


def _artifact_path_for_id(run: Run, artifact_id: int) -> str | None:
    for artifact in run.artifacts:
        if artifact.id == artifact_id:
            return artifact.path
    return None


def _candidate_bio_ast_base_dirs(*, run: Run, selected_path: str | None, accession: str | None, manifest: dict[str, Any], manifest_path: str | None) -> list[Path]:
    filenames = export_filenames()
    expected_names = set(filenames.values())
    candidates: list[Path] = []
    seen: set[Path] = set()

    def append_candidate(path_like: str | None):
        if not isinstance(path_like, str) or not path_like.strip():
            return
        resolved = Path(path_like).expanduser().resolve()
        options = [resolved.parent if resolved.is_file() else resolved]
        if accession:
            options.append((resolved.parent if resolved.is_file() else resolved) / accession)
        for option in options:
            if option in seen:
                continue
            if all((option / name).exists() for name in expected_names):
                seen.add(option)
                candidates.append(option)

    append_candidate(selected_path)
    if manifest_path:
        append_candidate(Path(manifest_path).expanduser().resolve().parent / 'bio_ast')
        if accession:
            append_candidate(Path(manifest_path).expanduser().resolve().parent / 'bio_ast' / accession)
    for artifact in manifest.get('artifacts') or []:
        if not isinstance(artifact, dict):
            continue
        artifact_path = artifact.get('path')
        if accession and accession.lower() not in str(artifact_path or '').lower():
            metadata = artifact.get('metadata') if isinstance(artifact.get('metadata'), dict) else {}
            if str(metadata.get('accession') or '').lower() != accession.lower():
                continue
        append_candidate(artifact_path)
    return candidates


def resolve_bio_ast_bundle(run: Run, *, artifact_id: int | None = None, accession: str | None = None) -> BioASTVisualizationBundleOut:
    selected_path = _artifact_path_for_id(run, artifact_id) if artifact_id is not None else None
    if artifact_id is not None and not selected_path:
        raise HTTPException(status_code=404, detail='Artifact not found')
    manifest_path = manifest_path_for_run(run)
    manifest = load_manifest(manifest_path)
    candidates = _candidate_bio_ast_base_dirs(run=run, selected_path=selected_path, accession=accession, manifest=manifest, manifest_path=manifest_path)
    if not candidates:
        raise HTTPException(status_code=404, detail='Bio-AST visualization bundle not found for this run')
    filenames = export_filenames()
    last_error: Exception | None = None
    for base_dir in candidates:
        try:
            canonical = load_json_file(base_dir / filenames['canonical_ast'])
            storage_map = load_json_file(base_dir / filenames['storage_map'])
            tree = load_json_file(base_dir / filenames['tree_json'])
            graph = load_json_file(base_dir / filenames['graph_json'])
            summary_path = base_dir / filenames['summary_json']
            summary = load_json_file(summary_path) if summary_path.exists() else {}
            resolved_accession = accession or canonical.get('accession') or tree.get('accession') or graph.get('accession')
            return BioASTVisualizationBundleOut(
                accession=str(resolved_accession) if resolved_accession else None,
                resolved_from={
                    'run_id': run.run_id,
                    'artifact_id': artifact_id,
                    'accession': str(resolved_accession) if resolved_accession else None,
                    'manifest_path': manifest_path,
                    'base_dir': str(base_dir),
                },
                canonical=canonical,
                storage_map=storage_map,
                tree=tree,
                graph=graph,
                summary=summary,
            )
        except Exception as exc:  # validation surfaces as 404/400 below
            last_error = exc
    detail = 'Bio-AST visualization bundle is invalid' if last_error else 'Bio-AST visualization bundle not found for this run'
    raise HTTPException(status_code=400 if last_error else 404, detail=detail)


def runs_summary_payload(runs: list[Run]) -> RunSummaryOut:
    state_counts = Counter(run.state for run in runs)
    latest_failed = next((run for run in runs if run.state == "failed"), None)
    return RunSummaryOut(total_runs=len(runs), state_counts=dict(state_counts), queued=state_counts.get("queued", 0), running=state_counts.get("running", 0), completed=state_counts.get("completed", 0), failed=state_counts.get("failed", 0), canceled=state_counts.get("canceled", 0), latest_failed_run_id=latest_failed.run_id if latest_failed else None, latest_failed_at=latest_failed.finished_at if latest_failed else None)


def runs_board(runs: list[Run]) -> RunsBoardOut:
    return RunsBoardOut(generated_at=utcnow(), runs=[run_to_out(run) for run in runs])


def download_artifact_response(run: Run, artifact: RunArtifact):
    target = Path(artifact.path).expanduser().resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)


def download_path_response(run: Run, path: str):
    target = Path(path).expanduser().resolve()
    known_paths = {Path(a.path).expanduser().resolve() for a in run.artifacts}
    if target not in known_paths:
        raise HTTPException(status_code=403, detail="Artifact path not permitted")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path=target, filename=target.name)


async def websocket_run_loop(websocket: WebSocket, user: User, db: Session):
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "status", "status": f"authenticated as {user.email}", "user_id": user.id, "role": user.role}))
    while True:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
        if msg.get("type") == "start_run":
            cfg = dict(msg.get("config", {}) or {})
            run_id, spec = parse_job_spec(cfg)
            try:
                record = upsert_run(run_id, spec)
                save_run_submission(db, user, run_id, spec, cfg)
            except HTTPException as exc:
                await websocket.send_text(json.dumps({"type": "error", "run_id": run_id, "state": "failed", "message": str(exc.detail)}))
                continue
            await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": "queued", "status": "queued", "phase": "queued", "progress": 0.0}))
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def sink(ev: JobEvent):
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
                        artifact = record_artifact(tx, run_id, ev.data["path"], phase=ev.stage)
                    artifact_payload = {"path": ev.data["path"]}
                    if artifact:
                        artifact_payload["download_url"] = artifact_download_url(run_id, artifact.id)
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "checkpoint", "run_id": run_id, "phase": ev.stage, "path": ev.data["path"], "download_url": artifact_payload["download_url"]})
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "artifact-available", "run_id": run_id, "state": "running", "artifact": artifact_payload, "phase": ev.stage})
                loop.call_soon_threadsafe(queue.put_nowait, payload)

            async def forward_events(task: asyncio.Task[Any]):
                while True:
                    if task.done() and queue.empty():
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    await websocket.send_text(json.dumps(item))

            record.state = "running"
            mark_run_started(db, run_id)
            await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": "running", "status": "running", "phase": "start", "progress": 0.01}))
            run_task = asyncio.create_task(asyncio.to_thread(lambda: JobEngine(event_sink=sink, cancel_event=record.cancel_event).run(spec)))
            forward_task = asyncio.create_task(forward_events(run_task))
            result = await run_task
            await forward_task
            final_state: RunState = "completed" if result.ok else ("canceled" if result.exit_code == 130 else "failed")
            result_data = dict(result.data or {})
            manifest_path = result_data.get("manifest_path") if isinstance(result_data.get("manifest_path"), str) else None
            manifest_uri = extract_manifest_uri(result_data, run_id)
            final_result = {"run_id": run_id, "ok": bool(result.ok), "state": final_state, "message": result.message, "manifest_path": manifest_path, "manifest_uri": manifest_uri, **result_data}
            with _run_lock:
                record.state = final_state
                record.result = final_result
            if manifest_path:
                record_artifact(db, run_id, manifest_path, phase="manifest", label="Run manifest")
            finalize_run(db, run_id, final_state, final_result, result.message)
            await websocket.send_text(json.dumps({"type": "result", "run_id": run_id, "state": final_state, "result": final_result}))
            await websocket.send_text(json.dumps({"type": "status", "run_id": run_id, "state": final_state, "status": final_state, "phase": "done", "progress": (1.0 if final_state == "completed" else 0.0)}))
        elif msg.get("type") == "stop_run":
            run_id = str(msg.get("run_id") or "")
            if not run_id:
                with _run_lock:
                    active = [rid for rid, rec in _runs.items() if rec.state in {"queued", "running"}]
                run_id = active[-1] if active else ""
            if not run_id or not stop_run_record(run_id):
                await websocket.send_text(json.dumps({"type": "error", "state": "failed", "message": "run_id not found"}))
            else:
                finalize_run(db, run_id, "canceled", {"run_id": run_id, "canceled": True}, "Cancellation requested")
                await websocket.send_text(json.dumps({"type": "run_stopped", "run_id": run_id, "state": "canceled", "message": "Cancellation requested"}))
        elif msg.get("type") == "ping":
            await websocket.send_text(json.dumps({"type": "pong", "ts": msg.get("ts")}))
        else:
            await websocket.send_text(json.dumps({"type": "status", "status": f"echo ({user.role})", "data": msg}))
